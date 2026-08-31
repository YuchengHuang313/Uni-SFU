#!/usr/bin/env python3
"""Safely resume an interrupted activation-optimization segment sweep.

This entry point preserves completed per-segment outputs, reconstructs the
in-memory/global summaries from their saved artifacts, and then runs only the
requested remaining segment range.
"""

import argparse
import ast
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import optimize_piecewise_activations as pipeline


def _counts(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(v) for v in ast.literal_eval(text)]
    return [int(v) for v in text.split("/")]


def _degrees(value):
    parsed = ast.literal_eval(str(value))
    return [int(v) for v in parsed]


def _load_saved_config(path):
    df = pd.read_csv(path)
    breakpoints = (
        df[df["type"] == "breakpoint"]
        .sort_values("point_idx")["value"]
        .to_numpy(dtype=np.float64)
    )
    degree_rows = df[df["type"] == "degree"].sort_values("segment_idx")
    degrees = degree_rows["value"].astype(int).tolist()
    coeffs = []
    for segment_idx in range(len(degrees)):
        values = (
            df[(df["type"] == "coeff") & (df["segment_idx"] == segment_idx)]
            .sort_values("point_idx")["value"]
            .to_numpy(dtype=np.float64)
        )
        coeffs.append(values)
    return breakpoints, degrees, coeffs


def _eval_mse(root, segment, activation, degree_max, degree_counts):
    count_suffix = "_".join(str(v) for v in degree_counts)
    config_path = (
        root
        / "runs"
        / f"seg_{segment:02d}"
        / "common_area_config"
        / f"{activation}_{segment}seg_d{count_suffix}.csv"
    )
    if (
        not config_path.exists()
        or activation not in pipeline.ACTIVATION_FUNCTIONS
    ):
        return np.nan

    breakpoints, degrees, coeffs = _load_saved_config(config_path)
    domain_min, domain_max = pipeline.helper_activation_domain(activation)
    with torch.no_grad():
        x_eval = torch.linspace(
            domain_min,
            domain_max,
            4096,
            device=pipeline.device,
            dtype=torch.float32,
        )
        y_eval = pipeline.ACTIVATION_FUNCTIONS[activation](x_eval)
        breakpoints_t = torch.tensor(
            breakpoints, device=pipeline.device, dtype=torch.float32
        )
        coeffs_t = [
            torch.tensor(c, device=pipeline.device, dtype=torch.float32)
            for c in coeffs
        ]
        y_hat, _, _ = pipeline.helper_predict_from_coeffs(
            x_eval, breakpoints_t, coeffs_t, degrees
        )
        return float(torch.mean((y_hat - y_eval) ** 2).item())


def _load_timings(root):
    timings = {}
    current_segment = None
    segment_re = re.compile(r"SEGMENT COUNT = (\d+)")
    activation_re = re.compile(
        r"\[Timing\]\[Stage 1\] (.+?): Phase 1=([0-9.]+)s \| "
        r"Phase 2=([0-9.]+)s \(DP=([0-9.]+)s, refine=([0-9.]+)s\) \| "
        r"Activation total=([0-9.]+)s"
    )
    stage2_re = re.compile(
        r"\[Timing\]\[Stage 2\] Phase 3 optimize=([0-9.]+)s \| "
        r"postprocess=([0-9.]+)s \| Total=([0-9.]+)s"
    )
    segment_timing_re = re.compile(
        r"\[Timing\]\[Segment (\d+)\] Stage 1=([0-9.]+)s \| "
        r"Stage 2=([0-9.]+)s \| Total=([0-9.]+)s"
    )

    for log_path in sorted((root / "logs").glob("run_*.log")):
        for line in log_path.read_text(errors="replace").splitlines():
            match = segment_re.search(line)
            if match:
                current_segment = int(match.group(1))
                timings.setdefault(current_segment, {"activations": {}})
                continue
            match = activation_re.search(line)
            if match and current_segment is not None:
                timings.setdefault(current_segment, {"activations": {}})[
                    "activations"
                ][match.group(1)] = {
                    "Phase1RuntimeSec": float(match.group(2)),
                    "Phase2RuntimeSec": float(match.group(3)),
                    "Phase2DPRuntimeSec": float(match.group(4)),
                    "Phase2RefineRuntimeSec": float(match.group(5)),
                    "ActivationRuntimeSec": float(match.group(6)),
                }
                continue
            match = stage2_re.search(line)
            if match and current_segment is not None:
                timings.setdefault(current_segment, {"activations": {}}).update(
                    {
                        "Stage2Phase3OptRuntimeSec": float(match.group(1)),
                        "Stage2PostRuntimeSec": float(match.group(2)),
                        "Stage2RuntimeSec": float(match.group(3)),
                    }
                )
                continue
            match = segment_timing_re.search(line)
            if match:
                segment = int(match.group(1))
                timings.setdefault(segment, {"activations": {}}).update(
                    {
                        "Stage1RuntimeSec": float(match.group(2)),
                        "Stage2RuntimeSec": float(match.group(3)),
                        "SegmentRuntimeSec": float(match.group(4)),
                    }
                )
    return timings


def _phase3_search_stats(summary_path):
    text = summary_path.read_text(errors="replace")
    match = re.search(r"Nodes Explored:\s*(\d+)\s*/\s*(\d+)", text)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def reconstruct_completed_results(root, degree_max, refine_plausible, phase3_algo):
    rows = []
    phase3_outputs = {}
    timings = _load_timings(root)

    for run_dir in sorted((root / "runs").glob("seg_[0-9][0-9]")):
        segment = int(run_dir.name.split("_")[1])
        area_dir = run_dir / "area_err_optimization"
        summary_path = area_dir / f"phase3_summary_{segment}seg.txt"
        comparison_path = area_dir / f"phase2_vs_phase3_comparison_{segment}seg.csv"
        if not summary_path.exists() or not comparison_path.exists():
            continue

        comparison = pd.read_csv(comparison_path)
        actions = comparison["Action"].astype(str)
        final_rows = comparison[
            actions.str.contains(r"final \(post-CD refinement\)", regex=True)
            & (comparison["Function"].astype(str) != "__SEARCH__")
        ]
        initial_rows = comparison[
            actions.str.contains(r"initial \(Phase 2\)", regex=True)
        ]
        if final_rows.empty or initial_rows.empty:
            continue

        initial_counts = _counts(initial_rows.iloc[0]["Upper_Bound_Counts"])
        initial_area = float(initial_rows.iloc[0]["Upper_Bound_Area"])
        final_counts = _counts(final_rows.iloc[0]["Upper_Bound_Counts"])
        final_area = float(final_rows.iloc[0]["Upper_Bound_Area"])
        nodes_explored, total_budgets = _phase3_search_stats(summary_path)

        final_solutions = {}
        segment_timing = timings.get(segment, {})
        for _, saved in final_rows.iterrows():
            activation = str(saved["Function"])
            degrees = _degrees(saved["Degrees"])
            degree_counts = _counts(saved["Degree_Counts"])
            solution = {
                "degrees": degrees,
                "counts": degree_counts,
                "area": float(saved["Individual_Area"]),
                "mse": float(saved["MSE"]),
            }
            final_solutions[activation] = solution

            row = {
                "Activation": activation,
                "SegmentsRequested": segment,
                "SegmentsFinal": segment,
                "DegreeMax": degree_max,
                "TotalParameters": int(sum(degree + 1 for degree in degrees)),
                "TotalArea": solution["area"],
                "TrainMSE": solution["mse"],
                "EvalMSE": _eval_mse(
                    root, segment, activation, degree_max, degree_counts
                ),
                "MixedEnabled": True,
                "RefinePlausible": bool(refine_plausible),
                "Phase3Algo": str(phase3_algo),
                "Counts_c0c1c2c3": str(degree_counts),
            }
            row.update(segment_timing.get("activations", {}).get(activation, {}))
            for key in (
                "Stage1RuntimeSec",
                "Stage2RuntimeSec",
                "Stage2Phase3OptRuntimeSec",
                "Stage2PostRuntimeSec",
                "SegmentRuntimeSec",
            ):
                row[key] = segment_timing.get(key, np.nan)
            rows.append(row)

        phase3_outputs[segment] = {
            "history": [
                {
                    "upper_bound": initial_counts,
                    "upper_bound_area": initial_area,
                }
            ],
            "final_upper_bound": final_counts,
            "final_upper_bound_area": final_area,
            "final_solutions": final_solutions,
            "nodes_explored": nodes_explored,
            "total_budgets": total_budgets,
        }

    return pd.DataFrame(rows), phase3_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-min", type=int, default=15)
    parser.add_argument("--seg-max", type=int, default=16)
    parser.add_argument("--max-error", type=float, required=True)
    parser.add_argument("--degree-max", type=int, default=3)
    parser.add_argument("--refine-plausible", action="store_true")
    parser.add_argument(
        "--phase3-algo",
        choices=["iterative", "tree", "bruteforce", "bestfirst"],
        default="bestfirst",
    )
    args = parser.parse_args()
    if args.seg_min > args.seg_max:
        parser.error("--seg-min cannot exceed --seg-max")

    output_name = pipeline.helper_format_output_root_name(
        degree_max=args.degree_max,
        max_error_budget=args.max_error,
        num_functions=len(pipeline.ACTIVATION_FUNCTIONS),
    )
    out_dirs = pipeline.io_prepare_output_dirs(out_root=output_name, fresh=False)
    root = Path(out_dirs["root"])
    log_dir = root / "logs"
    log_path = pipeline.setup_logging(
        log_dir, f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    existing_df, existing_phase3 = reconstruct_completed_results(
        root=root,
        degree_max=args.degree_max,
        refine_plausible=args.refine_plausible,
        phase3_algo=args.phase3_algo,
    )
    if existing_df.empty:
        raise RuntimeError(f"No completed segment artifacts found under {root / 'runs'}")

    existing_df.to_csv(root / "pareto_sweep_summary.csv", index=False)
    completed = sorted(existing_phase3)
    pipeline.log(f"Outputs: {root}")
    pipeline.log(f"Log file: {log_path}")
    pipeline.log(f"Reconstructed completed segments: {completed}")
    pipeline.log(f"Resuming segment range: {args.seg_min}..{args.seg_max}")

    start = time.perf_counter()
    pipeline.run_segment_sweep(
        out_dirs=out_dirs,
        degree_max=args.degree_max,
        n_train_samples=1024,
        n_eval_samples=1024,
        num_outer_iters=None,
        min_seg_points=10,
        lam=0.0,
        seg_min=args.seg_min,
        seg_max=args.seg_max,
        use_mixed_degrees=True,
        max_error_budget=args.max_error,
        save_per_budget_overfit_plots=True,
        refine_plausible_configs=args.refine_plausible,
        phase3_algo=args.phase3_algo,
        clear_existing_runs=False,
        existing_phase3_outputs_by_segments=existing_phase3,
    )
    pipeline.log(f"Resume complete in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
