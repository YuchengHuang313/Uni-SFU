#!/usr/bin/env python3
"""
Build a paper-friendly comparison CSV directly from exported Pareto point CSVs.

This reads the plot-export CSVs produced by optimize_piecewise_activations.py:
  - pareto_area_points_maxX_segY-Z.csv
  - pareto_params_points_maxX_segY-Z.csv (optional, for parameter columns)

It matches:
  - Flex-SFU 8-BP references   <-> our 7-segment budget
  - Flex-SFU 16-BP references  <-> our 15-segment budget

and writes a flat compare.csv with enough columns to fill both the MSE and area
comparison tables.

Usage:
    python3 compare_flex_sfu_baselines.py \
        Uni-SFU/max3_results_4/pareto/pareto_area_points_max3_seg1-16.csv

Optional:
    python3 compare_flex_sfu_baselines.py <area_csv> --params-csv <params_csv> --out-csv <compare_csv>
"""

import argparse
import csv
from pathlib import Path


MATCHED_CASES = [
    {
        "case_label": "8-BP",
        "ref_bp": 8,
        "matched_segment_budget": 7,
    },
    {
        "case_label": "16-BP",
        "ref_bp": 16,
        "matched_segment_budget": 15,
    },
]


def clean_row(row):
    """Strip padded CSV headers/values from exported Pareto-point files."""
    return {
        (key or "").strip(): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
    }


def parse_float(value):
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def parse_int(value):
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def factor_vs_reference(our_value, ref_value):
    """Return multiplicative difference and direction versus reference."""
    if our_value is None or ref_value is None:
        return None, ""
    if our_value <= 0.0 or ref_value <= 0.0:
        return None, ""
    if abs(our_value - ref_value) <= 1e-18:
        return 1.0, "equal"
    if our_value < ref_value:
        return ref_value / our_value, "down"
    return our_value / ref_value, "up"


def percent_vs_reference(our_value, ref_value):
    """Return percentage difference and direction versus reference."""
    if our_value is None or ref_value is None:
        return None, ""
    if ref_value == 0.0:
        return None, ""
    if abs(our_value - ref_value) <= 1e-18:
        return 0.0, "equal"
    if our_value < ref_value:
        return (ref_value - our_value) / ref_value * 100.0, "down"
    return (our_value - ref_value) / ref_value * 100.0, "up"


def format_factor_text(value, direction):
    if value is None or not direction:
        return ""
    if direction == "equal":
        return "1.0x"
    return f"{value:.1f}x{direction}"


def format_percent_text(value, direction):
    if value is None or not direction:
        return ""
    if direction == "equal":
        return "0.0%"
    return f"{value:.1f}%{direction}"


def load_rows(csv_path):
    csv_path = Path(csv_path)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [clean_row(row) for row in reader]


def derive_activation_order(*rows_groups):
    """
    Derive activation order from the source CSVs.

    We keep first-seen order from our exported train curves so the compare.csv
    naturally follows the input file. If those are absent, fall back to any
    activation-bearing rows.
    """
    ordered = []
    seen = set()

    def add_activation(name):
        name = (name or "").strip()
        if not name or name in seen:
            return
        seen.add(name)
        ordered.append(name)

    for rows in rows_groups:
        for row in rows:
            if row.get("series_type", "") == "ours" and row.get("curve", "") == "train":
                add_activation(row.get("activation", ""))

    if ordered:
        return ordered

    for rows in rows_groups:
        for row in rows:
            add_activation(row.get("activation", ""))

    return ordered


def infer_counterpart_csv(csv_path, want):
    csv_path = Path(csv_path)
    if want == "area":
        replacements = [
            ("pareto_params_points_", "pareto_area_points_"),
            ("params_points", "area_points"),
            ("params", "area"),
        ]
    elif want == "params":
        replacements = [
            ("pareto_area_points_", "pareto_params_points_"),
            ("area_points", "params_points"),
            ("area", "params"),
        ]
    else:
        raise ValueError(f"Unknown counterpart type: {want}")

    candidates = [
        csv_path.with_name(csv_path.name.replace(old, new))
        for old, new in replacements
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer matching {want} CSV from {csv_path}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def infer_phase3_summary_csv(area_csv):
    """Infer the aggregate Phase 3 summary CSV from a pareto area-points CSV path."""
    area_csv = Path(area_csv)
    candidates = [
        area_csv.parent.parent / "area_err_optimization" / "aggregate_phase3_summary.csv",
        area_csv.parent / "aggregate_phase3_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_ours_lookup(rows):
    """Map (activation, segment_budget) -> train row and segment_budget -> common-bound row."""
    ours = {}
    common_bounds = {}

    for row in rows:
        series_type = row.get("series_type", "")
        curve = row.get("curve", "")
        activation = row.get("activation", "")
        seg_budget = parse_int(row.get("segment_budget", ""))

        if series_type == "ours" and curve == "train" and activation and seg_budget is not None:
            ours[(activation, seg_budget)] = row
        elif series_type == "common_bound" and seg_budget is not None:
            common_bounds[seg_budget] = row

    return ours, common_bounds


def build_reference_lookup(rows):
    """Map (activation, ref_bp, ref_order) -> reference row."""
    refs = {}
    for row in rows:
        if row.get("series_type", "") != "ref":
            continue
        activation = row.get("activation", "")
        ref_bp = parse_int(row.get("ref_bp", ""))
        ref_order = parse_int(row.get("ref_order", ""))
        if activation and ref_bp is not None and ref_order is not None:
            refs[(activation, ref_bp, ref_order)] = row
    return refs


def load_shared_bounds(summary_csv):
    """Map segment budget -> [deg0, deg1, deg2, deg3] from aggregate_phase3_summary.csv."""
    if summary_csv is None:
        return {}

    rows = load_rows(summary_csv)
    bounds = {}
    for row in rows:
        seg = parse_int(row.get("segments", ""))
        if seg is None:
            continue
        bound = [
            parse_int(row.get("final_ub_deg0", "")),
            parse_int(row.get("final_ub_deg1", "")),
            parse_int(row.get("final_ub_deg2", "")),
            parse_int(row.get("final_ub_deg3", "")),
        ]
        if all(v is not None for v in bound):
            bounds[seg] = bound
    return bounds


def build_compare_rows(area_csv, params_csv=None, summary_csv=None):
    area_rows = load_rows(area_csv)
    params_rows = load_rows(params_csv) if params_csv is not None and Path(params_csv).exists() else []
    shared_bounds = load_shared_bounds(summary_csv)
    activation_order = derive_activation_order(area_rows, params_rows)

    ours_area, common_area = build_ours_lookup(area_rows)
    refs_area = build_reference_lookup(area_rows)
    ours_params, common_params = build_ours_lookup(params_rows) if params_rows else ({}, {})
    refs_params = build_reference_lookup(params_rows) if params_rows else {}

    compare_rows = []
    for case_idx, case in enumerate(MATCHED_CASES):
        seg_budget = int(case["matched_segment_budget"])
        ref_bp = int(case["ref_bp"])
        shared_bound = shared_bounds.get(seg_budget)
        shared_bound_str = str(shared_bound) if shared_bound is not None else ""

        common_params_row = common_params.get(seg_budget)
        common_area_row = common_area.get(seg_budget)
        our_common_params = parse_float(common_params_row.get("x", "")) if common_params_row else None
        our_common_area = parse_float(common_area_row.get("x", "")) if common_area_row else None

        for act_idx, activation in enumerate(activation_order):
            our_param_row = ours_params.get((activation, seg_budget))
            our_area_row = ours_area.get((activation, seg_budget))

            linear_param_ref = refs_params.get((activation, ref_bp, 1))
            quad_param_ref = refs_params.get((activation, ref_bp, 2))
            linear_area_ref = refs_area.get((activation, ref_bp, 1))
            quad_area_ref = refs_area.get((activation, ref_bp, 2))

            our_mse = parse_float(our_area_row.get("y", "")) if our_area_row else None
            our_params = parse_float(our_param_row.get("x", "")) if our_param_row else None
            our_area = parse_float(our_area_row.get("x", "")) if our_area_row else None

            linear_ref_mse = parse_float(linear_area_ref.get("y", "")) if linear_area_ref else None
            quadratic_ref_mse = parse_float(quad_area_ref.get("y", "")) if quad_area_ref else None
            linear_ref_params = parse_float(linear_param_ref.get("x", "")) if linear_param_ref else None
            quadratic_ref_params = parse_float(quad_param_ref.get("x", "")) if quad_param_ref else None
            linear_ref_area = parse_float(linear_area_ref.get("x", "")) if linear_area_ref else None
            quadratic_ref_area = parse_float(quad_area_ref.get("x", "")) if quad_area_ref else None

            mse_vs_linear_factor, mse_vs_linear_direction = factor_vs_reference(our_mse, linear_ref_mse)
            mse_vs_quad_factor, mse_vs_quad_direction = factor_vs_reference(our_mse, quadratic_ref_mse)
            area_vs_linear_pct, area_vs_linear_direction = percent_vs_reference(our_common_area, linear_ref_area)
            area_vs_quad_pct, area_vs_quad_direction = percent_vs_reference(our_common_area, quadratic_ref_area)

            compare_rows.append({
                "case_label": case["case_label"],
                "ref_breakpoints": ref_bp,
                "matched_segment_budget": seg_budget,
                "shared_bound": shared_bound_str,
                "activation": activation,
                "our_mse": our_mse if our_mse is not None else "",
                "our_params": our_params if our_params is not None else "",
                "our_area": our_area if our_area is not None else "",
                "our_common_params": our_common_params if our_common_params is not None else "",
                "our_common_area": our_common_area if our_common_area is not None else "",
                "linear_ref_mse": linear_ref_mse if linear_ref_mse is not None else "",
                "linear_ref_params": linear_ref_params if linear_ref_params is not None else "",
                "linear_ref_area": linear_ref_area if linear_ref_area is not None else "",
                "quadratic_ref_mse": quadratic_ref_mse if quadratic_ref_mse is not None else "",
                "quadratic_ref_params": quadratic_ref_params if quadratic_ref_params is not None else "",
                "quadratic_ref_area": quadratic_ref_area if quadratic_ref_area is not None else "",
                "mse_vs_linear_factor": mse_vs_linear_factor if mse_vs_linear_factor is not None else "",
                "mse_vs_linear_direction": mse_vs_linear_direction,
                "mse_vs_linear_text": format_factor_text(mse_vs_linear_factor, mse_vs_linear_direction),
                "mse_vs_quadratic_factor": mse_vs_quad_factor if mse_vs_quad_factor is not None else "",
                "mse_vs_quadratic_direction": mse_vs_quad_direction,
                "mse_vs_quadratic_text": format_factor_text(mse_vs_quad_factor, mse_vs_quad_direction),
                "area_vs_linear_pct": area_vs_linear_pct if area_vs_linear_pct is not None else "",
                "area_vs_linear_direction": area_vs_linear_direction,
                "area_vs_linear_text": format_percent_text(area_vs_linear_pct, area_vs_linear_direction),
                "area_vs_quadratic_pct": area_vs_quad_pct if area_vs_quad_pct is not None else "",
                "area_vs_quadratic_direction": area_vs_quad_direction,
                "area_vs_quadratic_text": format_percent_text(area_vs_quad_pct, area_vs_quad_direction),
                "_case_order": case_idx,
                "_act_order": act_idx,
            })

    compare_rows.sort(key=lambda row: (row["_case_order"], row["_act_order"]))
    return compare_rows


def write_compare_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_label",
        "ref_breakpoints",
        "matched_segment_budget",
        "shared_bound",
        "activation",
        "our_mse",
        "our_params",
        "our_area",
        "our_common_params",
        "our_common_area",
        "linear_ref_mse",
        "linear_ref_params",
        "linear_ref_area",
        "quadratic_ref_mse",
        "quadratic_ref_params",
        "quadratic_ref_area",
        "mse_vs_linear_factor",
        "mse_vs_linear_direction",
        "mse_vs_linear_text",
        "mse_vs_quadratic_factor",
        "mse_vs_quadratic_direction",
        "mse_vs_quadratic_text",
        "area_vs_linear_pct",
        "area_vs_linear_direction",
        "area_vs_linear_text",
        "area_vs_quadratic_pct",
        "area_vs_quadratic_direction",
        "area_vs_quadratic_text",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if not k.startswith("_")})


def main():
    parser = argparse.ArgumentParser(
        description="Compare our Pareto-point outputs against Flex-SFU reference markers.",
    )
    parser.add_argument(
        "points_csv",
        type=str,
        help="Path to a Pareto point CSV. Prefer pareto_area_points_maxX_segY-Z.csv.",
    )
    parser.add_argument(
        "--area-csv",
        type=str,
        default=None,
        help="Optional path to matching pareto_area_points CSV. If omitted, inferred when needed.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output compare.csv path. Defaults next to params_csv.",
    )
    parser.add_argument(
        "--params-csv",
        type=str,
        default=None,
        help="Optional path to matching pareto_params_points CSV for parameter columns.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional path to aggregate_phase3_summary.csv for the real shared bounds.",
    )
    args = parser.parse_args()

    input_csv = Path(args.points_csv)
    input_name = input_csv.name

    if "pareto_area_points_" in input_name or "_area_" in input_name:
        area_csv = Path(args.area_csv) if args.area_csv else input_csv
        params_csv = Path(args.params_csv) if args.params_csv else None
        if params_csv is None:
            try:
                params_csv = infer_counterpart_csv(area_csv, "params")
            except FileNotFoundError:
                params_csv = None
    elif "pareto_params_points_" in input_name or "_params_" in input_name:
        params_csv = Path(args.params_csv) if args.params_csv else input_csv
        area_csv = Path(args.area_csv) if args.area_csv else infer_counterpart_csv(params_csv, "area")
    else:
        area_csv = Path(args.area_csv) if args.area_csv else input_csv
        params_csv = Path(args.params_csv) if args.params_csv else None
        if not area_csv.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {area_csv}")

    summary_csv = Path(args.summary_csv) if args.summary_csv else infer_phase3_summary_csv(area_csv)
    out_csv = Path(args.out_csv) if args.out_csv else area_csv.with_name("compare.csv")

    rows = build_compare_rows(area_csv, params_csv=params_csv, summary_csv=summary_csv)
    write_compare_csv(rows, out_csv)

    print(f"Area CSV   : {area_csv}")
    print(f"Params CSV : {params_csv if params_csv is not None else '(not used)'}")
    print(f"Summary CSV: {summary_csv if summary_csv is not None else '(not found)'}")
    print(f"Output CSV : {out_csv}")
    print(f"Rows       : {len(rows)}")


if __name__ == "__main__":
    main()
