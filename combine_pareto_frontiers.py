#!/usr/bin/env python3
"""
Overlay AREA-vs-MSE sweep summaries from multiple optimize_piecewise_activations.py runs.

This script uses each run's `pareto_sweep_summary.csv` as the single source of
truth. From those summary rows it can:
  - plot all AREA-vs-MSE points on one figure
  - recompute and highlight Pareto frontiers per activation
  - export merged point CSVs
  - export frontier-only CSV and readable text summaries
  - derive per-run upper bounds by taking the elementwise max degree counts
    across activations for the same segment budget

Example:
    python3 combine_pareto_frontiers.py /path/to/Uni-SFU
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CSV_PATTERN = "pareto_sweep_summary.csv"
PARETO_AREA_PATTERN = "pareto_area_points_max*_seg*.csv"
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
REF_MARKER_BY_LABEL = {
    "8BP-1st": "P",
    "8BP-2nd": "X",
    "16BP-1st": "^",
    "16BP-2nd": "D",
}
DEFAULT_OUT_NAME = "combined_pareto_area_overlay.png"
DEFAULT_CSV_OUT_NAME = "combined_pareto_area_overlay_points.csv"
DEFAULT_FRONTIER_CONFIG_CSV_OUT_NAME = "combined_pareto_frontier_points.csv"
DEFAULT_TEXT_OUT_NAME = "combined_pareto_frontier_summary.txt"


class ValidationError(RuntimeError):
    """Raised when the inputs do not satisfy the plot requirements."""


@dataclass(frozen=True)
class SourceInfo:
    csv_path: Path
    result_dir: Path
    result_name: str
    max_error_token: str
    max_error_value: float | None
    max_error_label: str
    degree_max: int


@dataclass
class PlotPoint:
    activation: str
    area: float
    mse: float
    segment_budget: int
    source: SourceInfo
    summary_row: dict[str, str]
    degree_distribution_by_segment: str
    is_frontier: bool = False


@dataclass
class FrontierDisplayPoint:
    activation: str
    area: float
    mse: float
    segment_budget: int
    is_frontier: bool
    duplicate_count: int
    source_names: list[str]
    label_text: str


@dataclass
class LabelPoint:
    activation: str
    area: float
    mse: float
    is_frontier: bool
    label_text: str


@dataclass(frozen=True)
class RefPoint:
    activation: str
    area: float
    mse: float
    ref_label: str
    ref_bp: int | None
    ref_order: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple pareto_sweep_summary.csv files onto one AREA-vs-MSE plot "
            "and highlight recomputed Pareto frontiers."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help=(
            "Result folders, pareto folders, CSV files, or parent directories to scan. "
            "Defaults to the current directory."
        ),
    )
    parser.add_argument(
        "--curve",
        choices=["train", "eval"],
        default="train",
        help="Which 'ours' curve from the CSV to plot. Default: train.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: <common_parent>/combined_pareto_area_overlay.png"
        ),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path for the merged points plus frontier flags. "
            "Default: <common_parent>/combined_pareto_area_overlay_points.csv"
        ),
    )
    parser.add_argument(
        "--frontier-csv-out",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for frontier-only point export. "
            "Default: <common_parent>/combined_pareto_frontier_points.csv"
        ),
    )
    parser.add_argument(
        "--text-out",
        type=Path,
        default=None,
        help=(
            "Optional readable text summary path. "
            "Default: <common_parent>/combined_pareto_frontier_summary.txt"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI. Default: 220.",
    )
    parser.add_argument(
        "--label-frontier-only",
        action="store_true",
        help="Backward-compatible alias; frontier-only labels are now the default.",
    )
    parser.add_argument(
        "--label-all-points",
        action="store_true",
        help="Annotate all plotted points with SXX instead of only frontier points.",
    )
    return parser.parse_args()


def discover_csvs(paths: Iterable[str]) -> list[Path]:
    discovered: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.name == CSV_PATTERN:
                discovered.add(path.resolve())
            continue

        if not path.exists():
            continue

        for csv_path in path.rglob(CSV_PATTERN):
            if csv_path.is_file():
                discovered.add(csv_path.resolve())

    return sorted(discovered)


def parse_max_error_token(result_name: str) -> tuple[str, float | None, str]:
    match = re.search(r"maxerr([^_]+)", result_name)
    if not match:
        return "", None, result_name

    token = match.group(1)
    try:
        value = float(token.replace("p", ".", 1))
    except ValueError:
        return token, None, f"maxerr={token}"

    return token, value, f"maxerr={value:.2e}"


def clean_csv_row(raw_row: dict[object, object]) -> dict[str, str]:
    return {
        str(key or "").strip(): str(value or "").strip()
        for key, value in raw_row.items()
    }


def build_source_info(csv_path: Path, degree_max: int) -> SourceInfo:
    if csv_path.name != CSV_PATTERN:
        raise ValidationError(
            f"Expected summary CSV named {CSV_PATTERN}, got: {csv_path}"
        )

    result_dir = csv_path.parent
    result_name = result_dir.name
    max_error_token, max_error_value, max_error_label = parse_max_error_token(result_name)

    return SourceInfo(
        csv_path=csv_path,
        result_dir=result_dir,
        result_name=result_name,
        max_error_token=max_error_token,
        max_error_value=max_error_value,
        max_error_label=max_error_label,
        degree_max=degree_max,
    )


def parse_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_segment_budget(value: str) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def parse_degree_distribution_from_config(config_csv: Path) -> list[int]:
    degrees_by_segment: dict[int, int] = {}

    with config_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("type") or "").strip() != "degree":
                continue
            seg_idx = parse_segment_budget(row.get("segment_idx", ""))
            degree = parse_segment_budget(row.get("value", ""))
            if seg_idx is None or degree is None:
                continue
            degrees_by_segment[seg_idx] = degree

    if not degrees_by_segment:
        raise ValidationError(f"No degree rows found in config CSV: {config_csv}")

    expected_indices = list(range(max(degrees_by_segment) + 1))
    actual_indices = sorted(degrees_by_segment)
    if actual_indices != expected_indices:
        raise ValidationError(
            f"Degree rows in {config_csv} are not contiguous: found {actual_indices}"
        )

    return [degrees_by_segment[idx] for idx in expected_indices]


def find_segment_run_dir(result_dir: Path, segment_budget: int) -> Path | None:
    candidates = [
        result_dir / "runs" / f"seg_{segment_budget}",
        result_dir / "runs" / f"seg_{segment_budget:02d}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    runs_dir = result_dir / "runs"
    if not runs_dir.is_dir():
        return None

    for candidate in sorted(runs_dir.glob("seg_*")):
        suffix = candidate.name.split("seg_", 1)[-1]
        if parse_segment_budget(suffix) == segment_budget and candidate.is_dir():
            return candidate

    return None


def resolve_degree_distribution(
    source: SourceInfo,
    activation: str,
    segment_budget: int,
    count_signature: str,
    cache: dict[tuple[str, int, str, str], str],
) -> str:
    cache_key = (source.result_name, segment_budget, activation, count_signature)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    segment_dir = find_segment_run_dir(source.result_dir, segment_budget)
    if segment_dir is None:
        raise ValidationError(
            f"Missing run directory for {activation} S{segment_budget:02d} in {source.result_dir / 'runs'}"
        )

    config_dirs = [
        segment_dir / "common_area_config",
        segment_dir / "min_area_config",
    ]
    config_dir = next((path for path in config_dirs if path.is_dir()), None)
    if config_dir is None:
        raise ValidationError(
            f"Missing common_area_config/min_area_config for summary row: {segment_dir}"
        )

    pattern = f"{activation}_{segment_budget}seg_d*.csv"
    candidates = sorted(config_dir.glob(pattern))
    if not candidates:
        raise ValidationError(
            f"No common_area_config CSV found for {activation} S{segment_budget:02d} in {config_dir}"
        )

    desired_counts = parse_degree_count_list(count_signature)
    if desired_counts:
        matching_candidates = []
        for candidate in candidates:
            degrees = parse_degree_distribution_from_config(candidate)
            candidate_counts = [
                degrees.count(0),
                degrees.count(1),
                degrees.count(2),
                degrees.count(3),
            ]
            if candidate_counts == desired_counts:
                matching_candidates.append(candidate)
        if matching_candidates:
            candidates = matching_candidates

    if len(candidates) != 1:
        raise ValidationError(
            f"Expected exactly one config CSV for {activation} S{segment_budget:02d} "
            f"in {config_dir}, found {[str(path.name) for path in candidates]}"
        )

    degree_distribution = str(parse_degree_distribution_from_config(candidates[0]))
    cache[cache_key] = degree_distribution
    return degree_distribution


def load_points(csv_path: Path, curve: str) -> tuple[SourceInfo, list[str], list[PlotPoint]]:
    activation_order: list[str] = []
    points: list[PlotPoint] = []
    rows: list[dict[str, str]] = []
    degree_values: set[int] = set()
    degree_distribution_cache: dict[tuple[str, int, str, str], str] = {}
    mse_column = "TrainMSE" if curve == "train" else "EvalMSE"

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {str(k): str(v) for k, v in raw_row.items()}
            rows.append(row)
            degree_val = parse_segment_budget(row.get("DegreeMax", ""))
            if degree_val is not None:
                degree_values.add(degree_val)

    if not rows:
        raise ValidationError(f"No rows found in {csv_path}")
    if len(degree_values) != 1:
        raise ValidationError(
            f"Expected exactly one DegreeMax value in {csv_path}, found {sorted(degree_values)}"
        )

    source = build_source_info(csv_path, degree_max=next(iter(degree_values)))

    for row in rows:
        activation = (row.get("Activation") or "").strip()
        area = parse_float(row.get("TotalArea", ""))
        mse = parse_float(row.get(mse_column, ""))
        segment_budget = parse_segment_budget(row.get("SegmentsRequested", ""))

        if not activation or area is None or mse is None or mse <= 0.0 or segment_budget is None:
            continue

        if activation not in activation_order:
            activation_order.append(activation)

        degree_distribution = resolve_degree_distribution(
            source=source,
            activation=activation,
            segment_budget=segment_budget,
            count_signature=row.get("Counts_c0c1c2c3", ""),
            cache=degree_distribution_cache,
        )

        points.append(
            PlotPoint(
                activation=activation,
                area=area,
                mse=mse,
                segment_budget=segment_budget,
                source=source,
                summary_row=row,
                degree_distribution_by_segment=degree_distribution,
            )
        )

    if not points:
        raise ValidationError(
            f"No usable {curve} points found in {csv_path}"
        )

    return source, activation_order, points


def load_reference_points(
    sources: list[SourceInfo],
    valid_activations: set[str],
) -> list[RefPoint]:
    dedup: OrderedDict[tuple[str, str, str, str, int | None, int | None], RefPoint] = OrderedDict()

    for source in sources:
        pareto_dir = source.result_dir / "pareto"
        if not pareto_dir.is_dir():
            continue

        csv_candidates = sorted(pareto_dir.glob(f"pareto_area_points_max{source.degree_max}_seg*.csv"))
        if not csv_candidates:
            csv_candidates = sorted(pareto_dir.glob(PARETO_AREA_PATTERN))

        for csv_path in csv_candidates:
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for raw_row in reader:
                    row = clean_csv_row(raw_row)
                    if row.get("metric") != "AREA":
                        continue
                    if row.get("series_type") != "ref":
                        continue

                    activation = row.get("activation", "")
                    if not activation or activation not in valid_activations:
                        continue

                    area = parse_float(row.get("x", ""))
                    mse = parse_float(row.get("y", ""))
                    if area is None or mse is None or mse <= 0.0:
                        continue

                    ref_label = row.get("ref_label", "")
                    ref_bp = parse_segment_budget(row.get("ref_bp", ""))
                    ref_order = parse_segment_budget(row.get("ref_order", ""))
                    if not ref_label:
                        bp_text = "" if ref_bp is None else f"{ref_bp}BP"
                        order_text = "" if ref_order is None else f"-{ref_order}"
                        ref_label = f"{bp_text}{order_text}".strip("-")

                    ref_point = RefPoint(
                        activation=activation,
                        area=area,
                        mse=mse,
                        ref_label=ref_label,
                        ref_bp=ref_bp,
                        ref_order=ref_order,
                    )
                    key = (
                        ref_point.activation,
                        f"{ref_point.area:.12g}",
                        f"{ref_point.mse:.12g}",
                        ref_point.ref_label,
                        ref_point.ref_bp,
                        ref_point.ref_order,
                    )
                    dedup.setdefault(key, ref_point)

    activation_rank = {name: idx for idx, name in enumerate(sorted(valid_activations))}
    return sorted(
        dedup.values(),
        key=lambda point: (
            activation_rank.get(point.activation, len(activation_rank)),
            float("inf") if point.ref_bp is None else point.ref_bp,
            float("inf") if point.ref_order is None else point.ref_order,
            point.ref_label,
            point.area,
            point.mse,
        ),
    )


def validate_sources(
    loaded: list[tuple[SourceInfo, list[str], list[PlotPoint]]]
) -> tuple[list[str], list[PlotPoint], list[SourceInfo], int]:
    if not loaded:
        raise ValidationError(f"No {CSV_PATTERN} files were found.")

    reference_order = loaded[0][1]
    reference_set = set(reference_order)
    degree_max = loaded[0][0].degree_max
    reference_segments = {p.segment_budget for p in loaded[0][2]}

    all_points: list[PlotPoint] = []
    sources: list[SourceInfo] = []

    for source, activation_order, points in loaded:
        activation_set = set(activation_order)
        if activation_set != reference_set:
            missing = sorted(reference_set - activation_set)
            extra = sorted(activation_set - reference_set)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            detail_text = ", ".join(details) if details else "activation mismatch"
            raise ValidationError(
                f"{source.csv_path} does not match the activation set from "
                f"{loaded[0][0].csv_path} ({detail_text})."
            )

        if source.degree_max != degree_max:
            raise ValidationError(
                f"degree_max mismatch: {source.csv_path} has max{source.degree_max}, "
                f"expected max{degree_max}."
            )

        segment_set = {p.segment_budget for p in points}
        if segment_set != reference_segments:
            raise ValidationError(
                f"segment-budget mismatch: {source.csv_path} has {sorted(segment_set)}, "
                f"expected {sorted(reference_segments)}."
            )

        all_points.extend(points)
        sources.append(source)

    return reference_order, all_points, sources, degree_max


def dominates(a: PlotPoint, b: PlotPoint, rel_tol: float = 1e-12) -> bool:
    area_le = a.area <= b.area + max(abs(b.area), 1.0) * rel_tol
    mse_le = a.mse <= b.mse + max(abs(b.mse), 1.0) * rel_tol
    area_lt = a.area < b.area - max(abs(b.area), 1.0) * rel_tol
    mse_lt = a.mse < b.mse - max(abs(b.mse), 1.0) * rel_tol
    return area_le and mse_le and (area_lt or mse_lt)


def mark_frontiers(points: list[PlotPoint]) -> OrderedDict[str, list[PlotPoint]]:
    frontier_by_activation: OrderedDict[str, list[PlotPoint]] = OrderedDict()
    grouped: defaultdict[str, list[PlotPoint]] = defaultdict(list)

    for point in points:
        grouped[point.activation].append(point)

    for activation in sorted(grouped):
        frontier: list[PlotPoint] = []
        candidates = grouped[activation]
        for idx, point in enumerate(candidates):
            dominated = False
            for jdx, other in enumerate(candidates):
                if idx == jdx:
                    continue
                if dominates(other, point):
                    dominated = True
                    break
            if not dominated:
                point.is_frontier = True
                frontier.append(point)

        frontier.sort(key=lambda p: (p.area, p.mse, p.source.max_error_label, p.segment_budget))
        frontier_by_activation[activation] = frontier

    return frontier_by_activation


def build_frontier_display(
    frontier_by_activation: OrderedDict[str, list[PlotPoint]]
) -> OrderedDict[str, list[FrontierDisplayPoint]]:
    display_by_activation: OrderedDict[str, list[FrontierDisplayPoint]] = OrderedDict()

    for activation, frontier_points in frontier_by_activation.items():
        grouped: OrderedDict[tuple[str, str, str], list[PlotPoint]] = OrderedDict()
        for point in frontier_points:
            key = (
                f"{point.area:.12g}",
                f"{point.mse:.12g}",
                str(point.segment_budget),
            )
            grouped.setdefault(key, []).append(point)

        display_points: list[FrontierDisplayPoint] = []
        for group_points in grouped.values():
            ref = group_points[0]
            duplicate_count = len(group_points)
            label_text = f"S{ref.segment_budget:02d}"
            if duplicate_count > 1:
                label_text += f"x{duplicate_count}"

            display_points.append(
                FrontierDisplayPoint(
                    activation=activation,
                    area=ref.area,
                    mse=ref.mse,
                    segment_budget=ref.segment_budget,
                    is_frontier=True,
                    duplicate_count=duplicate_count,
                    source_names=sorted({p.source.result_name for p in group_points}),
                    label_text=label_text,
                )
            )

        display_points.sort(key=lambda p: (p.area, p.mse, p.segment_budget, p.label_text))
        display_by_activation[activation] = display_points

    return display_by_activation


def parse_literal_list(text: str) -> list[object]:
    raw = (text or "").strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return []


def parse_degree_count_list(text: str) -> list[int]:
    values = parse_literal_list(text)
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            return []
    return out


def degree_distribution_to_n0123(degree_distribution_text: str) -> str:
    degrees = parse_degree_count_list(degree_distribution_text)
    if not degrees:
        return ""

    counts = [0, 0, 0, 0]
    for degree in degrees:
        if 0 <= degree <= 3:
            counts[degree] += 1
    return str(counts)


def degree_counts_to_c_counts(degree_counts: list[int]) -> tuple[int, int, int, int]:
    total = sum(int(v) for v in degree_counts)
    c_counts: list[int] = []
    running_sum = total
    for n in degree_counts:
        c_counts.append(running_sum)
        running_sum -= int(n)
    while len(c_counts) < 4:
        c_counts.append(0)
    return tuple(c_counts[:4])


def hardware_area_from_c_counts(c_counts: tuple[int, int, int, int]) -> float:
    c0, c1, c2, c3 = [float(v) for v in c_counts]
    if c3 > 0:
        max_degree = 3.0
    elif c2 > 0:
        max_degree = 2.0
    elif c1 > 0:
        max_degree = 1.0
    else:
        max_degree = 0.0

    return (
        60.9 * c0
        + 57.06 * c1
        + 56.37 * c2
        + 57.07 * c3
        + 162.49
        + 960.0 * max_degree
    )


def build_frontier_duplicate_lookup(
    frontier_display_by_activation: OrderedDict[str, list[FrontierDisplayPoint]]
) -> dict[tuple[str, str, str, int], FrontierDisplayPoint]:
    lookup: dict[tuple[str, str, str, int], FrontierDisplayPoint] = {}
    for activation, display_points in frontier_display_by_activation.items():
        for display_point in display_points:
            key = (
                activation,
                f"{display_point.area:.12g}",
                f"{display_point.mse:.12g}",
                display_point.segment_budget,
            )
            lookup[key] = display_point
    return lookup


def write_points_csv(
    csv_out: Path,
    points: list[PlotPoint],
    upper_degree_n0123_lookup: dict[tuple[str, int], str],
):
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "activation",
        "approximation_mse",
        "area",
        "max_error_label",
        "segment_budget",
        "upper_degree_count_n0123",
        "config_degree_count_n0123",
        "degree_distribution_by_segment",
    ]

    sorted_points = sorted(
        points,
        key=lambda p: (
            p.activation,
            p.area,
            p.mse,
            p.segment_budget,
            p.source.max_error_value if p.source.max_error_value is not None else float("inf"),
            p.source.result_name,
        )
    )

    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for point in sorted_points:
            writer.writerow(
                {
                    "activation": point.activation,
                    "approximation_mse": f"{point.mse:.12g}",
                    "area": f"{point.area:.12g}",
                    "max_error_label": point.source.max_error_label,
                    "segment_budget": point.segment_budget,
                    "upper_degree_count_n0123": upper_degree_n0123_lookup.get(
                        (point.source.result_name, point.segment_budget),
                        "",
                    ),
                    "config_degree_count_n0123": degree_distribution_to_n0123(
                        point.degree_distribution_by_segment
                    ),
                    "degree_distribution_by_segment": point.degree_distribution_by_segment,
                }
            )


def build_upper_bound_rows(points: list[PlotPoint]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[PlotPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.source.result_name, point.segment_budget)].append(point)

    rows: list[dict[str, object]] = []
    for (result_name, segment_budget), group_points in sorted(grouped.items()):
        first = group_points[0]
        degree_count_lists = [
            parse_degree_count_list(point.summary_row.get("Counts_c0c1c2c3", ""))
            for point in group_points
        ]
        degree_count_lists = [vals for vals in degree_count_lists if vals]
        max_len = max((len(vals) for vals in degree_count_lists), default=0)
        padded_lists = [
            vals + [0] * (max_len - len(vals))
            for vals in degree_count_lists
        ]
        degree_upper = [
            max(vals[idx] for vals in padded_lists)
            for idx in range(max_len)
        ] if padded_lists else []
        c_upper = degree_counts_to_c_counts(degree_upper) if degree_upper else (0, 0, 0, 0)
        upper_area = hardware_area_from_c_counts(c_upper) if degree_upper else None
        upper_params = sum((idx + 1) * count for idx, count in enumerate(degree_upper))

        rows.append(
            {
                "result_name": result_name,
                "result_dir": str(first.source.result_dir),
                "max_error_label": first.source.max_error_label,
                "max_error_value": (
                    "" if first.source.max_error_value is None
                    else f"{first.source.max_error_value:.12g}"
                ),
                "segment_budget": segment_budget,
                "activation_count": len(group_points),
                "upper_degree_counts_n0_n1_n2_n3": str(degree_upper),
                "upper_c_counts_c0_c1_c2_c3": str(list(c_upper)),
                "upper_total_parameters": upper_params,
                "upper_area": "" if upper_area is None else f"{upper_area:.12g}",
            }
        )

    return rows


def build_upper_degree_n0123_lookup(points: list[PlotPoint]) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for row in build_upper_bound_rows(points):
        degree_counts = parse_degree_count_list(str(row["upper_degree_counts_n0_n1_n2_n3"]))
        while len(degree_counts) < 4:
            degree_counts.append(0)
        n0123 = [degree_counts[0], degree_counts[1], degree_counts[2], degree_counts[3]]
        lookup[(str(row["result_name"]), int(row["segment_budget"]))] = str(n0123)
    return lookup


def write_text_summary(
    text_out: Path,
    points: list[PlotPoint],
    sources: list[SourceInfo],
    frontier_by_activation: OrderedDict[str, list[PlotPoint]],
    curve: str,
):
    text_out.parent.mkdir(parents=True, exist_ok=True)
    frontier_display = build_frontier_display(frontier_by_activation)
    upper_bound_rows = build_upper_bound_rows(points)

    lines: list[str] = []
    lines.append("Combined Pareto Frontier Summary")
    lines.append("")
    lines.append(f"Curve: {curve}")
    lines.append(f"Runs: {len(sources)}")
    lines.append(f"Points: {len(points)}")
    lines.append(f"Raw frontier rows: {sum(len(v) for v in frontier_by_activation.values())}")
    lines.append(f"Displayed frontier points: {sum(len(v) for v in frontier_display.values())}")
    lines.append("")
    lines.append("Runs")
    for source in sorted(
        sources,
        key=lambda s: (
            float("inf") if s.max_error_value is None else s.max_error_value,
            s.result_name,
        ),
    ):
        lines.append(f"  {source.max_error_label}: {source.result_dir}")

    lines.append("")
    lines.append("Frontier Points By Activation")
    for activation, display_points in frontier_display.items():
        lines.append(f"  {activation}")
        grouped_raw: dict[tuple[str, str, int], list[PlotPoint]] = defaultdict(list)
        for raw_point in frontier_by_activation[activation]:
            key = (
                f"{raw_point.area:.12g}",
                f"{raw_point.mse:.12g}",
                raw_point.segment_budget,
            )
            grouped_raw[key].append(raw_point)

        for display_point in display_points:
            key = (
                f"{display_point.area:.12g}",
                f"{display_point.mse:.12g}",
                display_point.segment_budget,
            )
            raw_group = grouped_raw.get(key, [])
            summary_row = raw_group[0].summary_row if raw_group else {}
            run_labels = ", ".join(point.source.max_error_label for point in raw_group)
            lines.append(
                "    "
                f"{display_point.label_text}: area={display_point.area:.6f}, "
                f"mse={display_point.mse:.6e}, params={summary_row.get('TotalParameters', '')}, "
                f"counts={summary_row.get('Counts_c0c1c2c3', '')}, runs=[{run_labels}]"
            )

    lines.append("")
    lines.append("Upper Bounds From Sweep Summary")
    current_run = None
    for row in upper_bound_rows:
        if row["result_name"] != current_run:
            current_run = row["result_name"]
            lines.append(f"  {row['max_error_label']} ({row['result_dir']})")
        lines.append(
            "    "
            f"S{int(row['segment_budget']):02d}: degree_counts={row['upper_degree_counts_n0_n1_n2_n3']}, "
            f"c_counts={row['upper_c_counts_c0_c1_c2_c3']}, "
            f"params={row['upper_total_parameters']}, area={row['upper_area']}"
        )

    lines.append("")
    text_out.write_text("\n".join(lines) + "\n")


def format_source_label(source: SourceInfo, duplicates: set[str]) -> str:
    if source.max_error_label and source.max_error_label not in duplicates:
        return source.max_error_label
    if source.max_error_label:
        return f"{source.max_error_label} [{source.result_name}]"
    return source.result_name


def annotate_segment_labels(ax, points: list[LabelPoint], color_by_activation: dict[str, tuple]):
    from matplotlib import patheffects

    if not points:
        return

    fig = ax.figure
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72.0
    candidate_offsets_pt = [
        (4, 5), (4, -10), (-15, 5), (-15, -10),
        (9, 0), (-20, 0), (0, 11), (0, -13),
        (12, 8), (-24, 8), (12, -12), (-24, -12),
    ]
    placed_positions_px: list[tuple[float, float]] = []

    for point in sorted(points, key=lambda p: (p.area, p.mse, p.activation, p.label_text)):
        pt_x_px, pt_y_px = ax.transData.transform((point.area, point.mse))

        best = None
        for dx_pt, dy_pt in candidate_offsets_pt:
            tx_px = pt_x_px + dx_pt * px_per_pt
            ty_px = pt_y_px + dy_pt * px_per_pt
            collisions = 0
            min_sep = float("inf")
            for ex_px, ey_px in placed_positions_px:
                dx_abs = abs(tx_px - ex_px)
                dy_abs = abs(ty_px - ey_px)
                if dx_abs < 22 and dy_abs < 12:
                    collisions += 1
                min_sep = min(min_sep, dx_abs + dy_abs)
            score = (collisions, -min_sep)
            if best is None or score < best["score"]:
                best = {
                    "dx_pt": dx_pt,
                    "dy_pt": dy_pt,
                    "tx_px": tx_px,
                    "ty_px": ty_px,
                    "collisions": collisions,
                    "score": score,
                }
                if collisions == 0:
                    break

        if best is None:
            continue

        placed_positions_px.append((best["tx_px"], best["ty_px"]))
        color = color_by_activation[point.activation]
        arrowprops = None
        if best["collisions"] > 0:
            arrowprops = dict(
                arrowstyle="-",
                color=color,
                lw=0.35,
                alpha=0.45,
                shrinkA=0,
                shrinkB=2,
            )

        bbox_alpha = 0.48 if point.is_frontier else 0.28
        font_weight = "bold" if point.is_frontier else "normal"
        annotation = ax.annotate(
            point.label_text,
            xy=(point.area, point.mse),
            xytext=(best["dx_pt"], best["dy_pt"]),
            textcoords="offset points",
            fontsize=6.2 if point.is_frontier else 5.6,
            fontweight=font_weight,
            color=color,
            alpha=0.95 if point.is_frontier else 0.80,
            clip_on=True,
            bbox=dict(facecolor="white", alpha=bbox_alpha, edgecolor="none", pad=0.2),
            arrowprops=arrowprops,
        )
        annotation.set_path_effects(
            [patheffects.withStroke(linewidth=2.4 if point.is_frontier else 1.5, foreground="white")]
        )
def plot_points(
    points: list[PlotPoint],
    ref_points: list[RefPoint],
    activation_order: list[str],
    sources: list[SourceInfo],
    frontier_by_activation: OrderedDict[str, list[PlotPoint]],
    out_path: Path,
    curve: str,
    degree_max: int,
    dpi: int,
    title: str | None,
    label_all_points: bool,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import patheffects

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18.5, 11), dpi=dpi)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(10, len(activation_order)))
    color_by_activation = {
        activation: cmap(idx % cmap.N)
        for idx, activation in enumerate(activation_order)
    }

    max_error_labels = [source.max_error_label for source in sources if source.max_error_label]
    duplicate_max_error_labels = {
        label for label in max_error_labels if max_error_labels.count(label) > 1
    }
    ordered_source_keys = [
        source.result_name for source in sorted(
            sources,
            key=lambda s: (
                float("inf") if s.max_error_value is None else s.max_error_value,
                s.result_name,
            ),
        )
    ]
    marker_by_result = {
        result_name: MARKERS[idx % len(MARKERS)]
        for idx, result_name in enumerate(ordered_source_keys)
    }
    source_by_name = {source.result_name: source for source in sources}
    frontier_display_by_activation = build_frontier_display(frontier_by_activation)

    points_by_key: defaultdict[tuple[str, str], list[PlotPoint]] = defaultdict(list)
    for point in points:
        points_by_key[(point.activation, point.source.result_name)].append(point)

    for activation in activation_order:
        for result_name in ordered_source_keys:
            sub_points = points_by_key.get((activation, result_name), [])
            if not sub_points:
                continue
            background_points = [point for point in sub_points if not point.is_frontier]

            if background_points:
                bg_xs = [point.area for point in background_points]
                bg_ys = [point.mse for point in background_points]
                bg_color = color_by_activation[activation]
                bg_alpha = 0.10
                bg_size = 18
                bg_edge = "black"
                bg_linewidth = 0.15

                ax.scatter(
                    bg_xs,
                    bg_ys,
                    s=bg_size,
                    marker=marker_by_result[result_name],
                    color=bg_color,
                    alpha=bg_alpha,
                    edgecolors=bg_edge,
                    linewidths=bg_linewidth,
                    zorder=2,
                )

    for activation, display_points in frontier_display_by_activation.items():
        if not display_points:
            continue
        xs = [point.area for point in display_points]
        ys = [point.mse for point in display_points]
        ax.plot(
            xs,
            ys,
            color="white",
            linewidth=2.6,
            alpha=0.96,
            zorder=4,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        line, = ax.plot(
            xs,
            ys,
            color=color_by_activation[activation],
            linewidth=1.5,
            alpha=0.98,
            zorder=6,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        line.set_path_effects(
            [patheffects.withStroke(linewidth=2.3, foreground="white")]
        )

        for display_point in display_points:
            if display_point.duplicate_count == 1:
                marker = marker_by_result[display_point.source_names[0]]
            else:
                marker = "o"

            point_color = color_by_activation[activation]
            base_size = 40
            if display_point.duplicate_count > 1:
                base_size += 6
            halo_size = base_size + 28

            ax.scatter(
                [display_point.area],
                [display_point.mse],
                s=halo_size,
                marker=marker,
                facecolors="none",
                edgecolors="white",
                linewidths=1.8,
                zorder=7,
            )
            ax.scatter(
                [display_point.area],
                [display_point.mse],
                s=base_size,
                marker=marker,
                color=point_color,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.8,
                zorder=8,
            )
            ax.scatter(
                [display_point.area],
                [display_point.mse],
                s=base_size + 8,
                marker=marker,
                facecolors="none",
                edgecolors=point_color,
                linewidths=1.1,
                zorder=9,
            )

    for ref_point in ref_points:
        point_color = color_by_activation.get(ref_point.activation)
        if point_color is None:
            continue
        marker = REF_MARKER_BY_LABEL.get(ref_point.ref_label, "X")
        ax.scatter(
            [ref_point.area],
            [ref_point.mse],
            s=74,
            marker=marker,
            color=point_color,
            alpha=0.92,
            edgecolors="black",
            linewidths=0.85,
            zorder=10,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Hardware Area Cost (um^2)")
    ax.set_ylabel(f"{curve.upper()} MSE")
    if title is None:
        title = (
            f"Combined Pareto AREA Overlay ({curve}) | degree_max={degree_max} | "
            f"{len(sources)} result folders"
        )
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)

    finite_x = [point.area for point in points if math.isfinite(point.area)]
    finite_x.extend(point.area for point in ref_points if math.isfinite(point.area))
    if finite_x:
        x_min = min(finite_x)
        x_max = max(finite_x)
        x_span = max(x_max - x_min, 1e-12)
        x_upper = max(x_max + 0.02 * x_span, float(math.ceil(x_max / 1000.0) * 1000.0))
        ax.set_xlim(x_min - 0.04 * x_span, x_upper)

    note = (
        "Color = activation, marker = source folder / max-error, "
        "haloed line+marker = recomputed Pareto frontier, labels = segment budget SXX"
    )
    if ref_points:
        note += ", filled ref markers = reference data"
    note += ", coincident frontier duplicates are collapsed"
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.2),
    )

    label_points: list[LabelPoint] = []
    for display_points in frontier_display_by_activation.values():
        for display_point in display_points:
            label_points.append(
                LabelPoint(
                    activation=display_point.activation,
                    area=display_point.area,
                    mse=display_point.mse,
                    is_frontier=True,
                    label_text=display_point.label_text,
                )
            )

    if label_all_points:
        label_points.extend(
            [
                LabelPoint(
                    activation=point.activation,
                    area=point.area,
                    mse=point.mse,
                    is_frontier=False,
                    label_text=f"S{point.segment_budget:02d}",
                )
                for point in points
                if not point.is_frontier
            ]
        )
    annotate_segment_labels(ax, label_points, color_by_activation)

    activation_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="-",
            color=color_by_activation[activation],
            markerfacecolor=color_by_activation[activation],
            markeredgecolor="black",
            markeredgewidth=0.35,
            linewidth=2.2,
            label=activation,
        )
        for activation in activation_order
    ]
    source_handles = [
        Line2D(
            [0], [0],
            marker=marker_by_result[result_name],
            linestyle="none",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label=format_source_label(source_by_name[result_name], duplicate_max_error_labels),
        )
        for result_name in ordered_source_keys
    ]
    style_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#808080",
            markeredgecolor="black",
            markeredgewidth=0.25,
            alpha=0.22,
            markersize=5,
            label="Non-frontier point",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=6,
            label="Pareto frontier point",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.8,
            label="Pareto frontier line",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="xN in label = collapsed duplicate frontier points",
        ),
    ]
    ref_handles = []
    seen_ref_labels = []
    for ref_point in ref_points:
        if ref_point.ref_label in seen_ref_labels:
            continue
        seen_ref_labels.append(ref_point.ref_label)
        ref_handles.append(
            Line2D(
                [0], [0],
                marker=REF_MARKER_BY_LABEL.get(ref_point.ref_label, "X"),
                linestyle="none",
                color="black",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
                label=f"Ref {ref_point.ref_label}",
            )
        )

    legend1 = ax.legend(
        handles=activation_handles,
        title="Activation",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        framealpha=0.92,
        fontsize=9,
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=source_handles + ref_handles + style_handles,
        title="Run / Style",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.76),
        framealpha=0.92,
        fontsize=9,
        ncol=2 if len(source_handles) > 8 else 1,
    )
    ax.add_artist(legend2)

    fig.subplots_adjust(left=0.08, right=0.80, bottom=0.10, top=0.92)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_paths = discover_csvs(args.paths)

    try:
        loaded = [load_points(csv_path, curve=args.curve) for csv_path in csv_paths]
        activation_order, points, sources, degree_max = validate_sources(loaded)
        frontier_by_activation = mark_frontiers(points)
        ref_points = load_reference_points(sources, set(activation_order))
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    output_parent = sources[0].result_dir.parent
    out_path = args.out if args.out is not None else output_parent / DEFAULT_OUT_NAME
    csv_out = args.csv_out if args.csv_out is not None else output_parent / DEFAULT_CSV_OUT_NAME
    frontier_csv_out = (
        args.frontier_csv_out
        if args.frontier_csv_out is not None
        else output_parent / DEFAULT_FRONTIER_CONFIG_CSV_OUT_NAME
    )
    text_out = (
        args.text_out
        if args.text_out is not None
        else output_parent / DEFAULT_TEXT_OUT_NAME
    )
    label_all_points = bool(args.label_all_points)
    upper_degree_n0123_lookup = build_upper_degree_n0123_lookup(points)

    write_points_csv(csv_out, points, upper_degree_n0123_lookup)
    write_points_csv(
        frontier_csv_out,
        [point for point in points if point.is_frontier],
        upper_degree_n0123_lookup,
    )
    write_text_summary(text_out, points, sources, frontier_by_activation, args.curve)

    try:
        plot_points(
            points=points,
            ref_points=ref_points,
            activation_order=activation_order,
            sources=sources,
            frontier_by_activation=frontier_by_activation,
            out_path=out_path,
            curve=args.curve,
            degree_max=degree_max,
            dpi=args.dpi,
            title=args.title,
            label_all_points=label_all_points,
        )
    except ModuleNotFoundError as exc:
        print(
            "FAIL: matplotlib is required to render the combined plot. "
            "Please run this script in the project environment where matplotlib is installed.",
            file=sys.stderr,
        )
        print(f"Saved all-points CSV: {csv_out}")
        print(f"Saved frontier CSV: {frontier_csv_out}")
        print(f"Saved text summary: {text_out}")
        return 1

    num_frontier_points = sum(len(points_) for points_ in frontier_by_activation.values())
    print(f"Loaded {len(csv_paths)} CSV files.")
    print(f"Plotted {len(points)} points across {len(activation_order)} activations.")
    print(f"Marked {num_frontier_points} Pareto-frontier points.")
    print(f"Saved plot: {out_path}")
    print(f"Saved all-points CSV: {csv_out}")
    print(f"Saved frontier CSV: {frontier_csv_out}")
    print(f"Saved text summary: {text_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
