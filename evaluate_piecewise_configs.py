#!/usr/bin/env python3
"""
Evaluate saved piecewise polynomial configs against true activation functions.

Source data comes from `pareto_optimal_all/` under each `runs/seg_XX/` folder by
default. With `--common-only`, configs are read from `common_area_config/`
instead to save time. Evaluated results are written under a sibling tree named
`<results_root>_eval`.

For each `seg_XX`, the output structure is:
  - `pareto_optimal_all_eval/`
  - `pareto_optimal_feasible_eval/`
  - `common_config/`
  - `min_area_config/`

With `--common-only`, only `common_config/` is generated for each segment.
Every generated category contains both `eval.csv` and
`global_x_coefficients.csv`. The latter rewrites the saved local-coordinate
polynomials in ascending powers of physical `x` and is checked against the
original local-coordinate evaluation.

Category routing is based on degree-distribution signatures, using the config
filename convention shared by `pareto_optimal_all/`, `pareto_optimal_feasible/`,
`common_area_config/`, and `min_area_config/`.

Usage:
    python3 evaluate_piecewise_configs.py max3_results_4
    python3 evaluate_piecewise_configs.py max3_results_8 --n-samples 8192
    python3 evaluate_piecewise_configs.py max3_results_8 --common-only
    python3 evaluate_piecewise_configs.py max3_results_8 --common-only --segments 8
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SAMPLES_DEFAULT = 4096

CATEGORY_SOURCE_DIRS = {
    "pareto_optimal_all_eval": "pareto_optimal_all",
    "pareto_optimal_feasible_eval": "pareto_optimal_feasible",
    "common_config": "common_area_config",
    "min_area_config": "min_area_config",
}
SOURCE_CATEGORY = "pareto_optimal_all_eval"

ACTIVATION_COMPARE_ORDER = ["GELU", "Sigmoid", "SiLU", "Tanh"]

REFERENCE_COMPARISON_CASES = [
    {
        "case_label": "8-BP",
        "ref_breakpoints": 8,
        "matched_segment_budget": 7,
        "shared_bound": [0, 0, 2, 5],
        "linear_ref_label": "8BP-1st",
        "quadratic_ref_label": "8BP-2nd",
        "linear_ref_area": 30358.4594,
        "quadratic_ref_area": 31112.0988,
        "activations": {
            "GELU": {"linear_mse": 3.65e-6, "quadratic_mse": 3.78e-7},
            "Sigmoid": {"linear_mse": None, "quadratic_mse": None},
            "SiLU": {"linear_mse": 4.27e-5, "quadratic_mse": 1.35e-6},
            "Tanh": {"linear_mse": 1.37e-5, "quadratic_mse": 9.28e-7},
        },
    },
    {
        "case_label": "16-BP",
        "ref_breakpoints": 16,
        "matched_segment_budget": 15,
        "shared_bound": [0, 0, 2, 13],
        "linear_ref_label": "16BP-1st",
        "quadratic_ref_label": "16BP-2nd",
        "linear_ref_area": 32520.6594,
        "quadratic_ref_area": 33306.0988,
        "activations": {
            "GELU": {"linear_mse": 1.89e-7, "quadratic_mse": 9.07e-9},
            "Sigmoid": {"linear_mse": 2.88e-7, "quadratic_mse": 6.50e-9},
            "SiLU": {"linear_mse": None, "quadratic_mse": None},
            "Tanh": {"linear_mse": 4.26e-7, "quadratic_mse": 1.02e-8},
        },
    },
]


# ---------------------------------------------------------------------------
# Activation functions (self-contained, no dependency on optimize_piecewise_activations.py)
# ---------------------------------------------------------------------------

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x / (1 + np.exp(-x))


def tanh_act(x):
    return np.tanh(x)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def mish(x):
    return x * np.tanh(softplus(x))


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softsign(x):
    return x / (1 + np.abs(x))


def hardswish(x):
    return x * np.clip(x + 3, 0, 6) / 6


def exp_act(x):
    return np.exp(x)


ACTIVATION_FUNCTIONS = {
    "GELU": gelu,
    "Sigmoid": sigmoid,
    "SiLU": silu,
    "Tanh": tanh_act,
    "Mish": mish,
    "Softplus": softplus,
    "ELU": elu,
    "SELU": selu,
    "Softsign": softsign,
    "HardSwish": hardswish,
    "Exp": exp_act,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def reset_dir(path):
    """Remove a generated directory tree and recreate it empty."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_category_sources(seg_dir):
    """Build source-directory metadata and filename signatures for one seg_XX."""
    seg_dir = Path(seg_dir)
    category_meta = {}
    for out_name, source_name in CATEGORY_SOURCE_DIRS.items():
        source_dir = seg_dir / source_name
        csv_files = sorted(source_dir.glob("*.csv")) if source_dir.is_dir() else []
        signatures = {p.name for p in csv_files}
        category_meta[out_name] = {
            "source_name": source_name,
            "source_dir": source_dir,
            "csv_files": csv_files,
            "signatures": signatures,
        }
    return category_meta


def segment_index_from_name(seg_label):
    """Parse `seg_07` -> 7."""
    match = re.search(r"(\d+)", str(seg_label))
    if not match:
        raise ValueError(f"Could not parse segment index from '{seg_label}'")
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Config CSV reader
# ---------------------------------------------------------------------------

def load_config_csv(filepath):
    """
    Load a piecewise polynomial config from CSV.

    CSV schema (from optimize_piecewise_activations.py io_save_config_csv):
        type,segment_idx,point_idx,value
        breakpoint,-1,i,<x_value>
        degree,j,-1,<degree>
        coeff,j,k,<coeff_value>
    """
    filepath = Path(filepath)
    breakpoints = {}
    degrees = {}
    coeffs = {}

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rtype = row["type"].strip()
            seg_idx = int(row["segment_idx"])
            pt_idx = int(row["point_idx"])
            value = float(row["value"])

            if rtype == "breakpoint":
                breakpoints[pt_idx] = value
            elif rtype == "degree":
                degrees[seg_idx] = int(value)
            elif rtype == "coeff":
                coeffs[(seg_idx, pt_idx)] = value

    num_breakpoints = len(breakpoints)
    bp_array = np.array([breakpoints[i] for i in range(num_breakpoints)], dtype=np.float64)

    num_segments = num_breakpoints - 1
    deg_list = [degrees[j] for j in range(num_segments)]

    coeff_list = []
    for j in range(num_segments):
        d = deg_list[j]
        c = np.array([coeffs[(j, k)] for k in range(d + 1)], dtype=np.float64)
        coeff_list.append(c)

    activation_name = filepath.stem.split("_")[0]

    return {
        "breakpoints": bp_array,
        "degrees": deg_list,
        "coeffs": coeff_list,
        "num_segments": num_segments,
        "activation_name": activation_name,
        "filename": filepath.name,
        "filepath": str(filepath),
    }


# ---------------------------------------------------------------------------
# Piecewise polynomial conversion and evaluation (numpy only)
# ---------------------------------------------------------------------------

def segment_ids_for_x(x, breakpoints, num_segments):
    """Return the hard-segment assignment shared by both evaluation paths."""
    seg_ids = np.searchsorted(breakpoints[1:-1], x, side="right").clip(0, num_segments - 1)
    seg_ids[np.isclose(x, breakpoints[-1], atol=1e-6)] = num_segments - 1
    return seg_ids


def local_to_global_coefficients(local_coeffs, x_left, x_right):
    """Compose p(z) with z=2(x-left)/(right-left)-1.

    Both input and output coefficients are in ascending power order. NumPy's
    Polynomial composition performs the change of basis; this is floating-point
    algebra, not fixed-point or bit-accurate hardware quantization.
    """
    width = float(x_right) - float(x_left)
    if width <= 0.0:
        raise ValueError(f"Segment width must be positive, got [{x_left}, {x_right}]")

    z_of_x = np.polynomial.Polynomial([
        -(float(x_left) + float(x_right)) / width,
        2.0 / width,
    ])
    local_poly = np.polynomial.Polynomial(np.asarray(local_coeffs, dtype=np.float64))
    return np.asarray(local_poly(z_of_x).coef, dtype=np.float64)


def convert_config_to_global_coefficients(breakpoints, coeffs):
    """Convert every segment in one config to ascending global-x powers."""
    return [
        local_to_global_coefficients(coeff, breakpoints[j], breakpoints[j + 1])
        for j, coeff in enumerate(coeffs)
    ]


def evaluate_piecewise_poly(x, breakpoints, degrees, coeffs):
    """Evaluate a hard-segmented piecewise polynomial using local coords in [-1, 1]."""
    k_segments = len(degrees)
    seg_ids = segment_ids_for_x(x, breakpoints, k_segments)

    y_hat = np.zeros_like(x, dtype=np.float64)
    for j in range(k_segments):
        mask = seg_ids == j
        if not np.any(mask):
            continue
        b_left, b_right = breakpoints[j], breakpoints[j + 1]
        denom = max(b_right - b_left, 1e-12)
        x_local = 2.0 * (x[mask] - b_left) / denom - 1.0
        y_hat[mask] = np.polynomial.polynomial.polyval(x_local, coeffs[j])

    return y_hat, seg_ids


def evaluate_piecewise_poly_global_x(x, breakpoints, degrees, global_coeffs):
    """Evaluate a hard-segmented polynomial whose coefficients use physical x."""
    k_segments = len(degrees)
    seg_ids = segment_ids_for_x(x, breakpoints, k_segments)

    y_hat = np.zeros_like(x, dtype=np.float64)
    for j in range(k_segments):
        mask = seg_ids == j
        if np.any(mask):
            y_hat[mask] = np.polynomial.polynomial.polyval(x[mask], global_coeffs[j])

    return y_hat, seg_ids


def verify_global_coefficient_conversion(breakpoints, coeffs, global_coeffs):
    """Check both bases at 65 points inside every segment."""
    max_abs_diff = 0.0
    for j, (local_coeff, global_coeff) in enumerate(zip(coeffs, global_coeffs)):
        x_left = float(breakpoints[j])
        x_right = float(breakpoints[j + 1])
        x = np.linspace(x_left, x_right, 65, dtype=np.float64)
        z = 2.0 * (x - x_left) / (x_right - x_left) - 1.0
        local_y = np.polynomial.polynomial.polyval(z, local_coeff)
        global_y = np.polynomial.polynomial.polyval(x, global_coeff)
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(local_y - global_y))))
    return max_abs_diff


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_hat):
    err = y_true - y_hat
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    return {
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": mae,
        "sq_avg_ae": float(mae ** 2),
        "max_ae": float(np.max(abs_err)),
    }


def degree_type_counts(degrees, degree_max=3):
    """Count exact degree usage [n0, n1, n2, n3]."""
    counts = [0] * (degree_max + 1)
    for d in degrees:
        d_int = int(d)
        if 0 <= d_int <= degree_max:
            counts[d_int] += 1
    return counts


def degree_counts_to_c_counts(degree_counts):
    """Convert exact degree counts [n0, n1, n2, n3] -> cumulative counts (c0,c1,c2,c3)."""
    total = sum(degree_counts)
    c_counts = []
    running_sum = total
    for n in degree_counts:
        c_counts.append(running_sum)
        running_sum -= n
    while len(c_counts) < 4:
        c_counts.append(0)
    return tuple(c_counts[:4])


def hardware_area_from_c_counts(c0, c1, c2, c3):
    """Mirror the area model used in optimize_piecewise_activations.py."""
    c0 = float(c0)
    c1 = float(c1)
    c2 = float(c2)
    c3 = float(c3)

    if c3 > 0:
        max_degree = 3.0
    elif c2 > 0:
        max_degree = 2.0
    elif c1 > 0:
        max_degree = 1.0
    else:
        max_degree = 0.0

    return float(
        60.9 * c0
        + 57.06 * c1
        + 56.37 * c2
        + 57.07 * c3
        + 162.49
        + 960.0 * max_degree
    )


def hardware_area_from_degrees(degrees):
    """Compute hardware area directly from a degree list."""
    degree_counts = degree_type_counts(degrees, degree_max=3)
    c_counts = degree_counts_to_c_counts(degree_counts)
    return hardware_area_from_c_counts(*c_counts)


def factor_vs_reference(our_value, ref_value):
    """Return multiplicative difference and direction versus a reference value."""
    if our_value is None or ref_value is None:
        return None, ""
    our_value = float(our_value)
    ref_value = float(ref_value)
    if our_value <= 0.0 or ref_value <= 0.0:
        return None, ""
    if np.isclose(our_value, ref_value):
        return 1.0, "equal"
    if our_value < ref_value:
        return ref_value / our_value, "down"
    return our_value / ref_value, "up"


def percent_vs_reference(our_value, ref_value):
    """Return percentage difference and direction versus a reference value."""
    if our_value is None or ref_value is None:
        return None, ""
    our_value = float(our_value)
    ref_value = float(ref_value)
    if ref_value == 0.0:
        return None, ""
    if np.isclose(our_value, ref_value):
        return 0.0, "equal"
    if our_value < ref_value:
        return (ref_value - our_value) / ref_value * 100.0, "down"
    return (our_value - ref_value) / ref_value * 100.0, "up"


def format_factor_text(value, direction):
    """Format multiplicative comparisons as used in the paper tables."""
    if value is None or not direction:
        return ""
    if direction == "equal":
        return "1.0x"
    return f"{value:.1f}x{direction}"


def format_percent_text(value, direction):
    """Format percentage comparisons as used in the paper tables."""
    if value is None or not direction:
        return ""
    if direction == "equal":
        return "0.0%"
    return f"{value:.1f}%{direction}"


# ---------------------------------------------------------------------------
# Evaluate a single config
# ---------------------------------------------------------------------------

def evaluate_config(config, n_samples):
    """Evaluate local and converted global-x forms on the config's own domain."""
    act_name = config["activation_name"]
    act_func = ACTIVATION_FUNCTIONS.get(act_name)
    if act_func is None:
        print(f"  WARNING: Unknown activation '{act_name}' in {config['filename']}, skipping.")
        return None

    x_min = float(config["breakpoints"][0])
    x_max = float(config["breakpoints"][-1])
    x = np.linspace(x_min, x_max, n_samples, dtype=np.float64)
    y_true = act_func(x)
    y_hat, seg_ids = evaluate_piecewise_poly(x, config["breakpoints"], config["degrees"], config["coeffs"])
    metrics = compute_metrics(y_true, y_hat)
    global_coeffs = convert_config_to_global_coefficients(config["breakpoints"], config["coeffs"])
    y_hat_global, global_seg_ids = evaluate_piecewise_poly_global_x(
        x,
        config["breakpoints"],
        config["degrees"],
        global_coeffs,
    )
    if not np.array_equal(seg_ids, global_seg_ids):
        raise ValueError("Local and global-x evaluation selected different segments")

    global_metrics = compute_metrics(y_true, y_hat_global)
    local_global_max_abs_diff = max(
        float(np.max(np.abs(y_hat - y_hat_global))),
        verify_global_coefficient_conversion(
            config["breakpoints"],
            config["coeffs"],
            global_coeffs,
        ),
    )
    if local_global_max_abs_diff > 1e-9:
        raise ValueError(
            "Local/global-x conversion check failed: "
            f"max_abs_diff={local_global_max_abs_diff:.3e}"
        )

    seg_metrics = []
    for j in range(config["num_segments"]):
        mask = seg_ids == j
        if np.any(mask):
            seg_metric = compute_metrics(y_true[mask], y_hat[mask])
            seg_metric["segment"] = j
            seg_metric["degree"] = config["degrees"][j]
            seg_metric["n_points"] = int(np.sum(mask))
            seg_metric["x_range"] = (
                float(config["breakpoints"][j]),
                float(config["breakpoints"][j + 1]),
            )
            seg_metrics.append(seg_metric)

    return {
        "activation": act_name,
        "filename": config["filename"],
        "filepath": config["filepath"],
        "num_segments": config["num_segments"],
        "degrees": config["degrees"],
        "breakpoints": config["breakpoints"],
        "global_coeffs": global_coeffs,
        "x_min": x_min,
        "x_max": x_max,
        **metrics,
        **{f"global_x_{name}": value for name, value in global_metrics.items()},
        "local_global_max_abs_diff": local_global_max_abs_diff,
        "segment_metrics": seg_metrics,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_config_results(results, out_dir, n_samples):
    """Generate one plot per config: left = fit vs true, right = error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for res in results:
        act_name = res["activation"]
        act_func = ACTIVATION_FUNCTIONS[act_name]
        x = np.linspace(res["x_min"], res["x_max"], n_samples, dtype=np.float64)
        y_true = act_func(x)

        cfg = load_config_csv(res["filepath"])
        y_hat, _ = evaluate_piecewise_poly(x, cfg["breakpoints"], cfg["degrees"], cfg["coeffs"])
        plot_stem = Path(res["filename"]).stem

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ax1.plot(x, y_true, "k-", lw=1.5, label="True", alpha=0.7)
        ax1.plot(x, y_hat, "r--", lw=1.2, label="Approx")
        for bp in cfg["breakpoints"]:
            ax1.axvline(bp, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax1.set_title(f"{act_name} — {plot_stem}")
        ax1.legend(fontsize=8)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

        err = y_true - y_hat
        ax2.plot(x, err, "b-", lw=0.8)
        ax2.axhline(0, color="gray", ls="-", lw=0.5)
        for bp in cfg["breakpoints"]:
            ax2.axvline(bp, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax2.set_title(f"Error — MSE={res['mse']:.2e}, SqAvgAE={res['sq_avg_ae']:.2e}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Error")

        plt.tight_layout()
        save_path = out_dir / f"eval_{plot_stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_summary_csv(all_results, output_path):
    """Write a summary eval.csv for one category (or one aggregate view)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "activation",
        "num_segments",
        "degrees",
        "mse",
        "rmse",
        "mae",
        "sq_avg_ae",
        "max_ae",
        "global_x_mse",
        "global_x_rmse",
        "global_x_mae",
        "global_x_sq_avg_ae",
        "global_x_max_ae",
        "local_global_max_abs_diff",
        "x_min",
        "x_max",
        "filename",
        "source_filepath",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(all_results, key=lambda row: (row["num_segments"], row["activation"], row["filename"])):
            writer.writerow({
                "activation": r["activation"],
                "num_segments": r["num_segments"],
                "degrees": str(r["degrees"]),
                "mse": r["mse"],
                "rmse": r["rmse"],
                "mae": r["mae"],
                "sq_avg_ae": r["sq_avg_ae"],
                "max_ae": r["max_ae"],
                "global_x_mse": r["global_x_mse"],
                "global_x_rmse": r["global_x_rmse"],
                "global_x_mae": r["global_x_mae"],
                "global_x_sq_avg_ae": r["global_x_sq_avg_ae"],
                "global_x_max_ae": r["global_x_max_ae"],
                "local_global_max_abs_diff": r["local_global_max_abs_diff"],
                "x_min": r["x_min"],
                "x_max": r["x_max"],
                "filename": r["filename"],
                "source_filepath": r["filepath"],
            })


def save_global_coefficients_csv(results, output_path):
    """Write one row per segment with breakpoints and global-x coefficients."""
    if not results:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_degree = max(max(result["degrees"]) for result in results)
    coefficient_columns = [f"coeff_x{order}" for order in range(max_degree + 1)]
    fieldnames = [
        "activation",
        "source_config",
        "segment_idx",
        "breakpoint_left",
        "breakpoint_right",
        "degree",
        *coefficient_columns,
        "basis",
        "config_local_global_max_abs_diff",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda row: (row["activation"], row["filename"])):
            for segment_idx, (degree, coeffs) in enumerate(
                zip(result["degrees"], result["global_coeffs"])
            ):
                row = {
                    "activation": result["activation"],
                    "source_config": result["filename"],
                    "segment_idx": segment_idx,
                    "breakpoint_left": format(result["breakpoints"][segment_idx], ".17g"),
                    "breakpoint_right": format(result["breakpoints"][segment_idx + 1], ".17g"),
                    "degree": degree,
                    "basis": "global_x_power_ascending",
                    "config_local_global_max_abs_diff": format(
                        result["local_global_max_abs_diff"], ".17g"
                    ),
                }
                for order, column in enumerate(coefficient_columns):
                    value = coeffs[order] if order < len(coeffs) else 0.0
                    row[column] = format(value, ".17g")
                writer.writerow(row)


def save_reference_compare_csv(common_results_by_segment, output_path):
    """Write compare.csv joining evaluated common configs with fixed Flex-SFU references."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_label",
        "ref_breakpoints",
        "matched_segment_budget",
        "shared_bound",
        "activation",
        "our_mse",
        "our_area",
        "our_degrees",
        "our_degree_counts",
        "our_filename",
        "our_source_filepath",
        "linear_ref_label",
        "linear_ref_mse",
        "linear_ref_area",
        "mse_vs_linear_factor",
        "mse_vs_linear_direction",
        "mse_vs_linear_text",
        "area_vs_linear_pct",
        "area_vs_linear_direction",
        "area_vs_linear_text",
        "quadratic_ref_label",
        "quadratic_ref_mse",
        "quadratic_ref_area",
        "mse_vs_quadratic_factor",
        "mse_vs_quadratic_direction",
        "mse_vs_quadratic_text",
        "area_vs_quadratic_pct",
        "area_vs_quadratic_direction",
        "area_vs_quadratic_text",
    ]

    rows = []
    for case_idx, case in enumerate(REFERENCE_COMPARISON_CASES):
        seg_budget = int(case["matched_segment_budget"])
        common_results = common_results_by_segment.get(seg_budget, [])

        best_by_activation = {}
        for result in common_results:
            act_name = result["activation"]
            prev = best_by_activation.get(act_name)
            if prev is None or float(result["mse"]) < float(prev["mse"]):
                best_by_activation[act_name] = result

        for act_idx, act_name in enumerate(ACTIVATION_COMPARE_ORDER):
            ref_metrics = case["activations"].get(act_name, {})
            our_result = best_by_activation.get(act_name)
            our_mse = float(our_result["mse"]) if our_result is not None else None
            our_area = hardware_area_from_degrees(our_result["degrees"]) if our_result is not None else None
            our_deg_counts = (
                "/".join(str(v) for v in degree_type_counts(our_result["degrees"], degree_max=3))
                if our_result is not None
                else ""
            )

            linear_ref_mse = ref_metrics.get("linear_mse")
            quadratic_ref_mse = ref_metrics.get("quadratic_mse")
            linear_ref_area = float(case["linear_ref_area"])
            quadratic_ref_area = float(case["quadratic_ref_area"])

            mse_vs_linear_factor, mse_vs_linear_direction = factor_vs_reference(our_mse, linear_ref_mse)
            mse_vs_quadratic_factor, mse_vs_quadratic_direction = factor_vs_reference(our_mse, quadratic_ref_mse)
            area_vs_linear_pct, area_vs_linear_direction = percent_vs_reference(our_area, linear_ref_area)
            area_vs_quadratic_pct, area_vs_quadratic_direction = percent_vs_reference(our_area, quadratic_ref_area)

            rows.append({
                "case_label": case["case_label"],
                "ref_breakpoints": int(case["ref_breakpoints"]),
                "matched_segment_budget": seg_budget,
                "shared_bound": str(case["shared_bound"]),
                "activation": act_name,
                "our_mse": our_mse if our_mse is not None else "",
                "our_area": our_area if our_area is not None else "",
                "our_degrees": str(our_result["degrees"]) if our_result is not None else "",
                "our_degree_counts": our_deg_counts,
                "our_filename": our_result["filename"] if our_result is not None else "",
                "our_source_filepath": our_result["filepath"] if our_result is not None else "",
                "linear_ref_label": case["linear_ref_label"],
                "linear_ref_mse": linear_ref_mse if linear_ref_mse is not None else "",
                "linear_ref_area": linear_ref_area,
                "mse_vs_linear_factor": mse_vs_linear_factor if mse_vs_linear_factor is not None else "",
                "mse_vs_linear_direction": mse_vs_linear_direction,
                "mse_vs_linear_text": format_factor_text(mse_vs_linear_factor, mse_vs_linear_direction),
                "area_vs_linear_pct": area_vs_linear_pct if area_vs_linear_pct is not None else "",
                "area_vs_linear_direction": area_vs_linear_direction,
                "area_vs_linear_text": format_percent_text(area_vs_linear_pct, area_vs_linear_direction),
                "quadratic_ref_label": case["quadratic_ref_label"],
                "quadratic_ref_mse": quadratic_ref_mse if quadratic_ref_mse is not None else "",
                "quadratic_ref_area": quadratic_ref_area,
                "mse_vs_quadratic_factor": mse_vs_quadratic_factor if mse_vs_quadratic_factor is not None else "",
                "mse_vs_quadratic_direction": mse_vs_quadratic_direction,
                "mse_vs_quadratic_text": format_factor_text(mse_vs_quadratic_factor, mse_vs_quadratic_direction),
                "area_vs_quadratic_pct": area_vs_quadratic_pct if area_vs_quadratic_pct is not None else "",
                "area_vs_quadratic_direction": area_vs_quadratic_direction,
                "area_vs_quadratic_text": format_percent_text(area_vs_quadratic_pct, area_vs_quadratic_direction),
                "_case_order": case_idx,
                "_act_order": act_idx,
            })

    rows.sort(key=lambda row: (row["_case_order"], row["_act_order"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = {k: v for k, v in row.items() if not k.startswith("_")}
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_segment_header(seg_label):
    print(f"\n{'=' * 90}")
    print(f"  {seg_label}")
    print(f"{'=' * 90}")
    print(
        f"  {'Activation':<12} {'Degrees':<28} {'MSE':>12} {'RMSE':>12} "
        f"{'MAE':>12} {'SqAvgAE':>12} {'BasisDiff':>12}"
    )
    print(f"  {'-' * 100}")


def print_result_row(result):
    deg_str = str(result["degrees"])
    if len(deg_str) > 26:
        deg_str = deg_str[:23] + "..."
    print(
        f"  {result['activation']:<12} {deg_str:<28} "
        f"{result['mse']:>12.6e} {result['rmse']:>12.6e} {result['mae']:>12.6e} "
        f"{result['sq_avg_ae']:>12.6e} {result['local_global_max_abs_diff']:>12.3e}"
    )


def print_category_summary(seg_label, category_results, category_meta, category_dirs):
    print(f"\n  Routed outputs for {seg_label}:")
    for out_name in category_results:
        matched = len(category_results[out_name])
        expected = len(category_meta[out_name]["csv_files"])
        source_name = category_meta[out_name]["source_name"]
        print(
            f"    {out_name:<28} matched {matched:>4}/{expected:<4} "
            f"from {source_name}/ -> {category_dirs[out_name]}/"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved polynomial configs and route outputs by config membership.",
    )
    parser.add_argument(
        "results_root",
        type=str,
        help="Path to results root (e.g., max3_results_4). Must contain runs/seg_XX/pareto_optimal_all/ or common_area_config/.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES_DEFAULT,
        help=f"Number of evaluation points (default: {N_SAMPLES_DEFAULT})",
    )
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="Only evaluate configs from common_area_config/ to save time.",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        type=int,
        help="Only evaluate these segment counts (for example: --segments 8 16).",
    )

    args = parser.parse_args()
    results_root = Path(args.results_root)
    runs_dir = results_root / "runs"
    source_category = "common_config" if args.common_only else SOURCE_CATEGORY
    output_categories = ["common_config"] if args.common_only else list(CATEGORY_SOURCE_DIRS)

    if not runs_dir.is_dir():
        print(f"Error: {runs_dir} does not exist.")
        sys.exit(1)

    seg_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and re.match(r"seg_\d+", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if args.segments:
        requested_segments = set(args.segments)
        seg_dirs = [
            seg_dir
            for seg_dir in seg_dirs
            if segment_index_from_name(seg_dir.name) in requested_segments
        ]

    if not seg_dirs:
        requested = f" for --segments {args.segments}" if args.segments else ""
        print(f"No seg_XX directories found under {runs_dir}{requested}")
        sys.exit(1)

    eval_root = results_root.parent / f"{results_root.name}_eval"
    reset_dir(eval_root)

    print(f"Results root : {results_root}")
    print(f"Eval output  : {eval_root}")
    print(f"Source       : {CATEGORY_SOURCE_DIRS[source_category]}")
    print(f"Eval scope   : {'common configs only' if args.common_only else 'all pareto_optimal_all configs'}")
    print(f"Segments     : {len(seg_dirs)} ({seg_dirs[0].name} .. {seg_dirs[-1].name})")
    print(f"Eval range   : each config's breakpoint domain, {args.n_samples} samples")

    all_results = []
    evaluated_segment_count = 0
    common_results_by_segment = {}

    for seg_dir in seg_dirs:
        seg_label = seg_dir.name
        category_meta = discover_category_sources(seg_dir)
        source_meta = category_meta[source_category]
        source_dir = source_meta["source_dir"]
        source_csv_files = source_meta["csv_files"]
        source_name = source_meta["source_name"]

        if not source_dir.is_dir():
            print(f"\n  Skipping {seg_label}: no {source_name}/")
            continue
        if not source_csv_files:
            print(f"\n  Skipping {seg_label}: no CSV files in {source_name}/")
            continue

        seg_eval_root = eval_root / seg_label
        reset_dir(seg_eval_root)
        category_dirs = {}
        for out_name in output_categories:
            out_dir = seg_eval_root / out_name
            out_dir.mkdir(parents=True, exist_ok=True)
            category_dirs[out_name] = out_dir

        print_segment_header(seg_label)

        category_results = {out_name: [] for out_name in output_categories}
        seg_results = []

        for csv_path in source_csv_files:
            try:
                config = load_config_csv(csv_path)
                result = evaluate_config(config, n_samples=args.n_samples)
                if result is None:
                    continue

                seg_results.append(result)
                all_results.append(result)
                print_result_row(result)

                for out_name in output_categories:
                    meta = category_meta[out_name]
                    if result["filename"] in meta["signatures"]:
                        category_results[out_name].append(result)

            except Exception as exc:
                print(f"  ✗ Error loading {csv_path.name}: {exc}")

        if not seg_results:
            print(f"\n  No configs were successfully evaluated in {seg_label}.")
            continue

        for out_name, results in category_results.items():
            save_summary_csv(results, category_dirs[out_name] / "eval.csv")
            if results:
                global_csv_path = category_dirs[out_name] / "global_x_coefficients.csv"
                save_global_coefficients_csv(results, global_csv_path)
                print(f"  Global-x coefficients: {global_csv_path}")
                plot_config_results(results, category_dirs[out_name], args.n_samples)

        seg_idx = segment_index_from_name(seg_label)
        common_results_by_segment[seg_idx] = list(category_results.get("common_config", []))

        print_category_summary(seg_label, category_results, category_meta, category_dirs)
        evaluated_segment_count += 1

    if not all_results:
        print("\nNo configs were successfully evaluated.")
        sys.exit(1)

    save_summary_csv(all_results, eval_root / "eval.csv")
    print(f"\nAggregate summary CSV saved to {eval_root / 'eval.csv'}")
    compare_csv_path = eval_root / "compare.csv"
    save_reference_compare_csv(common_results_by_segment, compare_csv_path)
    print(f"Reference comparison CSV saved to {compare_csv_path}")
    scope_label = "common_area_config" if args.common_only else "pareto_optimal_all"
    print(f"Done. {len(all_results)} {scope_label} configs evaluated across {evaluated_segment_count} segment counts.")


if __name__ == "__main__":
    main()
