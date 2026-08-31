"""
CPU-based piecewise polynomial fitting for activation approximation.

Current workflow:
- Stage 1 (per activation): Phase 1 breakpoint optimization + Phase 2 mixed-degree selection.
- Stage 2 (joint): Phase 3 common-area optimization across all activations.

Design choices:
- Hard segmentation is used for both fitting and inference paths.
- Breakpoints are discrete x-grid indices.
- Segment coefficients are solved by least squares (optional ridge regularization).

Dependencies: numpy, matplotlib, torch, tqdm, pandas
"""

import shutil
import logging
import argparse
import ast
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from tqdm import tqdm


# =============================================================================
# Logging
# =============================================================================
_log_file_handler = None
_logger = None

def setup_logging(log_dir, log_filename="run.log"):
    """Set up logging to file. Console output uses tqdm.write for progress bar compatibility."""
    global _log_file_handler, _logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    # Create logger
    _logger = logging.getLogger("uni_sfu")
    _logger.setLevel(logging.INFO)
    _logger.handlers.clear()  # Clear any existing handlers

    # File handler only - console output goes through tqdm.write in log()
    _log_file_handler = logging.FileHandler(log_path, mode='w')
    _log_file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    _log_file_handler.setFormatter(file_formatter)
    _logger.addHandler(_log_file_handler)

    return log_path


def log(msg, pbar=None, level="info"):
    """Log message to file and console (tqdm-safe)."""
    global _logger

    # Write to log file if logger is set up
    if _logger is not None:
        if level == "info":
            _logger.info(msg)
        elif level == "warning":
            _logger.warning(msg)
        elif level == "error":
            _logger.error(msg)

    # tqdm-safe console printing
    if pbar is None:
        tqdm.write(str(msg))
    else:
        pbar.write(str(msg))


def log_file_only(msg):
    """Log message to file only (no console output)."""
    global _logger
    if _logger is not None:
        _logger.info(msg)


# =============================================================================
# Device
# =============================================================================
device = torch.device("cpu")
# Device info is logged in main() after logging is set up


# =============================================================================
# Activation functions
# =============================================================================
def gelu(x):
    if isinstance(x, torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / np.pi, device=x.device)) *
                                          (x + 0.044715 * x**3)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    return 1 / (1 + np.exp(-x))


def silu(x):
    if isinstance(x, torch.Tensor):
        return x * torch.sigmoid(x)
    return x / (1 + np.exp(-x))


def tanh_act(x):
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    return np.tanh(x)


def softplus(x):
    if isinstance(x, torch.Tensor):
        return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp(x, min=0)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def mish(x):
    if isinstance(x, torch.Tensor):
        return x * torch.tanh(softplus(x))
    return x * np.tanh(softplus(x))


def elu(x, alpha=1.0):
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    if isinstance(x, torch.Tensor):
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softsign(x):
    if isinstance(x, torch.Tensor):
        return x / (1 + torch.abs(x))
    return x / (1 + np.abs(x))


def log_act(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    return np.log(x)


def exp_act(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x)
    return np.exp(x)


# HardSwish is excluded because it is a piecewise function that is too easy to fit, and there
# is no point of using an approximation for it. If you want to include it, simply uncomment the code below and add "HardSwish": hardswish to the ACTIVATION_FUNCTIONS dict.
# def hardswish(x):
#     if isinstance(x, torch.Tensor):
#         return x * torch.clamp(x + 3, min=0, max=6) / 6
#     return x * np.clip(x + 3, 0, 6) / 6


ACTIVATION_FUNCTIONS = {
    "GELU": gelu,
    "SiLU": silu,
    "Sigmoid": sigmoid,
    "Tanh": tanh_act,
    "Softplus": softplus,
    "ELU": elu,
}

# =============================================================================
# Helper functions (math, conversions, utilities)
# =============================================================================
DEFAULT_ACTIVATION_DOMAIN = (-8.0, 8.0)
ACTIVATION_DOMAINS = {
    # Avoid the log singularity at x=0 while preserving the standard positive-input domain.
    "Log": (1e-4, 8.0),
    "Softplus": (-8.0, 8.0),
    "Exp": (-5.0, 0.0),
}


def helper_activation_domain(activation_name):
    """Return the fit/eval x-domain for an activation."""
    return ACTIVATION_DOMAINS.get(str(activation_name), DEFAULT_ACTIVATION_DOMAIN)


def helper_degrees_to_counts(degrees):
    """
    Convert per-segment degrees -> global counts (c0,c1,c2,c3).

    Interpretation:
      - c0 = #segments (constant term always present)
      - c1 = #segments with degree>=1
      - c2 = #segments with degree>=2
      - c3 = #segments with degree>=3
    """
    degs = [int(d) for d in degrees]
    K = len(degs)
    c0 = K
    c1 = sum(1 for d in degs if d >= 1)
    c2 = sum(1 for d in degs if d >= 2)
    c3 = sum(1 for d in degs if d >= 3)
    return (c0, c1, c2, c3)


def helper_hardware_area(c0, c1, c2, c3):
    """
    Hardware area function.
    Update ONLY this function if your area equation changes (as long as it depends on counts).
    """
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

    return (
        60.9 * c0
        + 57.06 * c1
        + 56.37 * c2
        + 57.07 * c3
        + 162.49
        + 960.0 * max_degree
    )

def helper_boundary_x_from_index(x_grid: torch.Tensor, idx: int) -> torch.Tensor:
    """
    x_grid: 1D tensor length N (monotone increasing)
    idx: int in [0, N]
    Returns x-value of the boundary at that index using midpoint convention:
      idx=0   -> x[0]
      idx=N   -> x[N-1]
      else    -> 0.5*(x[idx-1] + x[idx])
    """
    N = x_grid.numel()
    if idx <= 0:
        return x_grid[0]
    if idx >= N:
        return x_grid[-1]
    return 0.5 * (x_grid[idx - 1] + x_grid[idx])


def helper_to_local_x(x: torch.Tensor, b_left: torch.Tensor, b_right: torch.Tensor) -> torch.Tensor:
    """Convert x to local coordinates in [-1, 1] within segment [b_left, b_right]."""
    denom = (b_right - b_left).clamp_min(1e-12)
    return 2.0 * (x - b_left) / denom - 1.0


def helper_vandermonde(x_local: torch.Tensor, degree: int) -> torch.Tensor:
    """Build Vandermonde matrix for polynomial fitting."""
    powers = torch.arange(degree + 1, device=x_local.device, dtype=x_local.dtype)
    return x_local[:, None] ** powers[None, :]  # (N, degree+1)


def helper_segments_from_bp_indices(bp_idx):
    """
    bp_idx: strictly increasing integer indices length B=S+1
            boundaries in [0, N], with bp_idx[0]=0 and bp_idx[-1]=N

    segment j covers [bp_idx[j], bp_idx[j+1]) (half-open, no overlap)
    """
    S = len(bp_idx) - 1
    return [(int(bp_idx[j]), int(bp_idx[j + 1])) for j in range(S)]


def helper_solve_segment_poly(
    x_seg: torch.Tensor,
    y_seg: torch.Tensor,
    degree: int,
    b_left: torch.Tensor,
    b_right: torch.Tensor,
    lam: float = 0.0,
):
    """
    Fit y_seg ~ poly(z) where z is local coord in [-1,1] over [b_left, b_right].
    Returns coeff (degree+1,) and segment MSE scalar.

    IMPORTANT:
      b_left/b_right MUST match the boundaries used in inference/export,
      otherwise you will see train/eval drift at high segment counts.
    """
    if x_seg.numel() == 0:
        raise ValueError("helper_solve_segment_poly got empty segment")

    lam_eff = float(lam)

    x_local = helper_to_local_x(x_seg, b_left, b_right)
    Phi = helper_vandermonde(x_local, degree)  # (n, d+1)

    A = Phi.T @ Phi
    if lam_eff > 0:
        A = A + lam_eff * torch.eye(A.shape[0], device=Phi.device, dtype=Phi.dtype)

    rhs = Phi.T @ y_seg
    coeff = torch.linalg.solve(A, rhs)

    y_hat = Phi @ coeff
    seg_mse = torch.mean((y_hat - y_seg) ** 2)
    return coeff, seg_mse


@torch.no_grad()
def helper_fit_all_segments(
    x: torch.Tensor,
    y: torch.Tensor,
    bp_idx: torch.Tensor,
    degrees,
    lam: float = 0.0,
    return_seg_ids: bool = False,
):
    """
    Half-open segments [l, r). Mixed-degree LS per segment.
    Uses midpoint boundary convention for BOTH fit + export/inference compatibility.

    Args:
      degrees: list[int] length K (K = num_segments)

    Returns:
      coeffs: list of tensors with shape (degrees[j]+1,)
      y_hat: (N,)
      total_mse: scalar tensor (global MSE across all points)
      mse: scalar tensor (same as total_mse; kept for compatibility)
      seg_ids (optional): (N,) long
    """
    N = x.numel()
    K = len(bp_idx) - 1
    if len(degrees) != K:
        raise ValueError(f"degrees has len {len(degrees)} but expected {K}")

    y_hat = torch.zeros_like(y)
    coeffs = []
    total_mse = torch.tensor(0.0, device=x.device, dtype=y.dtype)

    segs = helper_segments_from_bp_indices(bp_idx)
    seg_ids = None
    if return_seg_ids:
        seg_ids = torch.empty(N, device=x.device, dtype=torch.long)

    for j, (l, r) in enumerate(segs):
        d = int(degrees[j])
        x_seg = x[l:r]
        y_seg = y[l:r]
        if x_seg.numel() == 0:
            raise ValueError(f"Empty segment {j}: [{l},{r})")

        b_left = helper_boundary_x_from_index(x, l)
        b_right = helper_boundary_x_from_index(x, r)

        coeff, seg_mse = helper_solve_segment_poly(x_seg, y_seg, d, b_left, b_right, lam=lam)
        coeffs.append(coeff)
        seg_weight = float(x_seg.numel()) / float(N)
        total_mse = total_mse + seg_mse * seg_weight

        xl = helper_to_local_x(x_seg, b_left, b_right)
        Phi = helper_vandermonde(xl, d)
        y_hat[l:r] = Phi @ coeff

        if return_seg_ids:
            seg_ids[l:r] = j

    mse = total_mse
    if return_seg_ids:
        return coeffs, y_hat, total_mse, mse, seg_ids
    return coeffs, y_hat, total_mse, mse


@torch.no_grad()
def helper_predict_from_coeffs(
    x: torch.Tensor,
    b: torch.Tensor,
    coeffs,
    degrees,
):
    """
    Mixed-degree inference.

    Args:
      b: breakpoints in x-space (1D tensor length K+1)
      degrees: list[int] length K
      coeffs: list[tensor] length K with len = degrees[j]+1

    Returns:
      yhat: (N,)
      x_local_out: (N,)
      seg_ids: (N,)
    """
    K = b.numel() - 1
    if len(degrees) != K or len(coeffs) != K:
        raise ValueError("degrees/coeffs length mismatch with b")

    seg_ids = torch.bucketize(x, b[1:-1], right=True).clamp(0, K - 1)
    seg_ids = torch.where(
        torch.isclose(x, b[-1], atol=1e-6),
        torch.full_like(seg_ids, K - 1),
        seg_ids
    )

    yhat = torch.zeros_like(x)
    x_local_out = torch.zeros_like(x)

    for j in range(K):
        m = seg_ids == j
        if not m.any():
            continue
        d = int(degrees[j])
        xl = helper_to_local_x(x[m], b[j], b[j + 1])
        Phi = helper_vandermonde(xl, d)
        yhat[m] = Phi @ coeffs[j]
        x_local_out[m] = xl

    return yhat, x_local_out, seg_ids


def helper_fit_and_eval_with_breakpoints(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    bp_idx: torch.Tensor,
    breakpoints_x: torch.Tensor,
    degrees,
    lam: float = 0.0,
):
    """
    Fit coefficients on train grid and evaluate MSE on train/eval grids.
    """
    coeffs, y_train_fit, _, mse_train = helper_fit_all_segments(
        x_train, y_train, bp_idx, degrees, lam=lam
    )
    train_mse = float(mse_train.item())

    with torch.no_grad():
        y_eval_fit, _, _ = helper_predict_from_coeffs(x_eval, breakpoints_x, coeffs, degrees)
        eval_mse = float(torch.mean((y_eval_fit - y_eval) ** 2).item())

    return coeffs, y_train_fit, train_mse, eval_mse

# =============================================================================
# Phase 2: Mixed-degree selection via DP (minimize HARDWARE AREA under MSE budget)
# =============================================================================
def phase2_select_degrees_dp(
    bp_idx: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    degree_max: int,
    lam: float,
    max_error_budget: float,
    return_frontier: bool = False,
):
    """
    Fixed breakpoints (half-open segments).
    Choose degrees per segment to minimize HARDWARE AREA subject to:
      total_mse <= budget

    Budget is determined by:
      budget = max_error_budget

    Area model depends on global counts (c0,c1,c2,c3):
      counts = helper_degrees_to_counts(degrees_list)
      area   = helper_hardware_area(*counts)

    Returns:
      degrees_best, mse0, mse_best, area_best

    If return_frontier=True, also returns:
      frontier_points: list of dicts {area, mse, counts, degrees, is_feasible}
        - all merged DP final states before any Pareto pruning
      budget: float
      all_feasible: list of dicts {degrees, area, mse, counts} - feasible subset (retained for optional analysis)

    Data structure hierarchy (when return_frontier=True):
      - frontier_points (dp_frontier_points): ALL merged DP final states
      - pareto_feasible_configs: ALL feasible DP final states (is_feasible=True)
      - degrees_best (min-area solution): single config with smallest area in feasible set
    """
    degree_max = int(degree_max)
    if degree_max > 3:
        raise ValueError(
            f"[AreaModel] Your current area equation only supports degrees up to 3 (c0..c3). "
            f"Got degree_max={degree_max}."
        )

    K = int(len(bp_idx) - 1)
    segs = helper_segments_from_bp_indices(bp_idx)  # [l, r)

    # -------------------------------------------------------------------------
    # Baseline MSE0 (all deg=degree_max)
    # -------------------------------------------------------------------------
    deg_all = [degree_max] * K
    _, _, _, mse0_t = helper_fit_all_segments(x, y, bp_idx, deg_all, lam=lam)
    mse0 = float(mse0_t.item())

    # Budget is directly max_error_budget (already in MSE units)
    N = int(x.numel())
    budget = float(max_error_budget)

    # -------------------------------------------------------------------------
    # Degree -> delta counts for DP
    # We do NOT track c0 in DP state because c0 = K is constant.
    # DP state is (c1,c2,c3)
    # -------------------------------------------------------------------------
    def degree_to_delta_counts(d: int):
        d = int(d)
        if d < 0 or d > 3:
            raise ValueError(f"[AreaModel] Degree must be in [0,3], got {d}")
        dc1 = 1 if d >= 1 else 0
        dc2 = 1 if d >= 2 else 0
        dc3 = 1 if d >= 3 else 0
        return (dc1, dc2, dc3)

    # -------------------------------------------------------------------------
    # Precompute per-seg MSE contribution for each degree option:
    # options[j] = list of (d, mse_d, (dc1,dc2,dc3))
    # -------------------------------------------------------------------------
    options = []
    for j, (l, r) in enumerate(segs):
        x_seg = x[l:r]
        y_seg = y[l:r]
        seg_len = int(r - l)
        if x_seg.numel() == 0:
            raise ValueError(f"Empty segment {j}: [{l},{r})")

        b_left = helper_boundary_x_from_index(x, l)
        b_right = helper_boundary_x_from_index(x, r)

        opts_j = []
        for d in range(degree_max + 1):
            _, seg_mse_d = helper_solve_segment_poly(x_seg, y_seg, d, b_left, b_right, lam=lam)
            mse_d = float(seg_mse_d.item()) * (float(seg_len) / float(N))
            opts_j.append((int(d), mse_d, degree_to_delta_counts(d)))
        options.append(opts_j)

    # -------------------------------------------------------------------------
    # DP: (c1,c2,c3) -> minimal MSE
    # and store back pointers for reconstruction
    # -------------------------------------------------------------------------
    dp = {(0, 0, 0): 0.0}
    back_layers = []  # list of dict: new_state -> (prev_state, chosen_degree)

    for seg_i in range(K):
        new_dp = {}
        new_back = {}
        for (c1, c2, c3), mse_so_far in dp.items():
            for d, mse_d, (dc1, dc2, dc3) in options[seg_i]:
                ns = (c1 + dc1, c2 + dc2, c3 + dc3)
                nmse = mse_so_far + mse_d

                # keep smallest MSE for each state
                if (ns not in new_dp) or (nmse < new_dp[ns]):
                    new_dp[ns] = nmse
                    new_back[ns] = ((c1, c2, c3), int(d))
        dp = new_dp
        back_layers.append(new_back)

    # -------------------------------------------------------------------------
    # Helper: reconstruct degrees from back pointers for a given final state
    # -------------------------------------------------------------------------
    def reconstruct_degrees_from_state(final_state):
        degrees_rec = [0] * K
        state = final_state
        for seg_i in reversed(range(K)):
            prev_state, chosen_d = back_layers[seg_i][state]
            degrees_rec[seg_i] = int(chosen_d)
            state = prev_state
        return degrees_rec

    def build_final_state_configs():
        raw_configs = []
        for (cc1, cc2, cc3), mm in dp.items():
            counts = (K, cc1, cc2, cc3)
            raw_configs.append({
                "area": float(helper_hardware_area(*counts)),
                "mse": float(mm),
                "counts": counts,
                "degrees": reconstruct_degrees_from_state((cc1, cc2, cc3)),
                "is_feasible": bool(mm <= budget),
            })
        raw_configs.sort(key=lambda t: (t.get("area", float("inf")), t.get("mse", float("inf"))))
        return raw_configs

    # -------------------------------------------------------------------------
    # Choose feasible solution with minimum Area (tie-break by MSE)
    # -------------------------------------------------------------------------
    feasible = []
    for (c1, c2, c3), total_mse in dp.items():
        if total_mse <= budget:
            counts = (K, c1, c2, c3)
            area = helper_hardware_area(*counts)
            feasible.append((float(area), float(total_mse), counts, (c1, c2, c3)))

    if not feasible:
        # No budget-feasible config exists. Fall back to the lowest-MSE member
        # of the full merged DP state set.
        counts_all = helper_degrees_to_counts(deg_all)
        area_all = helper_hardware_area(*counts_all)

        if return_frontier:
            raw_configs = build_final_state_configs()
            fallback_cfg = helper_pick_pareto_fallback_config(raw_configs)
            if fallback_cfg is None:
                # Defensive fallback; this should not happen because DP always
                # produces at least one state.
                return deg_all, mse0, mse0, float(area_all), raw_configs, budget, []

            return (
                [int(d) for d in fallback_cfg.get("degrees", deg_all)],
                mse0,
                float(fallback_cfg.get("mse", mse0)),
                float(fallback_cfg.get("area", area_all)),
                raw_configs,
                budget,
                [],
            )

        # Defensive non-frontier fallback path.
        return deg_all, mse0, mse0, float(area_all)

    feasible.sort(key=lambda t: (t[0], t[1]))
    area_best, mse_best, counts_best, state_best = feasible[0]
    _, c1b, c2b, c3b = counts_best

    # -------------------------------------------------------------------------
    # Reconstruct degrees_best from back pointers
    # -------------------------------------------------------------------------
    degrees_best = reconstruct_degrees_from_state((c1b, c2b, c3b))

    # -------------------------------------------------------------------------
    # Reconstruct degrees for ALL feasible configs (pass budget threshold)
    # -------------------------------------------------------------------------
    all_feasible = []
    for (area_f, mse_f, counts_f, state_f) in feasible:
        degrees_f = reconstruct_degrees_from_state(state_f)
        all_feasible.append({
            "area": float(area_f),
            "mse": float(mse_f),
            "counts": counts_f,
            "degrees": degrees_f,
            "is_feasible": True,  # Passes budget threshold
        })

    # -------------------------------------------------------------------------
    # Build the full merged DP state list for plotting / downstream search.
    # -------------------------------------------------------------------------
    if return_frontier:
        raw_configs = build_final_state_configs()
        return degrees_best, mse0, mse_best, float(area_best), raw_configs, budget, all_feasible

    return degrees_best, mse0, mse_best, float(area_best), all_feasible

# =============================================================================
# Phase 1: Coordinate descent on breakpoints (uniform experimental degree)
# =============================================================================
def phase1_optimize_breakpoints(
    x: torch.Tensor,
    y: torch.Tensor,
    num_segments: int,
    uniform_degree: int,
    num_outer_iters: int | None = None,
    min_seg_points: int = 8,
    lam: float = 0.0,
    verbose: bool = True,
    ncols: int = 140,
    bp_idx_init: torch.Tensor | None = None,
):
    """
    Phase 1: Optimize breakpoint locations using coordinate descent.

    Uses one shared degree for all segments during the Phase 1 objective.
    Upstream we currently hard-code this to degree_max.
    Output breakpoints serve as initialization for Phase 2.

    Runs until a full sweep produces no improving breakpoint move. An optional
    `num_outer_iters` safety cap can still be supplied by callers if needed.

    bp_idx boundaries in [0, N], with bp_idx[0]=0 and bp_idx[-1]=N.
    Segment j is [bp_idx[j], bp_idx[j+1]).

    IMPORTANT:
      MSE computations use midpoint boundaries via helper_boundary_x_from_index(),
      matching fit/inference/export.
    """
    N = x.numel()
    S = int(num_segments)
    B = S + 1
    # Preserve historical improvement threshold scale after objective normalization.
    improve_eps = 1e-12 / max(1, int(N))

    if bp_idx_init is not None and bp_idx_init.numel() != B:
        raise ValueError(f"bp_idx_init has {bp_idx_init.numel()} elems, expected {B}")

    if bp_idx_init is None:
        bp_idx = torch.linspace(0, N, B, device=x.device).round().long()
    else:
        bp_idx = bp_idx_init.clone().to(x.device).long()
        bp_idx[0] = 0
        bp_idx[-1] = N

    # enforce min segment size (project each internal breakpoint into feasible range)
    for j in range(1, B - 1):
        lo = int(bp_idx[j - 1].item()) + min_seg_points
        hi = int(bp_idx[j + 1].item()) - min_seg_points
        bp_idx[j] = torch.clamp(bp_idx[j], lo, hi)

    # Initial full fit with one shared degree across all segments.
    degrees_u = [uniform_degree] * S
    _, _, _, total_mse = helper_fit_all_segments(x, y, bp_idx, degrees_u, lam=lam)
    mse_hist = [float(total_mse.item())]

    # Cache per-segment MSE contribution for the same shared-degree objective.
    segs = helper_segments_from_bp_indices(bp_idx)
    mse_per_seg = []
    for (l, r) in segs:
        seg_len = int(r - l)
        b_left = helper_boundary_x_from_index(x, l)
        b_right = helper_boundary_x_from_index(x, r)
        _, seg_mse = helper_solve_segment_poly(x[l:r], y[l:r], uniform_degree, b_left, b_right, lam=lam)
        mse_per_seg.append(seg_mse * (float(seg_len) / float(N)))
    mse_per_seg = torch.stack(mse_per_seg, dim=0)

    pbar = None
    if verbose:
        pbar = tqdm(
            total=num_outer_iters,
            desc="  sweeps",
            dynamic_ncols=False,
            ncols=ncols,
            mininterval=0.2,
        )

    best_mse = mse_hist[-1]
    best_it = 0

    outer = 0
    while True:
        improved_any = False

        for k in range(1, B - 1):
            curr = int(bp_idx[k].item())
            left = int(bp_idx[k - 1].item())
            right = int(bp_idx[k + 1].item())

            lo = left + min_seg_points
            hi = right - min_seg_points
            if lo > hi:
                continue

            s_left = k - 1
            s_right = k
            unaffected = total_mse - mse_per_seg[s_left] - mse_per_seg[s_right]

            best_pos = curr
            best_total = total_mse

            for cand in range(lo, hi + 1):
                if cand == curr:
                    continue
                if (cand - left) < min_seg_points or (right - cand) < min_seg_points:
                    continue

                # segments: [left, cand) and [cand, right)
                bL = helper_boundary_x_from_index(x, left)
                bM = helper_boundary_x_from_index(x, cand)
                bR = helper_boundary_x_from_index(x, right)

                _, seg_mse1 = helper_solve_segment_poly(x[left:cand], y[left:cand], uniform_degree, bL, bM, lam=lam)
                _, seg_mse2 = helper_solve_segment_poly(x[cand:right], y[cand:right], uniform_degree, bM, bR, lam=lam)
                mse1 = seg_mse1 * (float(cand - left) / float(N))
                mse2 = seg_mse2 * (float(right - cand) / float(N))

                cand_total = unaffected + mse1 + mse2
                if cand_total < best_total - improve_eps:
                    best_total = cand_total
                    best_pos = cand

            if best_pos != curr:
                bp_idx[k] = best_pos

                bL = helper_boundary_x_from_index(x, left)
                bM = helper_boundary_x_from_index(x, best_pos)
                bR = helper_boundary_x_from_index(x, right)

                _, seg_mse1 = helper_solve_segment_poly(x[left:best_pos], y[left:best_pos], uniform_degree, bL, bM, lam=lam)
                _, seg_mse2 = helper_solve_segment_poly(x[best_pos:right], y[best_pos:right], uniform_degree, bM, bR, lam=lam)
                mse1 = seg_mse1 * (float(best_pos - left) / float(N))
                mse2 = seg_mse2 * (float(right - best_pos) / float(N))

                total_mse = unaffected + mse1 + mse2
                mse_per_seg[s_left] = mse1
                mse_per_seg[s_right] = mse2
                improved_any = True

        mse_val = float(total_mse.item())
        mse_hist.append(mse_val)

        if mse_val < best_mse:
            best_mse = mse_val
            best_it = outer + 1

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(
                f"mse={mse_val: .3e} best={best_mse: .3e} improved={int(improved_any)}",
                refresh=True
            )

        if not improved_any:
            break
        outer += 1
        if num_outer_iters is not None and outer >= num_outer_iters:
            break

    if pbar is not None:
        pbar.close()

    return bp_idx, mse_hist, best_it


# =============================================================================
# Phase 2: Coordinate descent refinement with mixed degrees
# =============================================================================
def phase2_refine_breakpoints(
    x: torch.Tensor,
    y: torch.Tensor,
    bp_idx_init: torch.Tensor,
    degrees: list,
    num_outer_iters: int | None = None,
    min_seg_points: int = 8,
    lam: float = 0.0,
    verbose: bool = True,
    ncols: int = 140,
):
    """
    Phase 2 (Part B): Refine breakpoints using the mixed degrees from DP.

    Unlike phase1_optimize_breakpoints which uses one shared Phase 1 degree,
    this version uses the provided degrees list for each segment.

    Args:
        x: input grid (N,)
        y: target values (N,)
        bp_idx_init: initial breakpoint indices, shape (S+1,)
        degrees: list of int, length S (one degree per segment)
        num_outer_iters: optional max sweeps safety cap; default runs until a
            full sweep produces no improving move
        min_seg_points: minimum points per segment
        lam: ridge regularization
        verbose: show progress bar
        ncols: tqdm column width

    Returns:
        bp_idx: optimized breakpoint indices
        mse_hist: list of MSE values per sweep
        best_it: index of best sweep in mse_hist
    """
    N = x.numel()
    S = len(degrees)
    B = S + 1
    # Preserve historical improvement threshold scale after objective normalization.
    improve_eps = 1e-12 / max(1, int(N))

    if bp_idx_init.numel() != B:
        raise ValueError(f"bp_idx_init has {bp_idx_init.numel()} elems, expected {B}")

    bp_idx = bp_idx_init.clone().to(x.device).long()
    bp_idx[0] = 0
    bp_idx[-1] = N

    # enforce min segment size
    for j in range(1, B - 1):
        lo = int(bp_idx[j - 1].item()) + min_seg_points
        hi = int(bp_idx[j + 1].item()) - min_seg_points
        bp_idx[j] = torch.clamp(bp_idx[j], lo, hi)

    # initial full fit with mixed degrees
    _, _, _, total_mse = helper_fit_all_segments(x, y, bp_idx, degrees, lam=lam)
    mse_hist = [float(total_mse.item())]

    # cache per-seg MSE contribution with mixed degrees
    segs = helper_segments_from_bp_indices(bp_idx)
    mse_per_seg = []
    for j, (l, r) in enumerate(segs):
        d = int(degrees[j])
        seg_len = int(r - l)
        b_left = helper_boundary_x_from_index(x, l)
        b_right = helper_boundary_x_from_index(x, r)
        _, seg_mse = helper_solve_segment_poly(x[l:r], y[l:r], d, b_left, b_right, lam=lam)
        mse_per_seg.append(seg_mse * (float(seg_len) / float(N)))
    mse_per_seg = torch.stack(mse_per_seg, dim=0)

    pbar = None
    if verbose:
        pbar = tqdm(
            total=num_outer_iters,
            desc="  mixed-cd",
            dynamic_ncols=False,
            ncols=ncols,
            mininterval=0.2,
        )

    best_mse = mse_hist[-1]
    best_it = 0

    outer = 0
    while True:
        improved_any = False

        for k in range(1, B - 1):
            curr = int(bp_idx[k].item())
            left = int(bp_idx[k - 1].item())
            right = int(bp_idx[k + 1].item())

            lo = left + min_seg_points
            hi = right - min_seg_points
            if lo > hi:
                continue

            s_left = k - 1
            s_right = k
            d_left = int(degrees[s_left])
            d_right = int(degrees[s_right])

            unaffected = total_mse - mse_per_seg[s_left] - mse_per_seg[s_right]

            best_pos = curr
            best_total = total_mse

            for cand in range(lo, hi + 1):
                if cand == curr:
                    continue
                if (cand - left) < min_seg_points or (right - cand) < min_seg_points:
                    continue

                # segments: [left, cand) and [cand, right)
                bL = helper_boundary_x_from_index(x, left)
                bM = helper_boundary_x_from_index(x, cand)
                bR = helper_boundary_x_from_index(x, right)

                # use the assigned degree for each segment
                _, seg_mse1 = helper_solve_segment_poly(x[left:cand], y[left:cand], d_left, bL, bM, lam=lam)
                _, seg_mse2 = helper_solve_segment_poly(x[cand:right], y[cand:right], d_right, bM, bR, lam=lam)
                mse1 = seg_mse1 * (float(cand - left) / float(N))
                mse2 = seg_mse2 * (float(right - cand) / float(N))

                cand_total = unaffected + mse1 + mse2
                if cand_total < best_total - improve_eps:
                    best_total = cand_total
                    best_pos = cand

            if best_pos != curr:
                bp_idx[k] = best_pos

                bL = helper_boundary_x_from_index(x, left)
                bM = helper_boundary_x_from_index(x, best_pos)
                bR = helper_boundary_x_from_index(x, right)

                _, seg_mse1 = helper_solve_segment_poly(x[left:best_pos], y[left:best_pos], d_left, bL, bM, lam=lam)
                _, seg_mse2 = helper_solve_segment_poly(x[best_pos:right], y[best_pos:right], d_right, bM, bR, lam=lam)
                mse1 = seg_mse1 * (float(best_pos - left) / float(N))
                mse2 = seg_mse2 * (float(right - best_pos) / float(N))

                total_mse = unaffected + mse1 + mse2
                mse_per_seg[s_left] = mse1
                mse_per_seg[s_right] = mse2
                improved_any = True

        mse_val = float(total_mse.item())
        mse_hist.append(mse_val)

        if mse_val < best_mse:
            best_mse = mse_val
            best_it = outer + 1

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(
                f"mse={mse_val: .3e} best={best_mse: .3e} improved={int(improved_any)}",
                refresh=True
            )

        if not improved_any:
            break
        outer += 1
        if num_outer_iters is not None and outer >= num_outer_iters:
            break

    if pbar is not None:
        pbar.close()

    return bp_idx, mse_hist, best_it


# =============================================================================
# I/O functions (save/load configs)
# =============================================================================
def helper_degree_type_counts(degrees, degree_max=None):
    """
    Convert per-segment degrees to counts of each degree type.

    Args:
      degrees: list of per-segment degrees
        degree_max: maximum degree to consider (if None, inferred from max(degrees))

    Returns:
        list of length (degree_max+1) where element i = number of segments with degree i
        e.g., [n0, n1, n2, n3, ...] for degree_max=3 or higher
    """
    degs = [int(d) for d in degrees]
    if degree_max is None:
        degree_max = max(degs) if degs else 0

    counts = [0] * (degree_max + 1)
    for d in degs:
        if 0 <= d <= degree_max:
            counts[d] += 1
    return counts


def helper_degrees_to_degree_counts(degrees, degree_max=None):
    """
    Backward-compatible alias for helper_degree_type_counts().
    """
    return helper_degree_type_counts(degrees, degree_max=degree_max)


def helper_degree_counts_to_c_counts(degree_counts):
    """
    Convert degree counts [n0, n1, n2, ...] to cumulative c counts (c0, c1, c2, ...) for area model.

    For helper_hardware_area, we need:
      c0 = total segments with constant term = sum of all
      c1 = segments with deg>=1 = n1 + n2 + n3 + ...
      c2 = segments with deg>=2 = n2 + n3 + ...
      etc.

    Returns:
        tuple (c0, c1, c2, c3) - note: area model currently only supports up to c3
    """
    # Sum from each degree to the end
    total = sum(degree_counts)
    c_counts = []
    running_sum = total
    for i, n in enumerate(degree_counts):
        c_counts.append(running_sum)
        running_sum -= n

    # Pad to at least 4 elements for the current area model
    while len(c_counts) < 4:
        c_counts.append(0)

    return tuple(c_counts[:4])  # (c0, c1, c2, c3)


def helper_reextract_pareto_frontier(dp_frontier_points, updated_config=None):
    """
    Re-extract Pareto frontier to remove dominated points after MSE changes.

    After 2nd CD refinement, MSE values change and some points may become dominated.
    A point is dominated if there exists another point with lower area AND lower MSE.
    On the Pareto frontier, lower area must imply higher MSE (monotonic trade-off).

    Args:
        dp_frontier_points: list of config dicts with 'area' and 'mse' keys
        updated_config: optional dict with 'degrees' and new 'mse' to update in-place
                        before re-extraction (e.g., min-area solution after 2nd CD)

    Returns:
        tuple: (num_removed, pareto_refined_list)
            - num_removed: number of dominated points removed
            - pareto_refined_list: new list of non-dominated points (sorted by area)
    """
    if not dp_frontier_points:
        return 0, []

    # Update the specific config if provided (e.g., min-area after 2nd CD)
    if updated_config is not None:
        updated_degrees = updated_config.get("degrees")
        updated_mse = updated_config.get("mse")
        if updated_degrees is not None and updated_mse is not None:
            for cfg in dp_frontier_points:
                if cfg.get("degrees") == updated_degrees:
                    cfg["mse"] = updated_mse
                    break

    # Sort by (area, mse) - lower area first, then lower mse as tiebreaker
    dp_frontier_points.sort(key=lambda t: (t.get("area", float("inf")), t.get("mse", float("inf"))))

    # Extract non-dominated points: as we go from low to high area,
    # each point must have strictly lower MSE than all previous points
    pareto_refined = []
    best_mse_so_far = float("inf")
    for p in dp_frontier_points:
        if p.get("mse", float("inf")) < best_mse_so_far:
            pareto_refined.append(p)
            best_mse_so_far = p["mse"]

    num_removed = len(dp_frontier_points) - len(pareto_refined)
    return num_removed, pareto_refined


def helper_find_pareto_config_by_degrees(pareto_configs, degrees):
    """Return the Pareto config whose degree pattern matches `degrees`, if present."""
    if not pareto_configs or degrees is None:
        return None

    target = tuple(int(d) for d in degrees)
    for cfg in pareto_configs:
        cfg_degrees = cfg.get("degrees")
        if cfg_degrees is None:
            continue
        if tuple(int(d) for d in cfg_degrees) == target:
            return cfg
    return None


def helper_pick_pareto_fallback_config(pareto_configs):
    """
    Pick a fallback directly from the Pareto frontier.

    When no configuration satisfies the error budget, use the frontier point with
    the lowest MSE (tie-break by lower area). This guarantees the fallback is an
    exported Pareto member instead of a dominated synthetic backup.
    """
    if not pareto_configs:
        return None

    candidates = [cfg for cfg in pareto_configs if cfg.get("degrees")]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda cfg: (
            float(cfg.get("mse", float("inf"))),
            float(cfg.get("area", float("inf"))),
        ),
    )


def io_save_config_csv(
    activation_name: str,
    num_breakpoints: int,
    degree_max: int,                 # naming only
    breakpoints_x,
    coeffs,
    out_dir=None,
    degrees=None,                    # list[int] length K
):
    """
    Saves a piecewise polynomial config to CSV.

    Filename format: {activation}_{K}seg_d{n0}_{n1}_{n2}_{n3}.csv
    where n0/n1/n2/n3 = count of constant/linear/quadratic/cubic segments.

    CSV schema:
      type,segment_idx,point_idx,value

    Rows:
      breakpoint,-1,i,breakpoints_x[i]
      degree,j,-1,degrees[j]
      coeff,j,k,coeff_value
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    b = np.asarray(breakpoints_x, dtype=np.float64).reshape(-1)
    if b.size != int(num_breakpoints):
        raise ValueError(f"breakpoints_x has {b.size} elems, expected num_breakpoints={num_breakpoints}")

    K = int(num_breakpoints) - 1

    if degrees is None:
        degrees = [int(len(c) - 1) for c in coeffs]
    else:
        degrees = [int(d) for d in degrees]

    if len(degrees) != K:
        raise ValueError(f"degrees length {len(degrees)} != K={K}")
    if len(coeffs) != K:
        raise ValueError(f"coeffs length {len(coeffs)} != K={K}")

    # Compute degree counts for filename (generalized for any degree_max)
    deg_counts = helper_degrees_to_degree_counts(degrees, degree_max)

    # New filename format: GELU_4seg_d1_1_1_1.csv (degree counts separated by underscore)
    deg_counts_str = '_'.join(str(c) for c in deg_counts)
    filename = out_dir / f"{activation_name}_{K}seg_d{deg_counts_str}.csv"

    with open(filename, "w") as f:
        f.write("type,segment_idx,point_idx,value\n")

        for i, bx in enumerate(b):
            f.write(f"breakpoint,-1,{i},{bx:.10f}\n")

        for j in range(K):
            dj = int(degrees[j])
            f.write(f"degree,{j},-1,{dj}\n")

            cj = coeffs[j]
            if isinstance(cj, torch.Tensor):
                cj = cj.detach().cpu().numpy()
            cj = np.asarray(cj, dtype=np.float64).reshape(-1)

            expected_len = dj + 1
            if cj.size != expected_len:
                raise ValueError(f"Segment {j}: coeff length {cj.size} != (degree+1)={expected_len}")

            for k in range(expected_len):
                f.write(f"coeff,{j},{k},{cj[k]:.10f}\n")

    return str(filename)


def io_save_inference_csv(res, out_csv):
    x = np.asarray(res["x_train"])
    y_fit = np.asarray(res["y_train_fit"])
    b = np.asarray(res["breakpoints_x_final"])

    x_t = torch.from_numpy(x).float().to(device)
    b_t = torch.from_numpy(b).float().to(device)
    K = b_t.numel() - 1
    seg_ids = torch.bucketize(x_t, b_t[1:-1], right=True).clamp(0, K - 1).cpu().numpy()

    x_local = np.zeros_like(x, dtype=np.float32)
    for j in range(K):
        m = (seg_ids == j)
        if not np.any(m):
            continue
        denom = max(1e-12, float(b[j + 1] - b[j]))
        x_local[m] = (2.0 * (x[m] - b[j]) / denom - 1.0).astype(np.float32)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("x,y_fitted,segment_id,x_local\n")
        for xv, yv, sid, xl in zip(x, y_fit, seg_ids, x_local):
            f.write(f"{xv:.10f},{yv:.10f},{int(sid)},{xl:.10f}\n")


def helper_attach_breakpoints_to_pareto_configs(
    pareto_configs,
    chosen_degrees,
    chosen_breakpoints_x,
    chosen_breakpoints_idx,
    chosen_mse,
    fallback_breakpoints_x,
    fallback_breakpoints_idx,
):
    """
    Attach breakpoint metadata to Pareto configs before optional per-config refinement.

    The chosen min-area config keeps the current final breakpoints/MSE. All other
    Pareto configs stay on the shared Phase 1 breakpoint baseline until they are
    optionally refined later.
    """
    if not pareto_configs:
        return

    chosen_key = tuple(int(d) for d in chosen_degrees) if chosen_degrees is not None else None
    chosen_bp_x = np.asarray(chosen_breakpoints_x).copy() if chosen_breakpoints_x is not None else None
    fallback_bp_x = np.asarray(fallback_breakpoints_x).copy() if fallback_breakpoints_x is not None else None

    for pareto_cfg in pareto_configs:
        cfg_degrees = pareto_cfg.get("degrees")
        if not cfg_degrees:
            continue

        cfg_key = tuple(int(d) for d in cfg_degrees)
        if chosen_key is not None and cfg_key == chosen_key:
            if chosen_bp_x is not None:
                pareto_cfg["breakpoints_x"] = chosen_bp_x.copy()
            if chosen_breakpoints_idx is not None:
                pareto_cfg["breakpoints_idx"] = (
                    chosen_breakpoints_idx.clone()
                    if torch.is_tensor(chosen_breakpoints_idx)
                    else np.asarray(chosen_breakpoints_idx).copy()
                )
            if chosen_mse is not None:
                pareto_cfg["mse"] = float(chosen_mse)
        else:
            if fallback_bp_x is not None:
                pareto_cfg["breakpoints_x"] = fallback_bp_x.copy()
            if fallback_breakpoints_idx is not None:
                pareto_cfg["breakpoints_idx"] = (
                    fallback_breakpoints_idx.clone()
                    if torch.is_tensor(fallback_breakpoints_idx)
                    else np.asarray(fallback_breakpoints_idx).copy()
                )


def helper_save_pareto_config_csvs(
    activation_name,
    degree,
    x_train,
    y_train,
    pareto_configs,
    pareto_config_dir,
    lam=0.0,
):
    """
    Save Pareto-optimal configs as exported CSV configs.
    Returns list of written filenames.
    """
    filenames = []
    if pareto_config_dir is None or not pareto_configs:
        return filenames

    for pareto_cfg in pareto_configs:
        pareto_degrees = pareto_cfg.get("degrees")
        if not pareto_degrees:
            continue

        bp_x = pareto_cfg.get("breakpoints_x")
        bp_idx = pareto_cfg.get("breakpoints_idx")
        if bp_x is None or bp_idx is None:
            continue

        pareto_coeffs, _, _, _ = helper_fit_all_segments(
            x_train, y_train, bp_idx, pareto_degrees, lam=lam
        )
        pareto_filename = io_save_config_csv(
            activation_name=activation_name,
            num_breakpoints=int(len(bp_x)),
            degree_max=int(degree),
            breakpoints_x=bp_x,
            coeffs=pareto_coeffs,
            out_dir=pareto_config_dir,
            degrees=pareto_degrees,
        )
        filenames.append(pareto_filename)

    return filenames

# =============================================================================
# Per-activation pipeline: Phase 1 + Phase 2 for a single activation function
# =============================================================================
def run_single_activation(
    activation_name,
    target_func,
    num_breakpoints=16,
    degree=3,                 # degree_max (NOTE: for area model, must be <= 3 if use_mixed_degrees=True)
    n_train_samples=2048,
    n_eval_samples=4096,
    num_outer_iters=None,
    min_seg_points=8,
    lam=0.0,
    verbose=True,
    out_dirs=None,
    use_mixed_degrees=True,
    max_error_budget=None,
    refine_plausible_configs=False,
):
    """
    Run Phase 1 + Phase 2 optimization for a single activation function.

    ==========================================================================
    PHASE 1: Breakpoint Optimization (calls phase1_optimize_breakpoints)
    ==========================================================================
      - Uses coordinate descent with a hard-coded uniform degree_max fit
      - Optimizes breakpoint locations to minimize MSE
      - Output: bp_opt (optimized breakpoint indices)

    ==========================================================================
    PHASE 2: Mixed-Degree Selection (DP)
    ==========================================================================
      - Selects per-segment degrees to minimize area under error budget
      - Uses Phase 1 breakpoints as the shared baseline before Stage 2
      - Output: degrees_mixed, full merged DP end-state set
      - Optional: if refine_plausible_configs=True, refine all merged DP
        end states with CD in-place until convergence

    Args:
        activation_name: Name of the activation function
        target_func: The activation function to fit
        num_breakpoints: Number of breakpoints (segments = breakpoints - 1)
        degree: Maximum polynomial degree (degree_max)
        n_train_samples: Number of training samples
        n_eval_samples: Number of evaluation samples
        num_outer_iters: Optional CD sweep safety cap; default runs until
            convergence
        min_seg_points: Minimum points per segment
        lam: Ridge regularization
        verbose: Print progress
        out_dirs: Output directory structure
        use_mixed_degrees: If True, run Phase 2; if False, use uniform degree
        max_error_budget: Maximum MSE budget (required for mixed degrees)
            refine_plausible_configs: If True, refine all merged DP end states
            with CD in-place until convergence

    Returns:
        dict with Phase 1+2 results including:
        - degrees_final: Per-segment degrees
        - coeffs_final: Polynomial coefficients
        - breakpoints_x_final: Breakpoint x-values
        - final_train_mse, final_eval_mse: MSE metrics
        - dp_frontier_points: All merged DP final states
        - pareto_feasible_configs: Feasible DP final states that pass budget
    """
    degree = int(degree)
    if use_mixed_degrees and degree > 3:
        raise ValueError(
            f"[AreaModel] use_mixed_degrees=True requires degree_max<=3 (your area model uses c0..c3). "
            f"Got degree_max={degree}."
        )
    t_activation_start = time.perf_counter()
    phase1_elapsed_sec = 0.0
    phase2a_elapsed_sec = 0.0
    phase2_refine_elapsed_sec = 0.0
    phase2_elapsed_sec = 0.0

    if verbose:
        log_file_only(f"\n{'='*70}")
        log_file_only(f"{activation_name} | segments={num_breakpoints-1} | degree_max={degree}")
        log_file_only(f"{'='*70}")

    num_segments = num_breakpoints - 1

    # =========================================================================
    # Setup: Training and evaluation grids
    # =========================================================================
    domain_min, domain_max = helper_activation_domain(activation_name)
    x_train = torch.linspace(domain_min, domain_max, n_train_samples, device=device, dtype=torch.float32)
    y_train = target_func(x_train)
    N = x_train.numel()

    bp_init = torch.linspace(0, N, num_breakpoints, device=device).round().long()
    bp_init[0] = 0
    bp_init[-1] = N

    phase1_uniform_degree = degree
    phase1_degrees = [phase1_uniform_degree] * num_segments
    degrees_uniform_degmax = [degree] * num_segments

    _, _, _, mse_init = helper_fit_all_segments(x_train, y_train, bp_init, phase1_degrees, lam=lam)
    initial_mse = float(mse_init.item())

    if verbose:
        log_file_only(f"Initial MSE: {initial_mse:.6e}")

    # =========================================================================
    # PHASE 1: Breakpoint optimization with a hard-coded uniform degree_max fit
    # =========================================================================
    if verbose:
        log_file_only(
            f"Phase 1: Breakpoint optimization (uniform degree={phase1_uniform_degree})..."
        )

    phase1_start_t = time.perf_counter()
    bp_opt, mse_hist, best_it = phase1_optimize_breakpoints(
        x_train, y_train,
        num_segments=num_segments,
        uniform_degree=phase1_uniform_degree,
        num_outer_iters=num_outer_iters,
        min_seg_points=min_seg_points,
        lam=lam,
        verbose=verbose,
        bp_idx_init=bp_init,
    )
    phase1_elapsed_sec = time.perf_counter() - phase1_start_t

    # Phase 1 output: breakpoint x-values
    b_x_final = torch.stack([helper_boundary_x_from_index(x_train, int(i.item())) for i in bp_opt]).detach()

    # =========================================================================
    # Setup: Evaluation grid
    # =========================================================================
    x_eval = torch.linspace(domain_min, domain_max, n_eval_samples, device=device, dtype=torch.float32)
    y_eval = target_func(x_eval)

    # =========================================================================
    # Uniform degree_max fit on the Phase 1 breakpoints (before Phase 2)
    # -------------------------
    degrees_final = degrees_uniform_degmax
    coeffs_final, y_train_fit, final_train_mse, final_eval_mse = helper_fit_and_eval_with_breakpoints(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        bp_idx=bp_opt,
        breakpoints_x=b_x_final,
        degrees=degrees_final,
        lam=lam,
    )

    # Phase tracking for visualization
    phase1_end_idx = len(mse_hist) - 1
    mse_after_dp = None

    # DP artifacts (populated by Phase 2)
    dp_frontier_points = None
    dp_budget_mse = None
    dp_mse0 = None
    dp_chosen_area = None
    dp_chosen_mse = None
    dp_chosen_degrees = None
    dp_chosen_counts = None
    total_parameters_final = int(sum((d + 1) for d in degrees_final))
    total_parameters_uniform = num_segments * (degree + 1)  # Baseline: all segments at degree_max
    total_area_final = None

    # Save Phase 1 breakpoints
    bp_x_phase1 = b_x_final.detach().cpu().numpy().copy()
    bp_idx_phase1 = bp_opt.clone()

    # Store all feasible configs for later saving
    all_feasible_configs = None
    phase2_start_t = time.perf_counter()

    # =========================================================================
    # PHASE 2: Mixed-degree selection on Phase 1 breakpoints
    # =========================================================================
    if use_mixed_degrees:
        # -----------------------------------------------------------------
        # Phase 2A: DP degree selection (minimize area under MSE budget)
        # -----------------------------------------------------------------
        if verbose:
            log_file_only(f"Phase 2A: DP degree selection...")

        phase2a_start_t = time.perf_counter()
        degrees_mixed, mse0, mse_mixed, area_mixed, frontier, budget, all_feasible_configs = phase2_select_degrees_dp(
            bp_idx=bp_opt,
            x=x_train,
            y=y_train,
            degree_max=degree,
            lam=lam,
            max_error_budget=max_error_budget,
            return_frontier=True,
        )
        phase2a_elapsed_sec = time.perf_counter() - phase2a_start_t

        dp_mse0 = float(mse0)
        dp_budget_mse = float(budget)
        dp_chosen_area = float(area_mixed)
        dp_chosen_mse = float(mse_mixed)
        dp_chosen_degrees = [int(d) for d in degrees_mixed]
        dp_chosen_counts = helper_degrees_to_counts(dp_chosen_degrees)
        # Phase 2A now returns all merged DP end states without Pareto pruning.
        dp_frontier_points = frontier

        # -----------------------------------------------------------------
        # Keep all candidate configs on the same Phase 1 breakpoint baseline
        # so Stage 2 compares them consistently.
        # -----------------------------------------------------------------
        if verbose:
            log_file_only("Phase 2B: deferred; all configs currently use Phase 1 breakpoints.")

        mse_after_dp = float(mse_mixed)

        # Fit final mixed model on Phase 1 breakpoints
        coeffs_m, y_train_fit_m, final_train_mse, final_eval_mse = helper_fit_and_eval_with_breakpoints(
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            bp_idx=bp_opt,
            breakpoints_x=b_x_final,
            degrees=degrees_mixed,
            lam=lam,
        )

        # Keep chosen MSE aligned with reported final metrics.
        dp_chosen_mse = float(final_train_mse)

        # adopt mixed model
        degrees_final = [int(d) for d in degrees_mixed]
        coeffs_final = coeffs_m
        y_train_fit = y_train_fit_m

        total_parameters_final = int(sum((d + 1) for d in degrees_final))
        total_area_final = float(area_mixed)

    else:
        # no mixed degrees: keep uniform degree_max
        dp_mse0 = float(final_train_mse)
        dp_budget_mse = float(final_train_mse)
        dp_chosen_area = None
        dp_chosen_mse = float(final_train_mse)
        dp_chosen_degrees = [degree] * num_segments
        dp_chosen_counts = helper_degrees_to_counts(dp_chosen_degrees)

        # frontier single point for plotting compatibility
        dp_frontier_points = [{"area": np.nan, "mse": float(final_train_mse), "counts": dp_chosen_counts}]

        total_parameters_final = int(num_segments * (degree + 1))
        total_area_final = None

    # -------------------------
    # Sanity: exported inference on x_train must match train MSE
    # -------------------------
    with torch.no_grad():
        y_train_infer, _, _ = helper_predict_from_coeffs(
            x_train, b_x_final, coeffs_final, degrees_final
        )
        mse_exported_on_train = float(torch.mean((y_train_infer - y_train) ** 2).item())

    if verbose:
        log_file_only(f"[SANITY] {activation_name} MSE exported-on-train: {mse_exported_on_train:.6e}")
        log_file_only(f"[SANITY] {activation_name} MSE reported-train:    {final_train_mse:.6e}")
        log_file_only(f"[SANITY] abs diff: {abs(mse_exported_on_train - final_train_mse):.6e}")

    improvement = (initial_mse - final_train_mse) / max(1e-12, initial_mse) * 100.0

    if verbose:
        log_file_only(f"\n{'='*70}")
        log_file_only("OVERFIT RESULTS (HARD)")
        log_file_only(f"{'='*70}")
        log_file_only(f"Initial Train MSE: {initial_mse:.6e}")
        log_file_only(f"Final   Train MSE: {final_train_mse:.6e}")
        log_file_only(f"Dense   Eval  MSE: {final_eval_mse:.6e}")
        log_file_only(f"Improvement:       {improvement:6.2f}%")
        log_file_only(f"Final breakpoints={int(b_x_final.numel())} | segments={int(b_x_final.numel()-1)}")
        if use_mixed_degrees:
            log_file_only(f"Mixed degrees enabled | total_params={total_parameters_final} | total_area={total_area_final:.3f}")
            log_file_only(f"Counts (c0,c1,c2,c3)={dp_chosen_counts}")
        if lam != 0.0:
            log_file_only(f"Ridge lam={lam}")

    # -------------------------
    # Attach breakpoint metadata to the current Phase 2 candidate configs.
    # When refine_plausible_configs=False these are already Pareto-optimal.
    # When refine_plausible_configs=True these are the merged DP end states and
    # will be frontier-pruned only after per-config refinement below.
    # -------------------------
    budget_config_dir = out_dirs.get("min_area_config") if out_dirs is not None else None
    config_filename = None
    pareto_config_filenames = []
    pareto_all_config_filenames = []
    helper_attach_breakpoints_to_pareto_configs(
        pareto_configs=dp_frontier_points,
        chosen_degrees=degrees_final,
        chosen_breakpoints_x=b_x_final.detach().cpu().numpy(),
        chosen_breakpoints_idx=bp_opt,
        chosen_mse=final_train_mse,
        fallback_breakpoints_x=bp_x_phase1,
        fallback_breakpoints_idx=bp_idx_phase1,
    )

    # Filter current candidate set to configs that pass the budget.
    pareto_feasible = [p for p in dp_frontier_points if p.get("is_feasible", False)] if dp_frontier_points else []

    # Run 2nd CD refinement for ALL merged DP end states if requested.
    # This block runs even when pareto_feasible is empty (refines non-feasible points too).
    if refine_plausible_configs and dp_frontier_points:
        phase2_refine_start_t = time.perf_counter()
        # Refine every merged DP final state in-place.
        feasible_before_refine = len([p for p in dp_frontier_points if p.get("is_feasible", False)])
        log_file_only(
            f"[{activation_name}] 2nd CD refinement: {len(dp_frontier_points)} DP end-state configs, "
            f"{feasible_before_refine} feasible before refinement"
        )

        for idx, pareto_cfg in enumerate(dp_frontier_points):
            pareto_degrees = pareto_cfg.get("degrees")
            if not pareto_degrees:
                continue

            # Store pre-CD MSE for this config (so we can track before/after for any config)
            pareto_cfg["mse_before_cd"] = pareto_cfg.get("mse", 0.0)

            # Run 2nd CD for this config's degree assignment (start from Phase 1 breakpoints)
            bp_opt_pareto, _, _ = phase2_refine_breakpoints(
                x=x_train,
                y=y_train,
                bp_idx_init=bp_idx_phase1,  # Use Phase 1 breakpoints (not Phase 2!)
                degrees=pareto_degrees,
                num_outer_iters=None,
                min_seg_points=min_seg_points,
                lam=lam,
                verbose=False,  # Suppress inner progress bars
            )

            # Get refined breakpoint x-values
            b_x_pareto = torch.stack([helper_boundary_x_from_index(x_train, int(i)) for i in bp_opt_pareto])

            # Fit coefficients with refined breakpoints and get updated MSE
            pareto_coeffs, _, _, pareto_mse = helper_fit_all_segments(
                x_train, y_train, bp_opt_pareto, pareto_degrees, lam=lam
            )

            # Update config with refined breakpoints and MSE in-place.
            pareto_cfg["breakpoints_x"] = b_x_pareto.detach().cpu().numpy()
            pareto_cfg["breakpoints_idx"] = bp_opt_pareto.clone()
            pareto_cfg["mse"] = float(pareto_mse.item())

        # Re-check feasibility after refinement: MSE may have decreased
        # Some previously non-feasible points could now be feasible
        newly_feasible_count = 0
        for pareto_cfg in dp_frontier_points:
            old_feasible = pareto_cfg.get("is_feasible", False)
            new_feasible = pareto_cfg["mse"] <= dp_budget_mse if dp_budget_mse is not None else False
            pareto_cfg["is_feasible"] = new_feasible
            if new_feasible and not old_feasible:
                newly_feasible_count += 1

        # Recompute the feasible subset after refinement. Stage 2 now searches
        # over the full merged DP state set rather than a Pareto-pruned subset.
        pareto_feasible = [p for p in dp_frontier_points if p.get("is_feasible", False)]

        # Update the chosen Phase 2 solution from the refined DP state set.
        # If feasible points exist, keep the minimum-area feasible point.
        # Otherwise, fall back to the lowest-MSE DP final state.
        if pareto_feasible:
            min_area_cfg = min(pareto_feasible, key=lambda p: p.get("area", float("inf")))
            degrees_final = min_area_cfg["degrees"]
            dp_chosen_degrees = min_area_cfg["degrees"]
            dp_chosen_counts = min_area_cfg.get("counts", helper_degrees_to_degree_counts(degrees_final, degree))
            dp_chosen_area = min_area_cfg["area"]
            dp_chosen_mse = min_area_cfg["mse"]
            final_train_mse = min_area_cfg["mse"]
            total_area_final = min_area_cfg["area"]

            # Update mse_after_dp to match the NEW min-area config's pre-CD MSE
            # This ensures visualization compares same config before/after CD
            if "mse_before_cd" in min_area_cfg:
                mse_after_dp = min_area_cfg["mse_before_cd"]

            # Get breakpoints and coefficients for the new min-area solution
            if "breakpoints_idx" in min_area_cfg:
                bp_opt = min_area_cfg["breakpoints_idx"].clone() if torch.is_tensor(min_area_cfg["breakpoints_idx"]) else torch.tensor(min_area_cfg["breakpoints_idx"], dtype=torch.long)
                b_x_final = torch.stack([helper_boundary_x_from_index(x_train, int(i)) for i in bp_opt])
                coeffs_final, _, _, _ = helper_fit_all_segments(x_train, y_train, bp_opt, degrees_final, lam=lam)
                y_train_fit, _, _ = helper_predict_from_coeffs(x_train, b_x_final, coeffs_final, degrees_final)
                final_eval_mse = float(((y_train_fit - y_train) ** 2).mean().item())
        elif dp_frontier_points:
            fallback_cfg = helper_pick_pareto_fallback_config(dp_frontier_points)
            if fallback_cfg is not None:
                degrees_final = [int(d) for d in fallback_cfg.get("degrees", degrees_final)]
                dp_chosen_degrees = degrees_final
                dp_chosen_counts = fallback_cfg.get("counts", helper_degrees_to_counts(degrees_final))
                dp_chosen_area = float(fallback_cfg.get("area", dp_chosen_area if dp_chosen_area is not None else np.nan))
                dp_chosen_mse = float(fallback_cfg.get("mse", final_train_mse))
                final_train_mse = float(fallback_cfg.get("mse", final_train_mse))
                total_area_final = float(fallback_cfg.get("area", total_area_final if total_area_final is not None else np.nan))

                if "mse_before_cd" in fallback_cfg:
                    mse_after_dp = fallback_cfg["mse_before_cd"]

                if "breakpoints_idx" in fallback_cfg:
                    bp_opt = (
                        fallback_cfg["breakpoints_idx"].clone()
                        if torch.is_tensor(fallback_cfg["breakpoints_idx"])
                        else torch.tensor(fallback_cfg["breakpoints_idx"], dtype=torch.long)
                    )
                    b_x_final = torch.stack([helper_boundary_x_from_index(x_train, int(i)) for i in bp_opt])
                    coeffs_final, y_train_fit, final_train_mse, final_eval_mse = helper_fit_and_eval_with_breakpoints(
                        x_train=x_train,
                        y_train=y_train,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        bp_idx=bp_opt,
                        breakpoints_x=b_x_final,
                        degrees=degrees_final,
                        lam=lam,
                    )

        log_file_only(f"[{activation_name}] 2nd CD complete: +{newly_feasible_count} feasible")
        phase2_refine_elapsed_sec = time.perf_counter() - phase2_refine_start_t

    phase2_elapsed_sec = time.perf_counter() - phase2_start_t
    activation_elapsed_sec = time.perf_counter() - t_activation_start

    if verbose:
        log_file_only(
            f"[Timing][{activation_name}] "
            f"Phase 1={phase1_elapsed_sec:.3f}s | "
            f"Phase 2={phase2_elapsed_sec:.3f}s (DP={phase2a_elapsed_sec:.3f}s, refine={phase2_refine_elapsed_sec:.3f}s) | "
            f"Activation total={activation_elapsed_sec:.3f}s"
        )

    # -------------------------
    # Save final min-area config AFTER any refinement/re-selection.
    # -------------------------
    if budget_config_dir is not None:
        config_filename = io_save_config_csv(
            activation_name=activation_name,
            num_breakpoints=int(b_x_final.numel()),
            degree_max=int(degree),
            breakpoints_x=b_x_final.detach().cpu().numpy(),
            coeffs=coeffs_final,
            out_dir=budget_config_dir,
            degrees=degrees_final,
        )

    # -------------------------
    # Save the full merged DP state set and the feasible subset after any
    # refinement/re-selection.
    # -------------------------
    if out_dirs is not None and dp_frontier_points:
        pareto_all_config_dir = out_dirs.get("pareto_all_configs")
        pareto_all_config_filenames = helper_save_pareto_config_csvs(
            activation_name=activation_name,
            degree=degree,
            x_train=x_train,
            y_train=y_train,
            pareto_configs=dp_frontier_points,
            pareto_config_dir=pareto_all_config_dir,
            lam=lam,
        )

    if out_dirs is not None and pareto_feasible:
        pareto_config_dir = out_dirs.get("pareto_configs")
        pareto_config_filenames = helper_save_pareto_config_csvs(
            activation_name=activation_name,
            degree=degree,
            x_train=x_train,
            y_train=y_train,
            pareto_configs=pareto_feasible,
            pareto_config_dir=pareto_config_dir,
            lam=lam,
        )

    return {
        "activation_name": activation_name,
        "num_breakpoints": int(b_x_final.numel()),
        "num_segments": int(b_x_final.numel() - 1),
        "degree_max": int(degree),

        "initial_mse": float(initial_mse),
        "final_train_mse": float(final_train_mse),
        "final_eval_mse": float(final_eval_mse),
        "improvement_pct": float(improvement),

        "mse_history": mse_hist,
        "best_it": int(best_it),

        # Phase tracking for visualization
        "phase1_end_idx": int(phase1_end_idx),  # last index of 1st CD (uniform degree)
        "mse_after_dp": float(mse_after_dp) if mse_after_dp is not None else None,

        "x_train": x_train.detach().cpu().numpy(),
        "y_train": y_train.detach().cpu().numpy(),
        "y_train_fit": y_train_fit.detach().cpu().numpy(),

        "breakpoints_x_final": b_x_final.detach().cpu().numpy(),
        "breakpoints_x_phase1": bp_x_phase1,  # Phase 1 breakpoints X values (before 2nd CD)
        "breakpoints_idx_phase1": bp_idx_phase1.detach().cpu().numpy(),  # Phase 1 breakpoint indices (exact)
        "coeffs_final": [c.detach().cpu().numpy() for c in coeffs_final],
        "degrees_final": degrees_final,

        # both metrics
        "total_parameters": int(total_parameters_final),
        "total_parameters_uniform_degmax": int(total_parameters_uniform),
        "total_area": float(total_area_final) if total_area_final is not None else None,

        "max_error_budget": float(max_error_budget) if max_error_budget is not None else None,
        "sanity_mse_exported_on_train": float(mse_exported_on_train),

        # DP end-state data (including both feasible and non-feasible states)
        "dp_frontier_points": dp_frontier_points,   # list of {area, mse, counts, degrees, is_feasible}
        "dp_budget_mse": float(dp_budget_mse) if dp_budget_mse is not None else None,
        "dp_mse0": float(dp_mse0) if dp_mse0 is not None else None,
        "dp_chosen_area": float(dp_chosen_area) if dp_chosen_area is not None else None,
        "dp_chosen_mse": float(dp_chosen_mse) if dp_chosen_mse is not None else None,
        "dp_chosen_degrees": dp_chosen_degrees,
        "dp_chosen_counts": dp_chosen_counts,

        "config_filename": config_filename,

        # Full merged DP state export
        "pareto_all_config_filenames": pareto_all_config_filenames,

        # Feasible merged DP states (used by Phase 3)
        "pareto_config_filenames": pareto_config_filenames,
        "pareto_feasible_configs": pareto_feasible,  # feasible DP states under budget
        "timing_seconds": {
            "phase1": float(phase1_elapsed_sec),
            "phase2_total": float(phase2_elapsed_sec),
            "phase2_dp": float(phase2a_elapsed_sec),
            "phase2_refine": float(phase2_refine_elapsed_sec),
            "activation_total": float(activation_elapsed_sec),
        },
    }

# =============================================================================
# Visualization: 3 columns per activation (AREA-aware DP plot) — CLEAN TEXTBOX
# =============================================================================
def viz_all_activations(all_results, num_breakpoints, out_dir=None, upper_bound_area=None, upper_bound_degree_counts=None, max_error_budget=None, refine_plausible_configs=False):
    """
    Visualize all activation results.

    Args:
        all_results: list of result dicts from run_single_activation
        num_breakpoints: number of breakpoints
        out_dir: output directory
        upper_bound_area: if provided, draws a vertical red line at this area value
                          representing the combined upper bound across all activations
        upper_bound_degree_counts: list of degree counts [n0, n1, n2, ...] for label
        max_error_budget: if provided (user-specified budget), changes budget line to green
        refine_plausible_configs: if True, feasible points (below budget) are colored differently
                                  to indicate they have been fully refined (phase 1 + phase 2)
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir) if out_dir is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_activations = len(all_results)

    # 3 columns:
    #   Col 0: MSE history
    #   Col 1: Final fit (train grid)
    #   Col 2: Degree-DP Pareto frontier (Area vs MSE) + chosen point + summary textbox
    fig, axes = plt.subplots(num_activations, 3, figsize=(24, 6 * num_activations))
    if num_activations == 1:
        axes = axes.reshape(1, -1)

    # -------------------------
    # global degree->color mapping for shading (col 1)
    # -------------------------
    all_degrees = set()
    for res in all_results:
        b_final = np.asarray(res["breakpoints_x_final"])
        K = len(b_final) - 1
        deg_default = int(res["degree_max"])
        degs = res.get("degrees_final", [deg_default] * K)
        for d in degs:
            all_degrees.add(int(d))

    degrees_sorted = sorted(all_degrees) if all_degrees else [0]
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(10, len(degrees_sorted)))
    deg2color = {d: cmap(i % cmap.N) for i, d in enumerate(degrees_sorted)}
    degree_patches = [
        Patch(facecolor=deg2color[d], edgecolor="none", alpha=0.18, label=f"deg={d}")
        for d in degrees_sorted
    ]

    def draw_fit_panel(ax, *, x_vals, y_target, y_fit_vals, breakpoints, degrees, title, fit_label, fit_color, textbox_lines):
        """
        Draw a single fit panel with degree shading, breakpoint markers, and textbox.
        """
        ax.plot(x_vals, y_target, "k-", linewidth=2, alpha=0.7, label="Target")
        ax.plot(x_vals, y_fit_vals, fit_color, linewidth=2, alpha=0.85, label=fit_label)

        num_segments_local = max(0, len(breakpoints) - 1)
        for j in range(num_segments_local):
            dj = int(degrees[j]) if j < len(degrees) else 0
            ax.axvspan(
                breakpoints[j], breakpoints[j + 1],
                color=deg2color.get(dj, (0, 0, 0, 1)),
                alpha=0.18,
                linewidth=0
            )

        for bx in breakpoints:
            ax.axvline(bx, linewidth=0.8, alpha=0.25)

        y_fit_bp = np.interp(breakpoints, x_vals, y_fit_vals)
        ax.scatter(breakpoints, y_fit_bp, color='red', s=30, zorder=5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02, 0.98,
            "\n".join(textbox_lines),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

        handles, _labels = ax.get_legend_handles_labels()
        ax.legend(handles=degree_patches + handles, fontsize=7, loc="best", ncol=2)

    # -------------------------
    # helpers for textbox
    # -------------------------
    def degree_counts_0to3(degs):
        """
        Return counts [c0,c1,c2,c3] where ci = #segments with degree==i.
        """
        cnt = helper_degree_type_counts(degs, degree_max=3)
        if len(cnt) < 4:
            cnt = cnt + [0] * (4 - len(cnt))
        return cnt[:4]  # [deg0,deg1,deg2,deg3]

    def safe_area_from_degrees(degs, degree_max):
        """
        Compute area via counts(c0,c1,c2,c3) only if degree_max<=3.
        Otherwise return NaN.
        """
        degs = [int(d) for d in degs]
        if int(degree_max) > 3:
            return float("nan")
        counts = helper_degrees_to_counts(degs)  # (c0,c1,c2,c3)
        return float(helper_hardware_area(*counts))

    for idx, res in enumerate(all_results):
        name = res["activation_name"]
        x = res["x_train"]
        y_true = res["y_train"]
        y_fit = res["y_train_fit"]
        mse_hist = res["mse_history"]
        best_it = int(res.get("best_it", -1))
        b_final = np.asarray(res["breakpoints_x_final"])
        degree_max = int(res["degree_max"])

        N_train = int(len(x)) if hasattr(x, "__len__") else 1
        K = len(b_final) - 1

        degrees_used = res.get("degrees_final", [degree_max] * K)
        degrees_used = [int(d) for d in degrees_used]

        # Phase tracking data
        phase1_end_idx = res.get("phase1_end_idx", len(mse_hist) - 1)
        mse_after_dp = res.get("mse_after_dp", None)

        # ============================================================
        # Col 0: MSE history with phase visualization
        # ============================================================
        ax0 = axes[idx, 0]

        # Build combined MSE history for plotting
        # Phase 1: uniform-degree CD up to best_it (exclude trailing no-improvement point)
        # Transition: DP (mse_after_dp)

        # Trim Phase 1 to only include up to best_it (the last improvement point)
        # DP is performed on this point, so the trailing "no improvement" point is not relevant
        phase1_end = best_it + 1 if best_it >= 0 else len(mse_hist)
        combined_mse = list(mse_hist[:phase1_end])  # Phase 1 (up to best_it inclusive)

        phase1_last_idx = len(combined_mse) - 1  # last index of phase 1 in combined plot
        phase2_start_idx = None
        final_mse_idx = None

        if mse_after_dp is not None:
            # Add DP transition point
            combined_mse.append(mse_after_dp)
            phase2_start_idx = len(combined_mse) - 1
            final_mse_idx = len(combined_mse) - 1

        # Plot combined MSE history
        x_indices = list(range(len(combined_mse)))
        ax0.plot(x_indices, combined_mse, linewidth=2, color="C0")
        ax0.set_yscale("log")
        ax0.set_xlabel("Iteration")
        ax0.set_ylabel("Training MSE (HARD objective)")
        ax0.set_title(f"{name}: MSE History (CD → DP)")
        ax0.grid(True, alpha=0.3)

        # Shade Phase 1 (uniform-degree CD) with light blue - from first to last point of phase 1
        ax0.axvspan(x_indices[0], phase1_last_idx, alpha=0.15, color="blue", label="Phase 1: CD (uniform deg)")

        if phase2_start_idx is not None:
            # Shade Phase 2 with light green - from DP point to final
            ax0.axvspan(phase2_start_idx, x_indices[-1], alpha=0.15, color="green", label="Phase 2: DP (mixed deg)")

            # Mark DP transition point with diamond
            ax0.plot(phase2_start_idx, mse_after_dp, "mD", markersize=8,
                     label=f"DP transition", zorder=5)

            # Draw vertical line at phase boundary
            ax0.axvline((phase1_last_idx + phase2_start_idx) / 2, linestyle="--", linewidth=1.5,
                        color="gray", alpha=0.7)

        # Mark best sweep in Phase 1 (uniform degree) - this is the last point of phase 1
        ax0.plot(phase1_last_idx, combined_mse[phase1_last_idx], "ro", markersize=6,
                 label=f"Best CD (uniform)")

        # Mark final solution with star
        if final_mse_idx is not None and final_mse_idx < len(combined_mse):
            ax0.plot(final_mse_idx, combined_mse[final_mse_idx], "g*", markersize=12,
                     label="Final (mixed deg)", zorder=6)

        ax0.legend(fontsize=7, loc="upper right")

        # ============================================================
        # Col 1: Final fit
        # ============================================================
        ax1 = axes[idx, 1]
        ax1.plot(x, y_true, "k-", linewidth=2, alpha=0.7, label="Target")
        ax1.plot(x, y_fit, "g--", linewidth=2, alpha=0.85, label="Final (hard LS)")

        for j in range(K):
            dj = degrees_used[j]
            ax1.axvspan(
                b_final[j], b_final[j + 1],
                color=deg2color.get(dj, (0, 0, 0, 1)),
                alpha=0.18,
                linewidth=0
            )

        for bx in b_final:
            ax1.axvline(bx, linewidth=0.8, alpha=0.25)

        # Breakpoint red dots (exported model)
        bx_t = torch.from_numpy(b_final).float().to(device)
        b_t = torch.from_numpy(b_final).float().to(device)
        coeffs_t = [torch.from_numpy(np.asarray(c)).float().to(device) for c in res["coeffs_final"]]
        with torch.no_grad():
            y_bx, _, _ = helper_predict_from_coeffs(bx_t, b_t, coeffs_t, degrees_used)
        ax1.plot(b_final, y_bx.detach().cpu().numpy(), "ro", markersize=4, label="Breakpoints")

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(f"Train Fit (MSE={res['final_train_mse']:.3e})")
        ax1.grid(True, alpha=0.3)

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=degree_patches + handles, fontsize=8, loc="best", ncol=2)

        # ============================================================
        # Col 2: DP Pareto (Area vs MSE) + chosen point + CLEAN TEXTBOX
        # ============================================================
        ax2 = axes[idx, 2]

        # Use Pareto frontier for the main curve (guarantees monotonicity)
        # Update MSE values from pareto_feasible_configs if available (may be refined after 2nd CD)
        dp_frontier = res.get("dp_frontier_points", [])
        pareto_feasible = res.get("pareto_feasible_configs", [])

        # Build lookup from degrees -> config in pareto_feasible (has potentially updated MSE)
        pareto_lookup = {tuple(cfg.get("degrees", [])): cfg for cfg in pareto_feasible if cfg.get("degrees")}

        # Update frontier points with refined MSE values where available
        frontier = []
        for p in dp_frontier:
            p_degrees = tuple(p.get("degrees", []))
            if p_degrees and p_degrees in pareto_lookup:
                # Use the MSE from pareto_feasible_configs (may be refined)
                updated_p = p.copy()
                updated_p["mse"] = pareto_lookup[p_degrees]["mse"]
                frontier.append(updated_p)
            else:
                frontier.append(p)

        if not frontier:
            frontier = dp_frontier if dp_frontier else pareto_feasible

        chosen_area = res.get("dp_chosen_area", None)
        chosen_mse = res.get("dp_chosen_mse", None)
        mse0 = res.get("dp_mse0", None)                 # baseline MSE0 (all-degmax)
        budget_mse = res.get("dp_budget_mse", None)

        # ---- plot frontier
        if frontier is None or len(frontier) == 0:
            ax2.set_title(f"{name}: Degree DP (no frontier data)")
        else:
            frontier_sorted = sorted(frontier, key=lambda t: float(t["area"]) if t["area"] is not None else 0.0)

            # Collect all points with feasibility info
            xs_feasible, ys_feasible = [], []
            xs_nonfeasible, ys_nonfeasible = [], []
            xs_all, ys_all = [], []

            for t in frontier_sorted:
                a = t.get("area", None)
                mm = t.get("mse", None)
                if a is None or mm is None:
                    continue
                a = float(a)
                if not np.isfinite(a):
                    continue
                mse_val = float(mm)
                xs_all.append(a)
                ys_all.append(mse_val)

                # Use is_feasible flag if available, otherwise fall back to budget comparison
                is_feas = t.get("is_feasible", None)
                if is_feas is None and budget_mse is not None:
                    is_feas = mse_val <= budget_mse

                if is_feas:
                    xs_feasible.append(a)
                    ys_feasible.append(mse_val)
                else:
                    xs_nonfeasible.append(a)
                    ys_nonfeasible.append(mse_val)

            if len(xs_all) > 0:
                # Always show feasible vs non-feasible split
                # Plot non-feasible points (above budget) in lighter color
                if len(xs_nonfeasible) > 0:
                    ax2.plot(xs_nonfeasible, ys_nonfeasible, marker="o", linestyle="none",
                             markersize=5, alpha=0.5, color="C0", label="Above budget (non-feasible)")

                # Plot feasible points (below budget) in brighter color
                if len(xs_feasible) > 0:
                    ax2.plot(xs_feasible, ys_feasible, marker="s", linestyle="none",
                             markersize=6, alpha=0.9, color="C1", label="Below budget (feasible)")

                # Connect all points with a line for Pareto frontier visualization
                ax2.plot(xs_all, ys_all, linestyle="-", linewidth=1.5, alpha=0.4, color="gray")

            # budget line - green for user-specified, blue (default) for inferred
            user_specified = res.get("user_specified_budget", False)
            if budget_mse is not None:
                if user_specified:
                    ax2.axhline(
                        budget_mse, color="green",
                        linestyle="--", linewidth=1.6, alpha=0.75,
                        label=f"Budget={budget_mse:.2e} (user-specified)"
                    )
                else:
                    ax2.axhline(
                        budget_mse,
                        linestyle="--", linewidth=1.6, alpha=0.75,
                        label=f"Budget={budget_mse:.2e}"
                    )

            # chosen point
            if chosen_area is not None and chosen_mse is not None:
                chosen_area_f = float(chosen_area)
                if np.isfinite(chosen_area_f):
                    ratio = None
                    if mse0 is not None and mse0 > 0:
                        ratio = chosen_mse / mse0

                    chosen_label = f"Chosen (MSE={chosen_mse:.2e})"
                    if ratio is not None:
                        chosen_label = f"Chosen (MSE={chosen_mse:.2e}, {ratio:.2f}×MSE0)"

                    ax2.scatter(
                        [chosen_area_f], [chosen_mse],
                        marker="*", s=280,
                        edgecolors="black", linewidths=0.8,
                        zorder=10,
                        label=chosen_label
                    )

        ax2.set_title(f"{name}: Degree DP Pareto (Area vs MSE)")
        ax2.set_yscale("log")
        ax2.set_xlabel("Hardware Area Cost (μm²)")
        ax2.set_ylabel("Total MSE (log scale)")
        ax2.grid(True, which="both", linestyle="--", alpha=0.45)

        # ---- build CLEAN textbox
        # mixed degrees list (DP chosen) if present; otherwise current final degrees
        mixed_degs = res.get("dp_chosen_degrees", None)
        if mixed_degs is None or len(mixed_degs) != K:
            mixed_degs = degrees_used
        mixed_degs = [int(d) for d in mixed_degs]

        # degree==0/1/2/3 counts
        d0, d1, d2, d3 = degree_counts_0to3(mixed_degs)

        # mixed metrics
        mixed_params = int(sum((d + 1) for d in mixed_degs))
        mixed_area = res.get("dp_chosen_area", None)
        if mixed_area is None or not np.isfinite(float(mixed_area)):
            mixed_area = safe_area_from_degrees(mixed_degs, degree_max)
        mixed_area = float(mixed_area)

        # mixed MSE from DP chosen MSE if present (preferred) else final train MSE
        if chosen_mse is None:
            mixed_mse = float(res.get("final_train_mse", np.nan))
        else:
            mixed_mse = float(chosen_mse)

        # baseline all-max-degree metrics (THIS is MSE0)
        allmax_degs = [degree_max] * K
        allmax_params = int(K * (degree_max + 1))
        allmax_area = safe_area_from_degrees(allmax_degs, degree_max)
        allmax_mse = float(mse0) if mse0 is not None else float("nan")  # MSE0

        # ratios
        ratio_to_mse0 = None
        if np.isfinite(allmax_mse) and allmax_mse > 0 and np.isfinite(mixed_mse):
            ratio_to_mse0 = mixed_mse / allmax_mse

        # budget MSE
        budget_mse_val = float(budget_mse) if budget_mse is not None else float("nan")

        textbox_lines = []
        textbox_lines.append(f"K={K}")
        # degree segment counts
        textbox_lines.append(f"deg0/1/2/3 = {d0}/{d1}/{d2}/{d3}")

        # degree parameter counts (how many coefficients are used by segments of each degree)
        p0 = d0 * 1
        p1 = d1 * 2
        p2 = d2 * 3
        p3 = d3 * 4
        textbox_lines.append(f"degCount(d0/d1/d2/d3)= {p0}/{p1}/{p2}/{p3}")

        textbox_lines.append(f"mixedDegArea={mixed_area:.2f}")
        textbox_lines.append(f"mixedDegParam={mixed_params}")

        # Check if user-specified budget was used
        user_specified = res.get("user_specified_budget", False)

        if user_specified:
            # User-specified budget: just show the mixed degree MSE without ratio
            textbox_lines.append(f"mixedDegMSE={mixed_mse:.2e}")
            # Show user-specified budget
            if np.isfinite(budget_mse_val):
                textbox_lines.append(f"budgetMSE(user)={budget_mse_val:.2e}")
            else:
                textbox_lines.append("budgetMSE(user)=nan")
        else:
            # Inferred budget: show allMax info and ratio
            if ratio_to_mse0 is not None:
                textbox_lines.append(f"mixedDegMSE={mixed_mse:.2e} ({ratio_to_mse0:.2f}×MSE0)")
            else:
                textbox_lines.append(f"mixedDegMSE={mixed_mse:.2e}")

            textbox_lines.append(f"allMaxDegArea={allmax_area:.2f}" if np.isfinite(allmax_area) else "allMaxDegArea=nan")
            textbox_lines.append(f"allMaxDegParam={allmax_params}")
            textbox_lines.append(f"allMaxDegMSE(MSE0)={allmax_mse:.2e}")

            if np.isfinite(budget_mse_val):
                textbox_lines.append(f"budgetMSE(τ×MSE0)={budget_mse_val:.2e}")
            else:
                textbox_lines.append("budgetMSE(τ×MSE0)=nan")

        # Add upper bound area info if available
        if upper_bound_area is not None and np.isfinite(upper_bound_area):
            textbox_lines.append(f"upperBoundArea={upper_bound_area:.2f}")

        ax2.text(
            0.02, 0.98,
            "\n".join(textbox_lines),
            transform=ax2.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.12),
        )

        # Draw vertical red line for upper bound area
        if upper_bound_area is not None and np.isfinite(upper_bound_area):
            ub_label = 'Upper Bound Area'
            if upper_bound_degree_counts is not None:
                ub_counts_str = '/'.join(str(c) for c in upper_bound_degree_counts)
                ub_label = f'Upper Bound ({ub_counts_str})'
            ax2.axvline(x=upper_bound_area, color='red', linestyle='--', linewidth=2,
                       label=ub_label, alpha=0.8)

        ax2.legend(fontsize=8, loc="best")

    degmax = int(all_results[0]["degree_max"]) if all_results else 3
    nbps = [r["num_breakpoints"] for r in all_results] if all_results else [num_breakpoints]
    num_segs = num_breakpoints - 1
    plt.suptitle(
        f"OVERFIT Piecewise Polynomial Results (HARD) | {num_segs} segments, degree_max {degmax}",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.995])

    save_path = out_dir / f"all_activations_{num_segs}seg_max{degmax}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(save_path)

# =============================================================================
# Output dirs
# =============================================================================
def io_prepare_output_dirs(out_root="outputs_hard_poly", fresh=True):
    out_root = Path(out_root)
    if fresh and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    return {
        "root": out_root,
        "min_area_config": out_root / "min_area_config",
        "pareto_all_configs": out_root / "pareto_optimal_all",  # All Pareto-optimal configs
        "pareto_configs": out_root / "pareto_optimal_feasible",  # Pareto-optimal feasible configs
        "common_area_config": out_root / "common_area_config",  # Phase 3 common-area solutions
        "plots": out_root / "plots",
        "inference": out_root / "inference",
        "logs": out_root / "logs",
        "area_err_optimization": out_root / "area_err_optimization",
    }


def helper_format_output_root_name(degree_max: int, max_error_budget: float, num_functions: int) -> str:
    """Build a descriptive run-folder name that is safe to use as a path."""
    max_error_str = f"{float(max_error_budget):.2e}".replace("+", "").replace(".", "p")
    return f"maxdeg{int(degree_max)}_maxerr{max_error_str}_funcs{int(num_functions)}_results"


# =============================================================================
# Phase 3: Area-Error Joint Optimization
# =============================================================================
def helper_find_best_config_under_bounds(pareto_configs, degree_bounds, degree_max=3, minimize_mse=False):
    """
    Find the best config from Pareto-optimal solutions that respects degree_bounds.

    Args:
        pareto_configs: list of dicts with 'degrees', 'area', 'mse', 'counts' - MUST be Pareto-optimal
        degree_bounds: list [max_d0, max_d1, max_d2, max_d3] - max count for each degree type
        degree_max: maximum degree (default 3)
        minimize_mse: if True, pick config with lowest MSE; if False, pick config with lowest area

    Returns:
        best_config or None if no config satisfies bounds
    """
    if not pareto_configs:
        return None

    candidates = []
    for cfg in pareto_configs:
        deg_counts = helper_degree_type_counts(cfg["degrees"], degree_max=degree_max)

        # Check if this config respects all bounds
        satisfies = True
        for i in range(min(len(deg_counts), len(degree_bounds))):
            if deg_counts[i] > degree_bounds[i]:
                satisfies = False
                break

        if satisfies:
            candidates.append((cfg["area"], cfg["mse"], cfg, deg_counts))

    if not candidates:
        return None

    if minimize_mse:
        # Return config with minimum MSE (best accuracy)
        candidates.sort(key=lambda x: (x[1], x[0]))  # Sort by MSE, then area
    else:
        # Return config with minimum area (tie-break by MSE)
        candidates.sort(key=lambda x: (x[0], x[1]))  # Sort by area, then MSE

    return candidates[0][2]  # Return the config dict


def helper_compute_degree_counts(cfg, degree_max=3):
    """
    Compute degree counts [n0, n1, n2, n3] from cfg["degrees"].
    """
    return helper_degree_type_counts(cfg["degrees"], degree_max=degree_max)


def _phase3_iterative(
    activation_pareto,
    current_solutions,
    initial_upper_bound,
    initial_upper_bound_area,
    search_min,
    search_max,
    degree_max,
    max_rounds,
    compute_upper_bound,
    compute_area_from_deg_counts,
    N_train,
    verbose=True,
):
    """
    Phase 3 ITERATIVE algorithm: Greedy refinement.

    Fast but may not find global optimal (gets stuck in local minima).

    Returns:
        (optimal_budget, nodes_explored, history_entries, trace_entries)
    """
    upper_bound = initial_upper_bound.copy()
    upper_bound_area = initial_upper_bound_area
    nodes_explored = 0
    history_entries = []
    trace_entries = []

    round_num = 0
    while round_num < max_rounds:
        round_num += 1
        improved_this_round = False
        attempts_this_round = 0
        round_start_upper = upper_bound.copy()
        round_start_area = float(upper_bound_area)
        round_action = "no improvement"

        for deg_idx in range(degree_max + 1):
            deg_name = ["constant", "linear", "quadratic", "cubic"][deg_idx] if deg_idx <= 3 else f"deg{deg_idx}"
            max_count = upper_bound[deg_idx]
            if max_count == 0:
                continue

            # Find dominating activations
            dominating_acts = [act_name for act_name, sol in current_solutions.items()
                               if sol["counts"][deg_idx] == max_count]

            for act_name in dominating_acts:
                target_bounds = upper_bound.copy()
                target_bounds[deg_idx] = max_count - 1
                attempts_this_round += 1

                # Find alternative config
                alt_config = helper_find_best_config_under_bounds(
                    [cfg for cfg, _ in activation_pareto[act_name]],
                    target_bounds, degree_max, minimize_mse=False
                )

                if alt_config is not None:
                    new_counts = helper_compute_degree_counts(alt_config, degree_max)
                    if new_counts[deg_idx] < current_solutions[act_name]["counts"][deg_idx]:
                        alt_mse = alt_config.get("mse", None)
                        current_solutions[act_name] = {
                            "degrees": alt_config["degrees"],
                            "area": alt_config["area"],
                            "mse": alt_mse,
                            "counts": new_counts,
                        }

                        new_upper = compute_upper_bound(current_solutions)
                        new_upper_area = compute_area_from_deg_counts(new_upper)

                        if new_upper_area < upper_bound_area:
                            upper_bound = new_upper
                            upper_bound_area = new_upper_area
                            improved_this_round = True
                            nodes_explored += 1
                            round_action = f"reduced {deg_name} via {act_name}"

                            history_entries.append({
                                "round": round_num,
                                "upper_bound": upper_bound.copy(),
                                "upper_bound_area": upper_bound_area,
                                "solutions": {k: v.copy() for k, v in current_solutions.items()},
                                "action": round_action,
                            })

                            if verbose:
                                log_file_only(f"[Phase 3] Round {round_num}: Reduced {deg_name} via {act_name}, "
                                              f"new upper bound: {upper_bound}, area={upper_bound_area:.2f}")
                            break

            if improved_this_round:
                break

        trace_entries.append({
            "step": int(round_num),
            "event": "iterative_round",
            "round": int(round_num),
            "improved": bool(improved_this_round),
            "attempts": int(attempts_this_round),
            "action": round_action,
            "start_upper_bound": round_start_upper,
            "start_upper_bound_area": float(round_start_area),
            "end_upper_bound": upper_bound.copy(),
            "end_upper_bound_area": float(upper_bound_area),
        })

        if not improved_this_round:
            if verbose:
                log_file_only(f"[Phase 3] Iterative converged after {round_num} rounds")
            break

    return upper_bound, nodes_explored, history_entries, trace_entries


def _phase3_tree(
    search_min,
    search_max,
    degree_max,
    check_full_feasibility,
    compute_area_from_deg_counts,
    activation_pareto,
    verbose=True,
):
    """
    Phase 3 TREE algorithm: Proper tree DFS with partial feasibility pruning.

    Tree structure (for degree_max=3):
      - Level 0 (root): empty partial assignment
      - Level 1: b3 values from search_min[3] to search_max[3]
      - Level 2: b2 values from search_min[2] to search_max[2]
      - Level 3: b1 values from search_min[1] to search_max[1]
      - Level 4 (leaves): b0 values from search_min[0] to search_max[0]

    Each leaf is a complete config [b0, b1, b2, b3].
    Intermediate nodes can be pruned if no activation config fits the partial assignment.

    Guaranteed optimal because it explores all feasible leaves.

    Returns:
        (optimal_budget, nodes_explored, history_entries, trace_entries)
    """
    best_budget = None
    best_area = float('inf')
    nodes_explored = 0
    nodes_pruned = 0
    trace_entries = []

    # Order of assignment: b3 -> b2 -> b1 -> b0 (high degree to low)
    # level 0: assign b3 (index 3)
    # level 1: assign b2 (index 2)
    # level 2: assign b1 (index 1)
    # level 3: assign b0 (index 0)
    level_to_deg_idx = list(range(degree_max, -1, -1))  # [3, 2, 1, 0] for degree_max=3

    def check_partial_feasibility(partial_budget, depth):
        """
        Check if the partial assignment (degrees assigned so far) is feasible.

        At depth d, we have assigned: b[level_to_deg_idx[0]], ..., b[level_to_deg_idx[d-1]]
        For each activation, check if at least one config satisfies all assigned constraints.
        """
        if depth == 0:
            return True  # No constraints yet

        for act_name, configs_with_counts in activation_pareto.items():
            has_valid = False
            for cfg, counts in configs_with_counts:
                valid = True
                for lvl in range(depth):
                    deg_idx = level_to_deg_idx[lvl]
                    if counts[deg_idx] > partial_budget[deg_idx]:
                        valid = False
                        break
                if valid:
                    has_valid = True
                    break
            if not has_valid:
                return False
        return True

    def dfs_search(partial_budget, depth):
        """
        DFS through the tree.

        Args:
            partial_budget: Current budget assignment (complete array, but only
                           indices level_to_deg_idx[0:depth] are meaningful)
            depth: Current depth (0 = root, degree_max+1 = leaf)
        """
        nonlocal best_budget, best_area, nodes_explored, nodes_pruned
        nodes_explored += 1
        step_idx = int(nodes_explored)

        # Check partial feasibility (prune if no config can satisfy assigned constraints)
        if not check_partial_feasibility(partial_budget, depth):
            nodes_pruned += 1
            trace_entries.append({
                "step": step_idx,
                "event": "pruned_partial",
                "depth": int(depth),
                "budget": partial_budget.copy(),
                "area": None,
                "feasible": False,
                "pruned": True,
            })
            return

        # Leaf node: complete assignment
        if depth == degree_max + 1:
            # Check full feasibility (should pass if partial checks passed, but verify)
            if check_full_feasibility(partial_budget):
                area = compute_area_from_deg_counts(partial_budget)
                if area < best_area:
                    best_area = area
                    best_budget = partial_budget.copy()
                    trace_entries.append({
                        "step": step_idx,
                        "event": "leaf_feasible_improved",
                        "depth": int(depth),
                        "budget": partial_budget.copy(),
                        "area": float(area),
                        "feasible": True,
                        "pruned": False,
                    })
                    if verbose:
                        log_file_only(f"[Phase 3] Tree found feasible: {partial_budget}, area={area:.2f}")
                else:
                    trace_entries.append({
                        "step": step_idx,
                        "event": "leaf_feasible",
                        "depth": int(depth),
                        "budget": partial_budget.copy(),
                        "area": float(area),
                        "feasible": True,
                        "pruned": False,
                    })
            else:
                trace_entries.append({
                    "step": step_idx,
                    "event": "leaf_infeasible",
                    "depth": int(depth),
                    "budget": partial_budget.copy(),
                    "area": None,
                    "feasible": False,
                    "pruned": False,
                })
            return

        trace_entries.append({
            "step": step_idx,
            "event": "expand_internal",
            "depth": int(depth),
            "budget": partial_budget.copy(),
            "area": None,
            "feasible": None,
            "pruned": False,
        })

        # Internal node: expand children (try all values for next degree)
        deg_idx = level_to_deg_idx[depth]
        for val in range(search_min[deg_idx], search_max[deg_idx] + 1):
            child_budget = partial_budget.copy()
            child_budget[deg_idx] = val
            dfs_search(child_budget, depth + 1)

    # Start DFS from root (empty assignment). Seed with lower bound values so
    # the root budget representation is aligned with the actual search range.
    initial_budget = search_min.copy()
    dfs_search(initial_budget, 0)

    if verbose:
        log_file_only(f"[Phase 3] Tree search explored {nodes_explored} nodes, pruned {nodes_pruned} subtrees")

    return best_budget, nodes_explored, [], trace_entries


def _phase3_bruteforce(
    search_min,
    search_max,
    degree_max,
    compute_area_from_deg_counts,
    check_full_feasibility,
    total_budgets,
    verbose=True,
):
    """
    Phase 3 BRUTE-FORCE algorithm: Generate all budgets, sort by area.

    Guaranteed to find global optimal. O(N) memory where N = total budgets.

    Returns:
        (optimal_budget, nodes_explored, history_entries, trace_entries)
    """
    # Generate all budget combinations
    all_budgets = []
    for b0 in range(search_min[0], search_max[0] + 1):
        for b1 in range(search_min[1], search_max[1] + 1):
            for b2 in range(search_min[2], search_max[2] + 1):
                for b3 in range(search_min[3], search_max[3] + 1):
                    budget = [b0, b1, b2, b3]
                    area = compute_area_from_deg_counts(budget)
                    all_budgets.append((area, budget))

    # Sort by area (ascending)
    all_budgets.sort(key=lambda x: x[0])

    if verbose:
        log_file_only(f"[Phase 3] Generated {len(all_budgets)} budget combinations")

    # Check feasibility in area order
    optimal_budget = None
    nodes_explored = 0

    for area, budget in all_budgets:
        nodes_explored += 1
        if check_full_feasibility(budget):
            optimal_budget = budget
            if verbose:
                log_file_only(f"[Phase 3] FOUND OPTIMAL at rank {nodes_explored}/{total_budgets}: {budget}, area={area:.2f}")
            break

    if verbose:
        log_file_only(f"[Phase 3] Brute-force checked {nodes_explored} budgets")

    # Keep only initial/final in downstream CSV for brute-force (no step trace).
    return optimal_budget, nodes_explored, [], []


def _phase3_bestfirst(
    search_min,
    search_max,
    degree_max,
    compute_area_from_deg_counts,
    check_full_feasibility,
    total_budgets,
    verbose=True,
):
    """
    Phase 3 BEST-FIRST algorithm: Priority queue / Dijkstra-style search.

    Guaranteed to find global optimal. O(frontier) memory, typically much less than O(N).

    Returns:
        (optimal_budget, nodes_explored, history_entries, trace_entries)
    """
    import heapq

    start_budget = tuple(search_min)
    start_area = compute_area_from_deg_counts(list(start_budget))

    heap = [(start_area, start_budget)]
    visited = set()
    visited.add(start_budget)

    optimal_budget = None
    nodes_explored = 0
    trace_entries = []

    while heap:
        heap_size_before_pop = len(heap)
        area, budget_tuple = heapq.heappop(heap)
        budget = list(budget_tuple)
        nodes_explored += 1

        is_feasible = check_full_feasibility(budget)
        if is_feasible:
            optimal_budget = budget
            trace_entries.append({
                "step": int(nodes_explored),
                "event": "feasible_found",
                "budget": budget.copy(),
                "area": float(area),
                "feasible": True,
                "expanded_neighbors": 0,
                "expanded_neighbor_configs": [],
                "frontier_size_before_pop": int(heap_size_before_pop),
                "frontier_size_after_expand": int(len(heap)),
            })
            if verbose:
                log_file_only(f"[Phase 3] FOUND OPTIMAL after {nodes_explored} nodes: {budget}, area={area:.2f}")
            break

        # Expand neighbors (increment each dimension by 1)
        expanded_neighbors = 0
        expanded_neighbor_configs = []
        for i in range(degree_max + 1):
            if budget[i] < search_max[i]:
                new_budget = budget.copy()
                new_budget[i] += 1
                new_tuple = tuple(new_budget)

                if new_tuple not in visited:
                    visited.add(new_tuple)
                    new_area = compute_area_from_deg_counts(new_budget)
                    heapq.heappush(heap, (new_area, new_tuple))
                    expanded_neighbors += 1
                    expanded_neighbor_configs.append(new_budget.copy())

        trace_entries.append({
            "step": int(nodes_explored),
            "event": "expand",
            "budget": budget.copy(),
            "area": float(area),
            "feasible": False,
            "expanded_neighbors": int(expanded_neighbors),
            "expanded_neighbor_configs": expanded_neighbor_configs,
            "frontier_size_before_pop": int(heap_size_before_pop),
            "frontier_size_after_expand": int(len(heap)),
        })

    if verbose:
        log_file_only(f"[Phase 3] Best-first search explored {nodes_explored} nodes (of {total_budgets} possible)")

    return optimal_budget, nodes_explored, [], trace_entries


def phase3_optimize_common_area(
    all_results_by_activation,
    degree_max=3,
    max_rounds=50,
    verbose=True,
    algorithm="bestfirst",
):
    """
    Phase 3: Find optimal minimum-area common bound.

    Supports 4 different search algorithms:
    - 'iterative': Greedy iterative refinement (fast but may not find optimal)
    - 'tree': DFS tree search with pruning
    - 'bruteforce': Generate all budgets, sort by area (guaranteed optimal, O(N) memory)
    - 'bestfirst': Priority queue / Dijkstra-style (guaranteed optimal, O(frontier) memory)

    Args:
        all_results_by_activation: dict mapping activation_name -> result dict
        degree_max: maximum degree (default 3)
        max_rounds: max iterations for iterative algorithm
        verbose: print progress
        algorithm: 'iterative', 'tree', 'bruteforce', or 'bestfirst' (default)

    Returns:
        dict with final_upper_bound, final_solutions, history, and search_trace.
    """
    activation_names = list(all_results_by_activation.keys())

    # Get N_train for MSE conversion (assume all activations have same N)
    first_res = next(iter(all_results_by_activation.values()))
    N_train = len(first_res.get("x_train", [])) or 2048

    # =========================================================================
    # Initialize from Phase 2 min-area solutions (already CD-refined)
    # =========================================================================
    current_solutions = {}
    initial_solutions = {}

    for act_name, res in all_results_by_activation.items():
        degrees_final = res.get("degrees_final", [])
        phase2_area = res.get("total_area", None)
        phase2_mse = res.get("final_train_mse", None)
        dp_frontier = res.get("dp_frontier_points", [])
        source_cfg = helper_find_pareto_config_by_degrees(dp_frontier, degrees_final)

        counts = helper_compute_degree_counts({"degrees": degrees_final}, degree_max)

        initial_solutions[act_name] = {
            "degrees": degrees_final,
            "area": phase2_area,
            "mse": phase2_mse,
            "counts": counts,
            "source_config": source_cfg,
        }

        current_solutions[act_name] = {
            "degrees": degrees_final,
            "area": phase2_area,
            "mse": phase2_mse,
            "counts": counts,
            "source_config": source_cfg,
        }

    # =========================================================================
    # Helper functions
    # =========================================================================
    def compute_upper_bound(solutions):
        upper = [0] * (degree_max + 1)
        for act_name, sol in solutions.items():
            counts = sol["counts"]
            for i in range(len(upper)):
                if i < len(counts):
                    upper[i] = max(upper[i], counts[i])
        return upper

    def compute_area_from_deg_counts(deg_counts):
        c_counts = helper_degree_counts_to_c_counts(deg_counts)
        return helper_hardware_area(*c_counts)

    # =========================================================================
    # Initial upper bound from Phase 2 solutions
    # =========================================================================
    initial_upper_bound = compute_upper_bound(current_solutions)
    initial_upper_bound_area = compute_area_from_deg_counts(initial_upper_bound)

    history = [{
        "round": 0,
        "upper_bound": initial_upper_bound.copy(),
        "upper_bound_area": initial_upper_bound_area,
        "solutions": {k: v.copy() for k, v in current_solutions.items()},
        "action": "initial (Phase 2)",
    }]

    if verbose:
        log_file_only(f"[Phase 3] Algorithm: {algorithm}")
        log_file_only(f"[Phase 3] Initial upper bound from Phase 2: {initial_upper_bound}, area={initial_upper_bound_area:.2f}")

    # =========================================================================
    # Precompute feasible Pareto configs for each activation
    # =========================================================================
    activation_pareto = {}
    for act_name, res in all_results_by_activation.items():
        dp_frontier = res.get("dp_frontier_points", [])
        pareto_feasible = [p for p in dp_frontier if p.get("is_feasible", False)]
        configs_with_counts = []
        for cfg in pareto_feasible:
            counts = helper_compute_degree_counts(cfg, degree_max)
            configs_with_counts.append((cfg, counts))

        # Fallback: when no feasible Pareto configs exist, inject the
        # lowest-MSE Pareto member from the full frontier.
        if not configs_with_counts:
            fallback_cfg = helper_pick_pareto_fallback_config(dp_frontier)
            if fallback_cfg is not None:
                fallback_counts = helper_compute_degree_counts(fallback_cfg, degree_max)
                configs_with_counts.append((fallback_cfg, fallback_counts))
            else:
                phase2_sol = initial_solutions[act_name]
                fallback_cfg = {
                    "degrees": phase2_sol["degrees"],
                    "area": phase2_sol["area"],
                    "mse": phase2_sol["mse"],
                    "is_feasible": True,
                }
                configs_with_counts.append((fallback_cfg, phase2_sol["counts"]))

        activation_pareto[act_name] = configs_with_counts

    # =========================================================================
    # Compute search bounds
    # =========================================================================
    search_min = [0] * (degree_max + 1)
    search_max = [0] * (degree_max + 1)

    for deg_idx in range(degree_max + 1):
        min_counts_per_activation = []
        max_counts_per_activation = []

        for act_name, configs_with_counts in activation_pareto.items():
            if configs_with_counts:
                counts_for_deg = [c[1][deg_idx] for c in configs_with_counts]
                min_counts_per_activation.append(min(counts_for_deg))
                max_counts_per_activation.append(max(counts_for_deg))
            else:
                min_counts_per_activation.append(initial_solutions[act_name]["counts"][deg_idx])
                max_counts_per_activation.append(initial_solutions[act_name]["counts"][deg_idx])

        search_min[deg_idx] = max(min_counts_per_activation) if min_counts_per_activation else 0
        search_max[deg_idx] = max(max_counts_per_activation) if max_counts_per_activation else 0

    if verbose:
        log_file_only(f"[Phase 3] Search range: min={search_min}, max={search_max}")

    def check_full_feasibility(budget):
        """Check if a complete budget is feasible for all activations."""
        for act_name, configs_with_counts in activation_pareto.items():
            has_valid = False
            for cfg, counts in configs_with_counts:
                if all(counts[i] <= budget[i] for i in range(degree_max + 1)):
                    has_valid = True
                    break
            if not has_valid:
                return False
        return True

    # Compute total search space size
    total_budgets = 1
    for i in range(degree_max + 1):
        total_budgets *= (search_max[i] - search_min[i] + 1)

    if verbose:
        log_file_only(f"[Phase 3] Total search space: {total_budgets} budget combinations")

    # =========================================================================
    # Dispatch to algorithm-specific helper
    # =========================================================================
    converged = True
    search_trace = []

    if algorithm == "iterative":
        optimal_budget, nodes_explored, extra_history, search_trace = _phase3_iterative(
            activation_pareto=activation_pareto,
            current_solutions=current_solutions,
            initial_upper_bound=initial_upper_bound,
            initial_upper_bound_area=initial_upper_bound_area,
            search_min=search_min,
            search_max=search_max,
            degree_max=degree_max,
            max_rounds=max_rounds,
            compute_upper_bound=compute_upper_bound,
            compute_area_from_deg_counts=compute_area_from_deg_counts,
            N_train=N_train,
            verbose=verbose,
        )
        history.extend(extra_history)

    elif algorithm == "tree":
        optimal_budget, nodes_explored, extra_history, search_trace = _phase3_tree(
            search_min=search_min,
            search_max=search_max,
            degree_max=degree_max,
            check_full_feasibility=check_full_feasibility,
            compute_area_from_deg_counts=compute_area_from_deg_counts,
            activation_pareto=activation_pareto,
            verbose=verbose,
        )

    elif algorithm == "bruteforce":
        optimal_budget, nodes_explored, extra_history, search_trace = _phase3_bruteforce(
            search_min=search_min,
            search_max=search_max,
            degree_max=degree_max,
            compute_area_from_deg_counts=compute_area_from_deg_counts,
            check_full_feasibility=check_full_feasibility,
            total_budgets=total_budgets,
            verbose=verbose,
        )

    elif algorithm == "bestfirst":
        optimal_budget, nodes_explored, extra_history, search_trace = _phase3_bestfirst(
            search_min=search_min,
            search_max=search_max,
            degree_max=degree_max,
            compute_area_from_deg_counts=compute_area_from_deg_counts,
            check_full_feasibility=check_full_feasibility,
            total_budgets=total_budgets,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'iterative', 'tree', 'bruteforce', 'bestfirst'")

    # =========================================================================
    # Use optimal budget (or fall back to initial)
    # =========================================================================
    if optimal_budget is not None:
        upper_bound = optimal_budget
        upper_bound_area = compute_area_from_deg_counts(upper_bound)
        if verbose:
            reduction = (1 - upper_bound_area / initial_upper_bound_area) * 100
            log_file_only(f"[Phase 3] Optimal bound: {upper_bound}, area={upper_bound_area:.2f}")
            log_file_only(f"[Phase 3] Area reduction: {reduction:.1f}%")
    else:
        upper_bound = initial_upper_bound
        upper_bound_area = initial_upper_bound_area
        if verbose:
            log_file_only(f"[Phase 3] No feasible budget found, keeping initial bound")

    # Update current_solutions to use configs that fit within optimal budget
    for act_name, configs_with_counts in activation_pareto.items():
        best_cfg = None
        best_area = float('inf')
        best_counts = None

        for cfg, counts in configs_with_counts:
            if all(counts[i] <= upper_bound[i] for i in range(degree_max + 1)):
                cfg_area = cfg.get("area", float('inf'))
                if cfg_area < best_area:
                    best_cfg = cfg
                    best_area = cfg_area
                    best_counts = counts

        if best_cfg is not None:
            best_mse = best_cfg.get("mse", None)
            current_solutions[act_name] = {
                "degrees": best_cfg["degrees"],
                "area": best_cfg["area"],
                "mse": best_mse,
                "counts": best_counts,
                "source_config": best_cfg,
            }

    # Add final state to history (only if different from initial)
    if upper_bound != initial_upper_bound:
        history.append({
            "round": len(history),
            "upper_bound": upper_bound.copy(),
            "upper_bound_area": upper_bound_area,
            "solutions": {k: v.copy() for k, v in current_solutions.items()},
            "action": f"{algorithm} optimal ({nodes_explored}/{total_budgets})",
        })

    # =========================================================================
    # Final Selection: Pick lowest-MSE solution for each activation within the
    # converged upper bound. This gives best accuracy while respecting the
    # common area constraint.
    # =========================================================================
    if verbose:
        log_file_only(f"[Phase 3] Selecting lowest-MSE solutions within final upper bound...")

    for act_name, res in all_results_by_activation.items():
        dp_frontier = res.get("dp_frontier_points", [])
        pareto_feasible = [p for p in dp_frontier if p.get("is_feasible", False)]

        if pareto_feasible:
            best_config = helper_find_best_config_under_bounds(
                pareto_feasible, upper_bound, degree_max, minimize_mse=True
            )

            if best_config is not None:
                new_counts = helper_compute_degree_counts(best_config, degree_max)
                best_mse = best_config.get("mse", None)

                old_sol = current_solutions[act_name]
                old_mse = old_sol.get("mse")
                old_counts = old_sol.get("counts", [])
                old_area = old_sol.get("area")
                if best_mse is not None and (old_mse is None or best_mse < old_mse):
                    current_solutions[act_name] = {
                        "degrees": best_config["degrees"],
                        "area": best_config["area"],
                        "mse": best_mse,
                        "counts": new_counts,
                        "source_config": best_config,
                    }
                    if verbose:
                        old_counts_str = '/'.join(str(c) for c in old_counts) if old_counts else "N/A"
                        new_counts_str = '/'.join(str(c) for c in new_counts)
                        old_area_str = f"{old_area:.2f}" if old_area is not None else "N/A"
                        log_file_only(f"[Phase 3] {act_name}: Reselected for lowest-MSE, "
                              f"counts {old_counts_str} -> {new_counts_str}, "
                              f"area {old_area_str} -> {best_config['area']:.2f}, "
                              f"MSE {old_mse:.2e} -> {best_mse:.2e}")

    # =========================================================================
    # CD Refinement: MANDATORY coordinate descent on final Phase 3 solutions
    # =========================================================================
    if verbose:
        log_file_only(f"[Phase 3] Running mandatory CD refinement on final solutions...")

    refined_solutions = {}
    for act_name, sol in current_solutions.items():
        res = all_results_by_activation[act_name]
        phase3_degrees = sol["degrees"]
        mse_before_cd = sol.get("mse", None)

        x_train = res.get("x_train")
        y_train = res.get("y_train")

        dp_frontier = res.get("dp_frontier_points", [])
        matching_cfg = sol.get("source_config")
        if matching_cfg is None:
            matching_cfg = helper_find_pareto_config_by_degrees(dp_frontier, phase3_degrees)

        if matching_cfg is not None and "breakpoints_idx" in matching_cfg:
            bp_idx_init = matching_cfg["breakpoints_idx"].clone() if torch.is_tensor(matching_cfg["breakpoints_idx"]) else torch.tensor(matching_cfg["breakpoints_idx"], dtype=torch.long)
        else:
            bp_final_x = res.get("breakpoints_x_final")
            if bp_final_x is not None:
                bp_final_x = np.asarray(bp_final_x)
                bp_idx_init = torch.tensor(
                    [0] + [int(np.searchsorted(x_train, bx)) for bx in bp_final_x[1:-1]] + [len(x_train)],
                    dtype=torch.long
                )
            else:
                num_segments = len(phase3_degrees)
                bp_idx_init = torch.linspace(0, len(x_train), num_segments + 1).round().long()

        if x_train is not None and y_train is not None:
            x_t = torch.from_numpy(np.array(x_train)).float().to(device)
            y_t = torch.from_numpy(np.array(y_train)).float().to(device)
            bp_idx_init = bp_idx_init.to(device)

            bp_idx_opt, mse_hist, best_it = phase2_refine_breakpoints(
                x=x_t,
                y=y_t,
                bp_idx_init=bp_idx_init,
                degrees=phase3_degrees,
                num_outer_iters=None,
                min_seg_points=8,
                lam=0.0,
                verbose=False,
            )

            coeffs, y_fit, _, mse = helper_fit_all_segments(
                x_t, y_t, bp_idx_opt, phase3_degrees, lam=0.0
            )

            bp_x_final = torch.stack([helper_boundary_x_from_index(x_t, int(idx)) for idx in bp_idx_opt])
            bp_x_final_np = bp_x_final.detach().cpu().numpy()

            refined_mse = float(mse.item())

            refined_solutions[act_name] = {
                "degrees": phase3_degrees,
                "area": sol["area"],
                "mse": refined_mse,
                "mse_before_cd": mse_before_cd,
                "counts": sol["counts"],
                "breakpoints": bp_x_final_np,
                "y_fit": y_fit.detach().cpu().numpy(),
                "coeffs": coeffs,
            }

            if verbose:
                improvement = (mse_before_cd - refined_mse) / max(1e-12, mse_before_cd) * 100 if mse_before_cd else 0
                log_file_only(f"[Phase 3] {act_name}: MSE {mse_before_cd:.2e} -> {refined_mse:.2e} (improvement: {improvement:.2f}%)")
        else:
            refined_solutions[act_name] = sol.copy()
            refined_solutions[act_name]["mse_before_cd"] = mse_before_cd
            if verbose:
                log_file_only(f"[Phase 3] {act_name}: WARNING - could not run CD (missing data)")

    return {
        "final_upper_bound": upper_bound,
        "final_upper_bound_area": upper_bound_area,
        "initial_solutions": initial_solutions,
        "final_solutions": refined_solutions,
        "history": history,
        "algorithm": algorithm,
        "search_trace": search_trace,
        "converged": converged,
        "total_rounds": len(history) - 1,
        "nodes_explored": nodes_explored,
        "total_budgets": total_budgets,
        "N_train": N_train,
    }


def viz_phase3_optimization(phase3_output, out_dir, num_segments, degree_max=3):
    """
    Create visualization for Phase 3 area optimization.

    Two columns showing improvement from initial to final:
    - Col 0: Per-activation Phase 2 vs Phase 3 segment-count comparison
    - Col 1: Upper bound area evolution over optimization rounds
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = phase3_output["history"]
    final_solutions = phase3_output["final_solutions"]
    initial_solutions = phase3_output.get("initial_solutions", {})

    if len(history) <= 1:
        # No optimization happened
        log_file_only("[Phase 3] No optimization iterations to visualize")
        return None

    activation_names = list(initial_solutions.keys()) if initial_solutions else list(final_solutions.keys())
    n_acts = len(activation_names)

    # Extract data for plotting
    rounds = [h["round"] for h in history]
    areas = [h["upper_bound_area"] for h in history]

    # Keep the figure compact so text appears larger relative to the plot.
    fig_w = max(17.5, 4.8 + 2.2 * n_acts + 1.0 * (degree_max + 1))
    fig_h = max(7.4, 6.2 + 0.25 * (degree_max + 1))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    # ============================================================
    # Left plot (Col 0): Phase 2 vs Phase 3 side-by-side per activation
    # ============================================================
    ax0 = axes[0]

    x = np.arange(n_acts)

    deg_label_map = {
        0: "deg0 (const)",
        1: "deg1 (linear)",
        2: "deg2 (quad)",
        3: "deg3 (cubic)",
    }
    deg_labels = [deg_label_map.get(d, f"deg{d}") for d in range(degree_max + 1)]

    cmap = plt.colormaps.get_cmap("tab10").resampled(max(10, degree_max + 1))
    phase3_colors = [cmap(i) for i in range(degree_max + 1)]

    def _lighter(color, blend=0.55):
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb + (1.0 - rgb) * blend, 0.0, 1.0))

    phase2_colors = [_lighter(c) for c in phase3_colors]

    def _count_at(sol, deg_idx):
        counts = sol.get("counts", []) if isinstance(sol, dict) else []
        return int(counts[deg_idx]) if deg_idx < len(counts) else 0

    # Build explicit degree-wise pairs with a visible gap between degree groups:
    # [deg0: P2,P3] [deg1: P2,P3] [deg2: P2,P3] ...
    num_deg = max(1, degree_max + 1)
    group_width = 0.88
    deg_group_gap = 0.03
    width = (group_width - deg_group_gap * (num_deg - 1)) / (2 * num_deg)
    offset_start = -group_width / 2.0 + width / 2.0
    pair_data = []
    y_max_val = 0

    for deg_idx in range(degree_max + 1):
        phase2_counts = [_count_at(initial_solutions.get(act, {}), deg_idx) for act in activation_names]
        phase3_counts = [_count_at(final_solutions.get(act, {}), deg_idx) for act in activation_names]

        deg_group_start = offset_start + deg_idx * (2 * width + deg_group_gap)
        off_phase2 = deg_group_start
        off_phase3 = deg_group_start + width

        ax0.bar(x + off_phase2, phase2_counts, width, color=phase2_colors[deg_idx], alpha=0.95)
        ax0.bar(x + off_phase3, phase3_counts, width, color=phase3_colors[deg_idx], alpha=0.95)

        x_phase2 = x + off_phase2
        x_phase3 = x + off_phase3
        for act_i in range(n_acts):
            c2 = int(phase2_counts[act_i])
            c3 = int(phase3_counts[act_i])
            pair_data.append((x_phase2[act_i], c2, x_phase3[act_i], c3))
            y_max_val = max(y_max_val, c2, c3)

    # Add arrows from each Phase 2 bar to its Phase 3 counterpart to show change direction.
    for x2, c2, x3, c3 in pair_data:
        delta = c3 - c2
        if delta > 0:
            arrow_color = "#2ca02c"  # increase
        elif delta < 0:
            arrow_color = "#d62728"  # decrease
        else:
            arrow_color = "#666666"  # unchanged

        # For 0->0 pairs, lift annotation off the x-axis so +0 is visible.
        if c2 == 0 and c3 == 0:
            y_start = 0.18
            y_end = 0.18
        else:
            y_start = c2 + 0.06
            y_end = c3 + 0.06
        ax0.annotate(
            "",
            xy=(x3, y_end),
            xytext=(x2, y_start),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.25, alpha=0.9),
            zorder=6,
        )

        # Place delta labels at the arrow midpoint for accurate positioning.
        y_mid = (y_start + y_end) / 2.0
        ax0.annotate(
            f"{delta:+d}",
            xy=((x2 + x3) / 2.0, y_mid),
            xytext=(0, 0),
            textcoords="offset points",
            color=arrow_color,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            alpha=0.98,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.76),
            zorder=7,
            clip_on=False,
        )

    ax0.set_ylim(0, y_max_val + 2.2)

    ax0.set_xlabel("Activation Function", fontsize=16)
    ax0.set_ylabel("Segment Count", fontsize=16)
    ax0.set_title("Per-Activation Segment Counts\n(Phase 2 lighter vs Phase 3 saturated)", fontsize=18)
    ax0.set_xticks(x)
    ax0.set_xticklabels(activation_names, rotation=15, fontsize=14)
    ax0.tick_params(axis='y', labelsize=14)
    ax0.grid(True, alpha=0.3, axis='y')

    phase_handles = [
        Patch(facecolor=(0.78, 0.78, 0.78), label="P2 light"),
        Patch(facecolor=(0.35, 0.35, 0.35), label="P3 sat"),
    ]
    degree_handles = [Patch(facecolor=phase3_colors[d], label=f"d{d}") for d in range(degree_max + 1)]
    arrow_handles = [
        Line2D([0], [0], color="#2ca02c", lw=1.5, label="inc"),
        Line2D([0], [0], color="#d62728", lw=1.5, label="dec"),
        Line2D([0], [0], color="#666666", lw=1.5, label="same"),
    ]
    legend_handles = phase_handles + degree_handles + arrow_handles
    ax0.legend(
        handles=legend_handles,
        fontsize=11,
        ncol=len(legend_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        framealpha=0.92,
        columnspacing=0.8,
        handlelength=1.3,
        handletextpad=0.45,
        borderpad=0.35,
    )

    initial_upper = history[0].get("upper_bound", [])
    final_upper = phase3_output["final_upper_bound"]
    initial_area = history[0]['upper_bound_area'] if len(history) > 0 else 0
    textbox_lines = [
        f"Upper Bound: {'/'.join(str(c) for c in initial_upper)} -> {'/'.join(str(c) for c in final_upper)}",
        f"Initial Area: {initial_area:.2f} | Final Area: {phase3_output['final_upper_bound_area']:.2f}",
    ]
    summary_line = "\n".join(textbox_lines)
    ax0.text(
        0.5, 0.88,
        summary_line,
        transform=ax0.transAxes,
        fontsize=12,
        va="center",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", alpha=0.18),
    )

    # ============================================================
    # Right plot (Col 1): Upper bound area evolution over rounds
    # ============================================================
    ax1 = axes[1]
    ax1.plot(rounds, areas, 'b-o', linewidth=2, markersize=8, label='Upper Bound Area')
    ax1.fill_between(rounds, areas, alpha=0.2)

    # Mark initial and final
    ax1.scatter([rounds[0]], [areas[0]], color='red', s=150, zorder=5,
                marker='s', label=f'Initial: {areas[0]:.2f}')
    ax1.scatter([rounds[-1]], [areas[-1]], color='green', s=150, zorder=5,
                marker='*', label=f'Final: {areas[-1]:.2f}')

    ax1.set_xlabel("Optimization Round", fontsize=16)
    ax1.set_ylabel("Upper Bound Area", fontsize=16)
    ax1.set_title(f"Phase 3: Upper Bound Area Convergence\n({num_segments} segments)", fontsize=18)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # Add text annotations for key transitions
    for i, h in enumerate(history):
        if i > 0 and h["upper_bound_area"] < history[i-1]["upper_bound_area"]:
            ax1.annotate(
                h.get("action", ""),
                xy=(rounds[i], areas[i]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=11,
                alpha=0.7,
                arrowprops=dict(arrowstyle='->', alpha=0.5)
            )

    # Add textbox with summary
    nodes_explored = phase3_output.get('nodes_explored', 0)
    total_budgets = phase3_output.get('total_budgets', 0)
    textbox_lines = [
        f"Initial Area: {history[0]['upper_bound_area']:.2f}",
        f"Final Area: {phase3_output['final_upper_bound_area']:.2f}",
        f"Reduction: {(1 - phase3_output['final_upper_bound_area']/history[0]['upper_bound_area'])*100:.1f}%",
        f"Nodes Explored: {nodes_explored}/{total_budgets}",
    ]
    ax1.text(
        0.98, 0.98,
        "\n".join(textbox_lines),
        transform=ax1.transAxes,
        fontsize=12,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout(pad=1.8, w_pad=3.0)

    save_path = out_dir / f"phase3_optimization_{num_segments}seg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(save_path)


def viz_phase3_pareto_overlay(phase3_output, all_results_by_activation, out_dir,
                                     num_segments, degree_max=3, max_error_budget=None,
                                     refine_plausible_configs=False):
    """
    Create a Pareto-style plot showing all activations' frontiers with Phase 3 solutions highlighted.
    Similar to col3 of all activation plots but overlaid and with upper bound evolution lines.

    If refine_plausible_configs is True, feasible points (below budget) are colored differently.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = phase3_output["history"]
    final_solutions = phase3_output["final_solutions"]
    N_train = phase3_output.get("N_train", 2048)

    fig, ax = plt.subplots(figsize=(14, 10))

    activation_names = list(all_results_by_activation.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(activation_names)))

    # Plot each activation's Pareto frontier
    for idx, (act_name, res) in enumerate(all_results_by_activation.items()):
        color = colors[idx]

        # Use Pareto frontier for the main curve (guarantees monotonicity)
        # Update MSE values from pareto_feasible_configs if available (may be refined after 2nd CD)
        dp_frontier = res.get("dp_frontier_points", [])
        pareto_feasible = res.get("pareto_feasible_configs", [])

        # Build lookup from degrees -> config in pareto_feasible (has potentially updated MSE)
        pareto_lookup = {tuple(cfg.get("degrees", [])): cfg for cfg in pareto_feasible if cfg.get("degrees")}

        # Update frontier points with refined MSE values where available
        frontier = []
        for p in dp_frontier:
            p_degrees = tuple(p.get("degrees", []))
            if p_degrees and p_degrees in pareto_lookup:
                # Use the MSE from pareto_feasible_configs (may be refined)
                updated_p = p.copy()
                updated_p["mse"] = pareto_lookup[p_degrees]["mse"]
                frontier.append(updated_p)
            else:
                frontier.append(p)

        if not frontier:
            frontier = pareto_feasible

        if frontier:
            # Sort by area
            frontier_sorted = sorted(frontier, key=lambda t: float(t.get("area", 0) or 0))

            # Collect all points with feasibility info
            xs_feasible, ys_feasible = [], []
            xs_nonfeasible, ys_nonfeasible = [], []
            xs_all, ys_all = [], []
            budget_mse = max_error_budget

            for t in frontier_sorted:
                a = t.get("area", None)
                mm = t.get("mse", None)
                if a is not None and mm is not None and np.isfinite(float(a)):
                    mse_val = float(mm)
                    xs_all.append(float(a))
                    ys_all.append(mse_val)

                    # Use is_feasible flag if available, otherwise fall back to budget comparison
                    is_feas = t.get("is_feasible", None)
                    if is_feas is None and budget_mse is not None:
                        is_feas = mse_val <= budget_mse

                    if is_feas:
                        xs_feasible.append(float(a))
                        ys_feasible.append(mse_val)
                    else:
                        xs_nonfeasible.append(float(a))
                        ys_nonfeasible.append(mse_val)

            if xs_all:
                # Always show feasible vs non-feasible split
                # Plot non-feasible points (above budget) in lighter color
                if len(xs_nonfeasible) > 0:
                    ax.plot(xs_nonfeasible, ys_nonfeasible, marker='o', linestyle='none',
                           markersize=5, alpha=0.4, color=color)

                # Plot feasible points (below budget) in brighter color
                if len(xs_feasible) > 0:
                    ax.plot(xs_feasible, ys_feasible, marker='s', linestyle='none',
                           markersize=6, alpha=0.9, color='C1')

                # Connect with line
                ax.plot(xs_all, ys_all, linestyle='-', linewidth=1.5, alpha=0.4, color=color,
                       label=f'{act_name} frontier')

        # Mark Phase 3 final solution (now stores MSE directly)
        final_sol = final_solutions.get(act_name, {})
        if final_sol:
            sol_area = final_sol.get("area", None)
            sol_mse = final_sol.get("mse", None)
            if sol_area is not None and sol_mse is not None:
                ax.scatter([sol_area], [sol_mse], marker='*', s=300, color=color,
                          edgecolors='black', linewidths=1.5, zorder=10,
                          label=f'{act_name} Phase3 solution')

    # Draw vertical lines for upper bound evolution (initial and final only with tree search)
    areas_history = [h["upper_bound_area"] for h in history]
    unique_areas = sorted(set(areas_history), reverse=True)

    cmap_lines = plt.cm.Reds(np.linspace(0.3, 0.9, len(unique_areas)))
    for i, area in enumerate(unique_areas):
        round_idx = next(j for j, h in enumerate(history) if h["upper_bound_area"] == area)
        if round_idx == 0:
            label = "Phase 2 Initial"
        elif round_idx == len(history) - 1:
            label = "Phase 3 Optimal"
        else:
            label = None
        ax.axvline(x=area, color=cmap_lines[i], linestyle='--', linewidth=1.5,
                  alpha=0.7, label=label)

    # Budget line if provided
    if max_error_budget is not None:
        ax.axhline(y=max_error_budget, color='green', linestyle='--', linewidth=2,
                  alpha=0.8, label=f'Budget MSE={max_error_budget:.2e}')

    ax.set_yscale('log')
    ax.set_xlabel("Hardware Area Cost (μm²)", fontsize=12)
    ax.set_ylabel("MSE (log scale)", fontsize=12)
    ax.set_title(f"Phase 3: Area-Error Pareto with Upper Bound Evolution\n({num_segments} segments, degree_max={degree_max})",
                fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # Legend - show only unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right', ncol=2)

    # Textbox with summary
    initial_ub = history[0]["upper_bound"]
    final_ub = phase3_output["final_upper_bound"]
    textbox_lines = [
        f"Initial UB: {'/'.join(str(c) for c in initial_ub)}",
        f"Final UB: {'/'.join(str(c) for c in final_ub)}",
        f"Initial Area: {history[0]['upper_bound_area']:.2f}",
        f"Final Area: {phase3_output['final_upper_bound_area']:.2f}",
    ]
    ax.text(
        0.02, 0.02,
        "\n".join(textbox_lines),
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()

    save_path = out_dir / f"phase3_pareto_overlay_{num_segments}seg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(save_path)


def viz_phase3_comparison(phase3_output, all_results_by_activation, out_dir,
                                  num_segments, degree_max=3, max_error_budget=None,
                                  refine_plausible_configs=False):
    """
    Create the main 3-column comparison visualization:
    - Col 1: Min-Area Solution (Phase 2) - fit plot with shading and textbox
    - Col 2: Pareto Overlay with convergence rounds for upper bound area
    - Col 3: Common-Area Solution (Phase 3) - fit plot with shading and textbox

    This is the key visualization showing the trade-off between individual min-area
    and common/shared area solutions.

    If refine_plausible_configs is True, feasible points (below budget) are colored differently.
    """
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = phase3_output["history"]
    initial_solutions = phase3_output.get("initial_solutions", {})
    final_solutions = phase3_output["final_solutions"]
    N_train = phase3_output.get("N_train", 2048)

    activation_names = list(all_results_by_activation.keys())
    num_activations = len(activation_names)

    # Compute initial and final upper bounds
    initial_upper_bound = history[0]["upper_bound"]
    initial_upper_bound_area = history[0]["upper_bound_area"]
    final_upper_bound = phase3_output["final_upper_bound"]
    final_upper_bound_area = phase3_output["final_upper_bound_area"]

    # Flatten the paper-facing comparison figure to a 2:1 canvas.
    # With the default 4-activation setup this becomes 24x12 instead of 24x24.
    fig_width = 24
    fig_height = 4 * num_activations
    fig, axes = plt.subplots(num_activations, 3, figsize=(fig_width, fig_height))
    if num_activations == 1:
        axes = axes.reshape(1, -1)

    # Build global degree->color mapping
    all_degrees = set()
    for res in all_results_by_activation.values():
        degrees = res.get("degrees_final", [])
        for d in degrees:
            all_degrees.add(int(d))
    for sol in final_solutions.values():
        for d in sol.get("degrees", []):
            all_degrees.add(int(d))

    degrees_sorted = sorted(all_degrees) if all_degrees else [0]
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(10, len(degrees_sorted)))
    deg2color = {d: cmap(i % cmap.N) for i, d in enumerate(degrees_sorted)}
    degree_patches = [
        Patch(facecolor=deg2color[d], edgecolor="none", alpha=0.18, label=f"deg={d}")
        for d in degrees_sorted
    ]

    def draw_fit_panel(ax, *, x_vals, y_target, y_fit_vals, breakpoints, degrees,
                       title, fit_label, fit_color, textbox_lines):
        """Draw one fit panel with degree shading, breakpoints, and summary textbox."""
        ax.plot(x_vals, y_target, "k-", linewidth=2, alpha=0.7, label="Target")
        ax.plot(x_vals, y_fit_vals, fit_color, linewidth=2, alpha=0.85, label=fit_label)

        num_segments_local = max(0, len(breakpoints) - 1)
        for j in range(num_segments_local):
            dj = int(degrees[j]) if j < len(degrees) else 0
            ax.axvspan(
                breakpoints[j], breakpoints[j + 1],
                color=deg2color.get(dj, (0, 0, 0, 1)),
                alpha=0.18,
                linewidth=0
            )

        for bx in breakpoints:
            ax.axvline(bx, linewidth=0.8, alpha=0.25)

        y_fit_bp = np.interp(breakpoints, x_vals, y_fit_vals)
        ax.scatter(breakpoints, y_fit_bp, color='red', s=30, zorder=5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02, 0.98,
            "\n".join(textbox_lines),
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", alpha=0.15),
        )

        handles, _labels = ax.get_legend_handles_labels()
        ax.legend(handles=degree_patches + handles, fontsize=7, loc="best", ncol=2)

    for idx, act_name in enumerate(activation_names):
        res = all_results_by_activation[act_name]
        x = res["x_train"]
        y_true = res["y_train"]
        b_final = np.asarray(res["breakpoints_x_final"])

        # Phase 2 solution (min-area)
        phase2_sol = initial_solutions.get(act_name, {})
        phase2_degrees = phase2_sol.get("degrees", res.get("degrees_final", []))
        phase2_area = phase2_sol.get("area", res.get("total_area", None))
        phase2_mse = phase2_sol.get("mse", res.get("final_train_mse", None))  # Final Phase 2 MSE
        phase2_mse_before_cd = res.get("mse_after_dp", None)  # DP-stage MSE (fallback only)
        phase2_counts = phase2_sol.get("counts", [])

        # Phase 3 solution (common-area, CD-refined)
        phase3_sol = final_solutions.get(act_name, {})
        phase3_degrees = phase3_sol.get("degrees", [])
        phase3_area = phase3_sol.get("area", None)
        phase3_mse = phase3_sol.get("mse", None)
        phase3_mse_before_cd = phase3_sol.get("mse_before_cd", None)  # MSE before CD refinement
        phase3_counts = phase3_sol.get("counts", [])

        # Get y_fit from Phase 2
        y_fit_phase2 = res["y_train_fit"]

        # Get y_fit from Phase 3 (CD-refined, stored in final_solutions)
        y_fit_phase3 = phase3_sol.get("y_fit", None)
        if y_fit_phase3 is None:
            # Fallback: use Phase 2 fit if Phase 3 didn't have a different fit
            y_fit_phase3 = y_fit_phase2

        # Get Phase 3 breakpoints (may differ from Phase 2 after CD refinement)
        phase3_breakpoints = phase3_sol.get("breakpoints", b_final)
        if phase3_breakpoints is None:
            phase3_breakpoints = b_final
        phase3_breakpoints = np.asarray(phase3_breakpoints)

        # ============================================================
        # Col 0: Min-Area Solution (Phase 2)
        # ============================================================
        ax0 = axes[idx, 0]
        deg_counts_str = '/'.join(str(c) for c in phase2_counts) if phase2_counts else "N/A"
        textbox_lines_phase2 = [
            f"Area: {phase2_area:.2f}" if phase2_area else "Area: N/A",
            f"MSE: {phase2_mse:.2e}" if phase2_mse else "MSE: N/A",
            f"Deg counts: {deg_counts_str}",
        ]
        draw_fit_panel(
            ax0,
            x_vals=x,
            y_target=y_true,
            y_fit_vals=y_fit_phase2,
            breakpoints=b_final,
            degrees=phase2_degrees,
            title=f"{act_name}: Min-Area Solution (Phase 2)",
            fit_label="Min-Area Fit",
            fit_color="b--",
            textbox_lines=textbox_lines_phase2,
        )

        # ============================================================
        # Col 1: Pareto Overlay with Convergence Rounds
        # ============================================================
        ax1 = axes[idx, 1]

        # Use Pareto frontier for the main curve (guarantees monotonicity)
        # Update MSE values from pareto_feasible_configs if available (may be refined after 2nd CD)
        dp_frontier = res.get("dp_frontier_points", [])
        pareto_feasible = res.get("pareto_feasible_configs", [])

        # Build lookup from degrees -> config in pareto_feasible (has potentially updated MSE)
        pareto_lookup = {tuple(cfg.get("degrees", [])): cfg for cfg in pareto_feasible if cfg.get("degrees")}

        # Update frontier points with refined MSE values where available
        frontier = []
        for p in dp_frontier:
            p_degrees = tuple(p.get("degrees", []))
            if p_degrees and p_degrees in pareto_lookup:
                # Use the MSE from pareto_feasible_configs (may be refined)
                updated_p = p.copy()
                updated_p["mse"] = pareto_lookup[p_degrees]["mse"]
                frontier.append(updated_p)
            else:
                frontier.append(p)

        if not frontier:
            frontier = pareto_feasible

        if frontier:
            frontier_sorted = sorted(frontier, key=lambda t: float(t.get("area", 0) or 0))

            # Collect all points with feasibility info
            xs_feasible, ys_feasible = [], []
            xs_nonfeasible, ys_nonfeasible = [], []
            xs_all, ys_all = [], []
            budget_mse = max_error_budget

            for t in frontier_sorted:
                a = t.get("area", None)
                mm = t.get("mse", None)
                if a is not None and mm is not None and np.isfinite(float(a)):
                    mse_val = float(mm)
                    xs_all.append(float(a))
                    ys_all.append(mse_val)

                    # Use is_feasible flag if available, otherwise fall back to budget comparison
                    is_feas = t.get("is_feasible", None)
                    if is_feas is None and budget_mse is not None:
                        is_feas = mse_val <= budget_mse

                    if is_feas:
                        xs_feasible.append(float(a))
                        ys_feasible.append(mse_val)
                    else:
                        xs_nonfeasible.append(float(a))
                        ys_nonfeasible.append(mse_val)

            if xs_all:
                # Always show feasible vs non-feasible split
                # Plot non-feasible points (above budget) in lighter color
                if len(xs_nonfeasible) > 0:
                    ax1.plot(xs_nonfeasible, ys_nonfeasible, marker='o', linestyle='none',
                            markersize=5, alpha=0.5, color='C0', label='Above budget')

                # Plot feasible points (below budget) in brighter color
                if len(xs_feasible) > 0:
                    ax1.plot(xs_feasible, ys_feasible, marker='s', linestyle='none',
                            markersize=6, alpha=0.9, color='C1', label='Below budget')

                # Connect with line
                ax1.plot(xs_all, ys_all, linestyle='-', linewidth=1.5, alpha=0.4, color='gray')

        # Mark Phase 2 (min-area) solution as a single regular data point
        phase2_plot_mse = phase2_mse if phase2_mse is not None else phase2_mse_before_cd
        if phase2_area is not None and phase2_plot_mse is not None:
            ax1.scatter([phase2_area], [phase2_plot_mse], marker='o', s=80, color='blue',
                       edgecolors='black', linewidths=1, zorder=10,
                       label=f'Min-Area (Phase 2, MSE={phase2_plot_mse:.2e})')

        # Mark Phase 3 common-area solution as a single regular data point (post-CD)
        phase3_plot_mse = phase3_mse if phase3_mse is not None else phase3_mse_before_cd
        if phase3_area is not None and phase3_plot_mse is not None:
            ax1.scatter([phase3_area], [phase3_plot_mse], marker='o', s=80, color='green',
                       edgecolors='black', linewidths=1, zorder=10,
                       label=f'Common-Area (Phase 3 post-CD, MSE={phase3_plot_mse:.2e})')

        # Draw vertical lines for upper bound evolution (Phase 2 initial and Phase 3 optimal)
        areas_history = [h["upper_bound_area"] for h in history]
        unique_areas = sorted(set(areas_history), reverse=True)

        cmap_lines = plt.cm.Reds(np.linspace(0.3, 0.9, len(unique_areas)))
        for i, area in enumerate(unique_areas):
            round_idx = next(j for j, h in enumerate(history) if h["upper_bound_area"] == area)
            linestyle = '--' if i < len(unique_areas) - 1 else '-'
            linewidth = 1.5 if i < len(unique_areas) - 1 else 2.5
            if round_idx == 0:
                label = f"Phase 2: {area:.1f}"
            elif round_idx == len(history) - 1:
                label = f"Phase 3 Optimal: {area:.1f}"
            else:
                label = None
            ax1.axvline(x=area, color=cmap_lines[i], linestyle=linestyle, linewidth=linewidth,
                       alpha=0.8, label=label)

        # Add error threshold line (max_error_budget) if provided
        if max_error_budget is not None:
            ax1.axhline(y=max_error_budget, color='green', linestyle='--', linewidth=2,
                       label=f'Error Budget = {max_error_budget:.2e}')

        ax1.set_yscale('log')
        ax1.set_xlabel("Hardware Area Cost (μm²)", fontsize=10)
        ax1.set_ylabel("MSE (log scale)", fontsize=10)
        nodes_explored = phase3_output.get('nodes_explored', 0)
        ax1.set_title(f"{act_name}: Pareto + Best-First ({nodes_explored} explored)")
        ax1.grid(True, which='both', linestyle='--', alpha=0.4)
        ax1.legend(fontsize=7, loc='upper right')

        # ============================================================
        # Col 2: Common-Area Solution (Phase 3 post-CD)
        # ============================================================
        ax2 = axes[idx, 2]
        deg_counts_str = '/'.join(str(c) for c in phase3_counts) if phase3_counts else "N/A"
        textbox_lines_phase3 = [
            f"Area: {phase3_area:.2f}" if phase3_area else "Area: N/A",
            f"MSE (post-CD): {phase3_mse:.2e}" if phase3_mse else "MSE (post-CD): N/A",
            f"Deg counts: {deg_counts_str}",
            f"Upper Bound: {'/'.join(str(c) for c in final_upper_bound)}",
            f"UB Area: {final_upper_bound_area:.2f}",
        ]
        draw_fit_panel(
            ax2,
            x_vals=x,
            y_target=y_true,
            y_fit_vals=y_fit_phase3,
            breakpoints=phase3_breakpoints,
            degrees=phase3_degrees,
            title=f"{act_name}: Common-Area Solution (Phase 3 post-CD)",
            fit_label="Common-Area Fit",
            fit_color="g--",
            textbox_lines=textbox_lines_phase3,
        )

    # Suptitle
    area_reduction = (1 - final_upper_bound_area / initial_upper_bound_area) * 100 if initial_upper_bound_area > 0 else 0
    plt.suptitle(
        f"Min-Area vs Common-Area Comparison | {num_segments} segments | "
        f"UB: {'/'.join(str(c) for c in initial_upper_bound)} → {'/'.join(str(c) for c in final_upper_bound)} "
        f"(Area: {initial_upper_bound_area:.1f} → {final_upper_bound_area:.1f}, -{area_reduction:.1f}%)",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.995])

    save_path = out_dir / f"phase3_comparison_{num_segments}seg.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(save_path)


def viz_phase3_table_and_plot(phase3_output, out_dir, num_segments):
    """
    Create a table and plot comparing Phase 2 (min-area) vs Phase 3 (common-area) solutions.

    Table structure:
    - Initial Phase 2 rows (one row per function)
    - Optional search-trace rows in actual exploration order (Function="__SEARCH__")
    - Any additional Phase 3 history rows (one row per function per recorded round)
    - Final post-CD rows (one row per function)

    Plot:
    - X-axis: Hardware Area Cost (μm²)
    - Y-axis: MSE (log scale)
    - Min-area solutions with one color scheme + vertical line for initial upper bound
    - Common-area solutions with same color per function + vertical line for final upper bound
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = phase3_output.get("history", [])
    initial_solutions = phase3_output.get("initial_solutions", {})
    final_solutions = phase3_output.get("final_solutions", {})
    algorithm = phase3_output.get("algorithm", "unknown")
    search_trace = phase3_output.get("search_trace", []) or []
    total_budgets = phase3_output.get("total_budgets", None)

    if not initial_solutions or not final_solutions or len(history) == 0:
        log_file_only("[Phase 3] No data available for comparison table/plot")
        return None, None

    # Get initial and final upper bound areas
    initial_ub_area = history[0]["upper_bound_area"]
    final_ub_area = phase3_output["final_upper_bound_area"]
    final_ub_counts = phase3_output["final_upper_bound"]
    initial_ub_counts = history[0]["upper_bound"]

    activation_names = list(initial_solutions.keys())

    # =========================================================================
    # Create the comparison table (CSV) - include ALL rounds
    # =========================================================================
    table_rows = []

    def fmt_counts(vals):
        if not isinstance(vals, (list, tuple)):
            return "N/A"
        return "/".join(str(int(v)) for v in vals)

    def fmt_budget_list(budgets):
        if not isinstance(budgets, (list, tuple)):
            return "N/A"
        parts = []
        for b in budgets:
            if isinstance(b, (list, tuple)):
                parts.append("/".join(str(int(v)) for v in b))
        return "; ".join(parts) if parts else "N/A"

    def normalize_action(action):
        action_s = str(action) if action is not None else ""
        return action_s.replace(" explored", "")

    def append_solution_rows(round_num, action, ub_area, ub_counts, solutions):
        for act_name in activation_names:
            sol = solutions.get(act_name, {})
            mse = sol.get("mse")
            area = sol.get("area")
            counts = sol.get("counts", [])
            degrees = sol.get("degrees", [])
            table_rows.append({
                "Round": round_num,
                "Action": normalize_action(action),
                "Function": act_name,
                "MSE": mse,
                "Individual_Area": area,
                "Upper_Bound_Area": ub_area,
                "Upper_Bound_Counts": fmt_counts(ub_counts),
                "Degree_Counts": fmt_counts(counts),
                "Degrees": str(degrees) if degrees else "N/A",
                "Expanded_Node_Configs": "N/A",
            })

    # Initial rows first
    initial_h = history[0]
    append_solution_rows(
        round_num=initial_h.get("round", 0),
        action=initial_h.get("action", "initial"),
        ub_area=initial_h.get("upper_bound_area", np.nan),
        ub_counts=initial_h.get("upper_bound", []),
        solutions=initial_h.get("solutions", {}),
    )

    # Search rows in algorithm exploration order
    if algorithm in {"bestfirst", "tree", "iterative"} and search_trace:
        trace_total = len(search_trace)
        trace_denominator = int(total_budgets) if total_budgets else trace_total
        for i, t in enumerate(search_trace, start=1):
            if "end_upper_bound" in t:
                ub_counts = t.get("end_upper_bound")
            elif "upper_bound" in t:
                ub_counts = t.get("upper_bound")
            else:
                ub_counts = t.get("budget")

            if "end_upper_bound_area" in t:
                ub_area = t.get("end_upper_bound_area")
            elif "upper_bound_area" in t:
                ub_area = t.get("upper_bound_area")
            else:
                ub_area = t.get("area")

            table_rows.append({
                "Round": i,
                "Action": f"{algorithm} ({i}/{trace_denominator})",
                "Function": "__SEARCH__",
                "MSE": np.nan,
                "Individual_Area": np.nan,
                "Upper_Bound_Area": ub_area if ub_area is not None else np.nan,
                "Upper_Bound_Counts": fmt_counts(ub_counts),
                "Degree_Counts": "N/A",
                "Degrees": "N/A",
                "Expanded_Node_Configs": fmt_budget_list(t.get("expanded_neighbor_configs")),
            })

    # Additional history rounds (skip initial because already written)
    for h in history[1:]:
        append_solution_rows(
            round_num=h.get("round", np.nan),
            action=h.get("action", ""),
            ub_area=h.get("upper_bound_area", np.nan),
            ub_counts=h.get("upper_bound", []),
            solutions=h.get("solutions", {}),
        )

    # Final post-CD rows
    final_round_num = history[-1]["round"] + 1 if history else 0
    append_solution_rows(
        round_num=final_round_num,
        action="final (post-CD refinement)",
        ub_area=final_ub_area,
        ub_counts=final_ub_counts,
        solutions=final_solutions,
    )

    # Save table as CSV
    table_df = pd.DataFrame(table_rows)
    table_csv_path = out_dir / f"phase2_vs_phase3_comparison_{num_segments}seg.csv"
    table_df.to_csv(table_csv_path, index=False)

    # =========================================================================
    # Create the comparison plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for functions (consistent across phases)
    num_funcs = len(activation_names)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(10, num_funcs))
    func_colors = {name: cmap(i % cmap.N) for i, name in enumerate(activation_names)}

    # Collect data for plotting
    min_area_data = []  # (name, area, mse)
    common_area_data = []  # (name, area, mse)

    for act_name in activation_names:
        # Min-area (Phase 2)
        sol2 = initial_solutions.get(act_name, {})
        if sol2.get("area") is not None and sol2.get("mse") is not None:
            min_area_data.append((act_name, sol2["area"], sol2["mse"]))

        # Common-area (Phase 3)
        sol3 = final_solutions.get(act_name, {})
        if sol3.get("area") is not None and sol3.get("mse") is not None:
            common_area_data.append((act_name, sol3["area"], sol3["mse"]))

    # Plot Min-Area solutions (circles)
    for name, area, mse in min_area_data:
        ax.scatter(area, mse, c=[func_colors[name]], marker='o', s=100,
                  edgecolors='black', linewidths=1, zorder=5)

    # Plot Common-Area solutions (stars) and draw connecting lines
    for name, area3, mse3 in common_area_data:
        ax.scatter(area3, mse3, c=[func_colors[name]], marker='*', s=200,
                  edgecolors='black', linewidths=1, zorder=6)

        # Find corresponding min-area point and draw connecting line
        for name2, area2, mse2 in min_area_data:
            if name2 == name:
                ax.plot([area2, area3], [mse2, mse3],
                       color=func_colors[name], linestyle='--', linewidth=1.5, alpha=0.6)
                break

    # Draw vertical lines for upper bound areas
    ax.axvline(initial_ub_area, color='royalblue', linestyle='--', linewidth=2.5, alpha=0.8,
              label=f'Min-Area Upper Bound: {initial_ub_area:.1f}')
    ax.axvline(final_ub_area, color='green', linestyle='-', linewidth=2.5, alpha=0.8,
              label=f'Common-Area Upper Bound: {final_ub_area:.1f}')

    # Shade the region of area reduction
    if final_ub_area < initial_ub_area:
        ax.axvspan(final_ub_area, initial_ub_area, alpha=0.1, color='green',
                  label=f'Area Reduction: {(1 - final_ub_area/initial_ub_area)*100:.1f}%')

    # Create legend entries for functions
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = []
    # Add marker legend
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                 markersize=10, markeredgecolor='black', label='Min-Area (Phase 2)'))
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                                 markersize=15, markeredgecolor='black', label='Common-Area (Phase 3)'))

    # Add function color legend
    for name in activation_names:
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                     markerfacecolor=func_colors[name], markersize=8,
                                     label=name))

    # Add vertical line legend entries (already in ax.legend via label)
    ax.set_yscale('log')
    ax.set_xlabel("Hardware Area Cost (μm²)", fontsize=12)
    ax.set_ylabel("MSE (log scale)", fontsize=12)

    # Compute area reduction
    area_reduction_pct = (1 - final_ub_area / initial_ub_area) * 100 if initial_ub_area > 0 else 0

    ax.set_title(
        f"Phase 2 (Min-Area) vs Phase 3 (Common-Area) Comparison | {num_segments} segments\n"
        f"Upper Bound: {'/'.join(str(c) for c in initial_ub_counts)} → {'/'.join(str(c) for c in final_ub_counts)} | "
        f"Area: {initial_ub_area:.1f} → {final_ub_area:.1f} (-{area_reduction_pct:.1f}%)",
        fontsize=12, fontweight='bold'
    )

    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # Combined legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_elements + handles, loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()

    plot_path = out_dir / f"phase2_vs_phase3_comparison_{num_segments}seg.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(table_csv_path), str(plot_path)


def io_save_phase3_summary(phase3_output, out_dir, num_segments, degree_max=3):
    """Save Phase 3 optimization summary to text file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = phase3_output["history"]
    final_solutions = phase3_output["final_solutions"]
    nodes_explored = phase3_output.get('nodes_explored', 0)
    total_budgets = phase3_output.get('total_budgets', 0)

    # Save summary text
    summary_lines = [
        f"Phase 3 Area-Error Optimization Summary (Best-First Search)",
        f"=" * 50,
        f"Segments: {num_segments}",
        f"Degree Max: {degree_max}",
        f"",
        f"Initial Upper Bound (Phase 2): {history[0]['upper_bound']}",
        f"Initial Upper Bound Area: {history[0]['upper_bound_area']:.2f}",
        f"",
        f"Final Upper Bound (Optimal): {phase3_output['final_upper_bound']}",
        f"Final Upper Bound Area: {phase3_output['final_upper_bound_area']:.2f}",
        f"",
        f"Area Reduction: {(1 - phase3_output['final_upper_bound_area']/history[0]['upper_bound_area'])*100:.2f}%",
        f"Nodes Explored: {nodes_explored}/{total_budgets}",
        f"",
        f"Final Solutions:",
        f"-" * 30,
    ]

    for act_name, sol in final_solutions.items():
        mse_val = sol.get('mse', None)
        mse_str = f"{mse_val:.4e}" if mse_val is not None else "N/A"
        summary_lines.append(f"  {act_name}:")
        summary_lines.append(f"    Degrees: {sol['degrees']}")
        summary_lines.append(f"    Counts (d0/d1/d2/d3): {'/'.join(str(c) for c in sol['counts'])}")
        summary_lines.append(f"    Area: {sol['area']:.2f}")
        summary_lines.append(f"    MSE: {mse_str}")

    summary_txt = out_dir / f"phase3_summary_{num_segments}seg.txt"
    with open(summary_txt, "w") as f:
        f.write("\n".join(summary_lines))

    return str(summary_txt)

def viz_aggregate_phase3_summary(phase3_outputs_by_segments, out_dir, degree_max, seg_min, seg_max):
    """
    Create aggregate visualization showing Phase 3 optimization results across all segment counts.

    This creates a multi-panel plot showing:
    1. Upper bound area vs segments (initial vs final)
    2. Area reduction percentage vs segments
    3. Final degree count distribution per activation across segments
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = sorted(phase3_outputs_by_segments.keys())

    if len(segments) == 0:
        return None

    # Extract data
    initial_areas = []
    final_areas = []
    reductions = []

    for seg in segments:
        res = phase3_outputs_by_segments[seg]
        initial_area = res["history"][0]["upper_bound_area"]
        final_area = res["final_upper_bound_area"]
        initial_areas.append(initial_area)
        final_areas.append(final_area)
        reductions.append((1 - final_area / initial_area) * 100 if initial_area > 0 else 0)

    # Get activation names from first result
    first_res = phase3_outputs_by_segments[segments[0]]
    activation_names = list(first_res["final_solutions"].keys())

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ============================================================
    # Top-left: Upper bound area vs segments
    # ============================================================
    ax0 = axes[0, 0]
    ax0.plot(segments, initial_areas, 'b-o', linewidth=2, markersize=6, label='Initial (Phase 2)', alpha=0.7)
    ax0.plot(segments, final_areas, 'g-s', linewidth=2, markersize=6, label='Final (Phase 3)', alpha=0.9)
    ax0.fill_between(segments, initial_areas, final_areas, alpha=0.2, color='green')
    ax0.set_xlabel("Number of Segments", fontsize=12)
    ax0.set_ylabel("Upper Bound Area", fontsize=12)
    ax0.set_title("Phase 3: Upper Bound Area Reduction", fontsize=14)
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)
    ax0.set_xticks(segments)

    # ============================================================
    # Top-right: Area reduction percentage vs segments
    # ============================================================
    ax1 = axes[0, 1]
    bars = ax1.bar(segments, reductions, color='green', alpha=0.7, edgecolor='darkgreen')
    ax1.set_xlabel("Number of Segments", fontsize=12)
    ax1.set_ylabel("Area Reduction (%)", fontsize=12)
    ax1.set_title("Phase 3: Area Reduction by Segment Count", fontsize=14)
    ax1.set_xticks(segments)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax1.annotate(f'{reduction:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # ============================================================
    # Bottom-left: Final upper bound degree distribution vs segments
    # ============================================================
    ax2 = axes[1, 0]

    deg_labels = ['deg0 (const)', 'deg1 (linear)', 'deg2 (quad)', 'deg3 (cubic)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Stack upper bounds by degree type
    deg_data = {i: [] for i in range(degree_max + 1)}
    for seg in segments:
        res = phase3_outputs_by_segments[seg]
        ub = res["final_upper_bound"]
        for i in range(degree_max + 1):
            deg_data[i].append(ub[i] if i < len(ub) else 0)

    x = np.arange(len(segments))
    width = 0.8
    bottom = np.zeros(len(segments))

    for deg_idx in range(degree_max + 1):
        ax2.bar(x, deg_data[deg_idx], width, bottom=bottom,
               label=deg_labels[deg_idx], color=colors[deg_idx], alpha=0.8)
        bottom = bottom + np.array(deg_data[deg_idx])

    ax2.set_xlabel("Number of Segments", fontsize=12)
    ax2.set_ylabel("Upper Bound (count per degree)", fontsize=12)
    ax2.set_title("Final Upper Bound Degree Distribution", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(segments)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # Bottom-right: Per-activation area comparison
    # ============================================================
    ax3 = axes[1, 1]

    # For each activation, plot its final area vs segments
    act_colors = plt.cm.tab10(np.linspace(0, 1, len(activation_names)))

    for idx, act_name in enumerate(activation_names):
        act_areas = []
        for seg in segments:
            res = phase3_outputs_by_segments[seg]
            sol = res["final_solutions"].get(act_name, {})
            act_areas.append(sol.get("area", 0) or 0)

        ax3.plot(segments, act_areas, '-o', linewidth=2, markersize=5,
                color=act_colors[idx], label=act_name, alpha=0.8)

    # Also plot the upper bound
    ax3.plot(segments, final_areas, 'k--', linewidth=2.5, label='Upper Bound', alpha=0.7)

    ax3.set_xlabel("Number of Segments", fontsize=12)
    ax3.set_ylabel("Hardware Area", fontsize=12)
    ax3.set_title("Per-Activation Final Area vs Segments", fontsize=14)
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(segments)

    plt.suptitle(f"Aggregate Phase 3 Area-Error Optimization Summary\n(degree_max={degree_max}, segments={seg_min}-{seg_max})",
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = out_dir / f"aggregate_phase3_summary_seg{seg_min}-{seg_max}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    log_file_only(f"Saved aggregate Phase 3 plot: {save_path}")
    return str(save_path)


def io_save_aggregate_phase3_csv(phase3_outputs_by_segments, out_dir, degree_max):
    """
    Save aggregate Phase 3 results to CSV files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = sorted(phase3_outputs_by_segments.keys())

    # Summary CSV
    summary_rows = []
    for seg in segments:
        res = phase3_outputs_by_segments[seg]
        history = res["history"]

        row = {
            "segments": seg,
            "initial_area": history[0]["upper_bound_area"],
            "final_area": res["final_upper_bound_area"],
            "area_reduction_pct": (1 - res["final_upper_bound_area"] / history[0]["upper_bound_area"]) * 100,
            "nodes_explored": res.get("nodes_explored", 0),
            "total_budgets": res.get("total_budgets", 0),
        }

        # Add initial and final upper bounds
        for i, ub in enumerate(history[0]["upper_bound"]):
            row[f"initial_ub_deg{i}"] = ub
        for i, ub in enumerate(res["final_upper_bound"]):
            row[f"final_ub_deg{i}"] = ub

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"aggregate_phase3_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Per-activation solutions CSV
    solution_rows = []
    for seg in segments:
        res = phase3_outputs_by_segments[seg]
        for act_name, sol in res["final_solutions"].items():
            row = {
                "segments": seg,
                "activation": act_name,
                "area": sol.get("area", None),
                "mse": sol.get("mse", None),
                "degrees": str(sol.get("degrees", [])),
            }
            for i, c in enumerate(sol.get("counts", [])):
                row[f"deg{i}_count"] = c
            solution_rows.append(row)

    solutions_df = pd.DataFrame(solution_rows)
    solutions_csv = out_dir / f"aggregate_phase3_solutions.csv"
    solutions_df.to_csv(solutions_csv, index=False)

    log_file_only(f"Saved aggregate Phase 3 CSVs: {summary_csv}, {solutions_csv}")
    return str(summary_csv), str(solutions_csv)


def helper_prepare_segment_output_dirs(runs_root: Path, num_segments: int):
    """
    Create and return per-segment output directories and budget_dirs mapping.
    """
    run_root = runs_root / f"seg_{num_segments:02d}"
    plots_dir = run_root / "plots"
    min_area_config_dir = run_root / "min_area_config"
    pareto_all_configs_dir = run_root / "pareto_optimal_all"
    pareto_configs_dir = run_root / "pareto_optimal_feasible"
    common_area_config_dir = run_root / "common_area_config"
    inference_dir = run_root / "inference"
    for d in [plots_dir, min_area_config_dir, pareto_all_configs_dir, pareto_configs_dir, common_area_config_dir, inference_dir]:
        d.mkdir(parents=True, exist_ok=True)

    budget_dirs = {
        "root": run_root,
        "plots": plots_dir,
        "min_area_config": min_area_config_dir,
        "pareto_all_configs": pareto_all_configs_dir,
        "pareto_configs": pareto_configs_dir,
        "common_area_config": common_area_config_dir,
        "inference": inference_dir,
    }
    return run_root, budget_dirs


def helper_compute_phase2_upper_bound(all_results, degree_max):
    """
    Compute phase-2 element-wise upper bound on degree counts and corresponding area.
    """
    max_degree_counts = [0] * (degree_max + 1)
    for res in all_results:
        degrees_final = res.get("degrees_final", [])
        if not degrees_final:
            continue
        counts = helper_degree_type_counts(degrees_final, degree_max)
        for i in range(len(max_degree_counts)):
            if i < len(counts):
                max_degree_counts[i] = max(max_degree_counts[i], counts[i])

    combined_c_counts = helper_degree_counts_to_c_counts(max_degree_counts)
    phase2_upper_bound_area = helper_hardware_area(*combined_c_counts)
    return max_degree_counts, float(phase2_upper_bound_area)


def helper_normalize_segment_schedule(seg_min, seg_max, segment_budgets):
    """
    Build a validated segment schedule and return:
      (segment_schedule, seg_min_effective, seg_max_effective)
    """
    if segment_budgets is not None:
        cleaned_budgets = []
        for v in segment_budgets:
            vi = int(v)
            if vi < 1:
                raise ValueError(f"segment_budgets must contain positive integers, got {vi}")
            cleaned_budgets.append(vi)

        segment_schedule = sorted(set(cleaned_budgets))
        if not segment_schedule:
            raise ValueError("segment_budgets cannot be empty")
        return segment_schedule, int(segment_schedule[0]), int(segment_schedule[-1])

    if seg_min > seg_max:
        raise ValueError(f"seg_min ({seg_min}) cannot be greater than seg_max ({seg_max})")

    segment_schedule = list(range(int(seg_min), int(seg_max) + 1))
    return segment_schedule, int(seg_min), int(seg_max)


def helper_merge_sweep_summary(csv_path, new_df, append_existing):
    """
    Merge newly produced sweep rows with an existing summary CSV.
    """
    if (not append_existing) or (not csv_path.exists()):
        return new_df

    try:
        prev_df = pd.read_csv(csv_path)
    except Exception:
        prev_df = pd.DataFrame()

    if len(prev_df) == 0:
        return new_df

    df = pd.concat([prev_df, new_df], ignore_index=True)
    dedup_cols = [c for c in ["Activation", "SegmentsRequested"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
    return df


def helper_infer_segment_range(df, seg_col, fallback_min, fallback_max):
    """
    Infer [min, max] segment range from dataframe content for plot titles/filenames.
    """
    if seg_col is None or seg_col not in df.columns or len(df) == 0:
        return int(fallback_min), int(fallback_max)

    values = pd.to_numeric(df[seg_col], errors="coerce")
    values = values[np.isfinite(values.values)]
    if len(values) == 0:
        return int(fallback_min), int(fallback_max)

    return int(np.floor(values.min())), int(np.ceil(values.max()))


def helper_build_runtime_matrix_df(summary_df):
    """
    Build runtime matrix with:
      - one row per segment budget
      - per-activation Stage1/Stage2/Total runtime columns
      - segment-level Stage1/Stage2/Total runtime columns
    """
    if not isinstance(summary_df, pd.DataFrame) or len(summary_df) == 0:
        return pd.DataFrame()
    if "SegmentsRequested" not in summary_df.columns or "Activation" not in summary_df.columns:
        return pd.DataFrame()

    df = summary_df.copy()
    df["_seg_num"] = pd.to_numeric(df["SegmentsRequested"], errors="coerce")
    df = df[np.isfinite(df["_seg_num"].values)]
    if len(df) == 0:
        return pd.DataFrame()
    df["_seg_num"] = df["_seg_num"].round().astype(int)

    activations_seen = [str(a) for a in df["Activation"].dropna().astype(str).unique().tolist()]
    act_order = [a for a in ACTIVATION_FUNCTIONS.keys() if a in activations_seen]
    act_order.extend(sorted(a for a in activations_seen if a not in act_order))

    def last_numeric(sub_df, col_name):
        if col_name not in sub_df.columns or len(sub_df) == 0:
            return np.nan
        vals = pd.to_numeric(sub_df[col_name], errors="coerce").dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.iloc[-1])

    runtime_rows = []
    for seg in sorted(df["_seg_num"].unique().tolist()):
        seg_df = df[df["_seg_num"] == int(seg)]
        row = {
            "SegmentBudget": int(seg),
            "SegmentStage1Sec": last_numeric(seg_df, "Stage1RuntimeSec"),
            "SegmentStage2Sec": last_numeric(seg_df, "Stage2RuntimeSec"),
            "SegmentTotalSec": last_numeric(seg_df, "SegmentRuntimeSec"),
            "SweepWallClockSec": np.nan,
        }
        for act_name in act_order:
            act_df = seg_df[seg_df["Activation"].astype(str) == act_name]
            row[f"{act_name}_Stage1Sec"] = last_numeric(act_df, "Phase1RuntimeSec")
            row[f"{act_name}_Stage2Sec"] = last_numeric(act_df, "Phase2RuntimeSec")
            row[f"{act_name}_TotalSec"] = last_numeric(act_df, "ActivationRuntimeSec")
        runtime_rows.append(row)

    runtime_df = pd.DataFrame(runtime_rows)
    if len(runtime_df) == 0:
        return runtime_df

    ordered_cols = ["SegmentBudget"]
    for act_name in act_order:
        ordered_cols.extend(
            [
                f"{act_name}_Stage1Sec",
                f"{act_name}_Stage2Sec",
                f"{act_name}_TotalSec",
            ]
        )
    ordered_cols.extend(
        [
            "SegmentStage1Sec",
            "SegmentStage2Sec",
            "SegmentTotalSec",
            "SweepWallClockSec",
        ]
    )
    remaining_cols = [c for c in runtime_df.columns if c not in ordered_cols]
    runtime_df = runtime_df.reindex(columns=ordered_cols + remaining_cols)
    runtime_df = runtime_df.sort_values("SegmentBudget").reset_index(drop=True)
    return runtime_df


def io_save_runtime_matrix_csv(summary_df, out_root, append_existing=False, run_elapsed_sec=None):
    """
    Save runtime matrix CSV to <out_root>/runtime/runtime_by_segment.csv.

    Includes a final row "ALL_TOTAL" with summed columns and cumulative wall-clock time.
    """
    runtime_df_new = helper_build_runtime_matrix_df(summary_df)
    if len(runtime_df_new) == 0:
        return None

    runtime_dir = Path(out_root) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_csv_path = runtime_dir / "runtime_by_segment.csv"

    prev_wallclock_total = 0.0
    merged_df = runtime_df_new.copy()
    if append_existing and runtime_csv_path.exists():
        try:
            prev_df = pd.read_csv(runtime_csv_path)
        except Exception:
            prev_df = pd.DataFrame()

        if len(prev_df) > 0:
            if "SegmentBudget" in prev_df.columns:
                seg_text = prev_df["SegmentBudget"].astype(str)
                total_mask = seg_text == "ALL_TOTAL"
                if total_mask.any() and "SweepWallClockSec" in prev_df.columns:
                    prev_vals = pd.to_numeric(
                        prev_df.loc[total_mask, "SweepWallClockSec"], errors="coerce"
                    ).dropna()
                    if len(prev_vals) > 0:
                        prev_wallclock_total = float(prev_vals.iloc[-1])
                prev_df = prev_df.loc[~total_mask].copy()
            merged_df = pd.concat([prev_df, runtime_df_new], ignore_index=True, sort=False)

    if "SegmentBudget" in merged_df.columns:
        seg_vals = pd.to_numeric(merged_df["SegmentBudget"], errors="coerce")
        valid_mask = np.isfinite(seg_vals.values)
        merged_df = merged_df.loc[valid_mask].copy()
        merged_df["SegmentBudget"] = seg_vals.loc[valid_mask].round().astype(int).values
        merged_df = merged_df.drop_duplicates(subset=["SegmentBudget"], keep="last")
        merged_df = merged_df.sort_values("SegmentBudget").reset_index(drop=True)

    if "SweepWallClockSec" not in merged_df.columns:
        merged_df["SweepWallClockSec"] = np.nan
    merged_df["SweepWallClockSec"] = np.nan

    detected_acts = set()
    for col in merged_df.columns:
        for suffix in ("_Stage1Sec", "_Stage2Sec", "_TotalSec"):
            if col.endswith(suffix):
                prefix = col[: -len(suffix)]
                if prefix and not prefix.startswith("Segment") and prefix != "SweepWallClock":
                    detected_acts.add(prefix)
                break
    act_order = [a for a in ACTIVATION_FUNCTIONS.keys() if a in detected_acts]
    act_order.extend(sorted(a for a in detected_acts if a not in act_order))

    ordered_cols = ["SegmentBudget"]
    for act_name in act_order:
        ordered_cols.extend(
            [
                f"{act_name}_Stage1Sec",
                f"{act_name}_Stage2Sec",
                f"{act_name}_TotalSec",
            ]
        )
    ordered_cols.extend(
        [
            "SegmentStage1Sec",
            "SegmentStage2Sec",
            "SegmentTotalSec",
            "SweepWallClockSec",
        ]
    )
    ordered_cols.extend([c for c in merged_df.columns if c not in ordered_cols])
    merged_df = merged_df.reindex(columns=ordered_cols)

    total_row = {"SegmentBudget": "ALL_TOTAL"}
    numeric_cols = [c for c in merged_df.columns if c != "SegmentBudget"]
    for col in numeric_cols:
        vals = pd.to_numeric(merged_df[col], errors="coerce")
        total_row[col] = float(vals.sum(skipna=True)) if vals.notna().any() else np.nan

    this_run_elapsed = (
        float(run_elapsed_sec)
        if run_elapsed_sec is not None and np.isfinite(run_elapsed_sec)
        else 0.0
    )
    total_row["SweepWallClockSec"] = prev_wallclock_total + this_run_elapsed

    final_df = pd.concat([merged_df, pd.DataFrame([total_row])], ignore_index=True)
    final_df.to_csv(runtime_csv_path, index=False)
    log_file_only(f"Saved runtime matrix CSV: {runtime_csv_path}")
    return str(runtime_csv_path)


def helper_run_single_segment_budget(
    *,
    num_segments,
    runs_root,
    degree_max,
    n_train_samples,
    n_eval_samples,
    num_outer_iters,
    min_seg_points,
    lam,
    use_mixed_degrees,
    max_error_budget,
    refine_plausible_configs,
    phase3_algo,
    save_per_budget_overfit_plots,
):
    """
    Execute one segment budget end-to-end:
      - Stage 1: per-activation optimization (Phase 1 + Phase 2)
      - Stage 2: joint common-area optimization (Phase 3)
    """
    segment_start_t = time.perf_counter()
    num_breakpoints = num_segments + 1
    run_root, budget_dirs = helper_prepare_segment_output_dirs(runs_root, num_segments)
    plots_dir = budget_dirs["plots"]

    log_file_only("")
    log_file_only(f"{'=' * 80}")
    log_file_only(f"  SEGMENT COUNT = {num_segments}")
    log_file_only(f"{'=' * 80}")
    log_file_only("")
    log_file_only(f"{'=' * 60}")
    log_file_only("STAGE 1: Individual Optimization (Phase 1 + Phase 2)")
    log_file_only(f"{'=' * 60}")

    all_results = []
    segment_rows = []
    stage1_start_t = time.perf_counter()
    stage1_pbar = tqdm(
        ACTIVATION_FUNCTIONS.items(),
        desc=f"  seg={num_segments:02d} Stage 1",
        leave=False,
        ncols=120,
    )

    for act_name, act_fn in stage1_pbar:
        phase2_output_single = run_single_activation(
            activation_name=act_name,
            target_func=act_fn,
            num_breakpoints=num_breakpoints,
            degree=degree_max,
            n_train_samples=n_train_samples,
            n_eval_samples=n_eval_samples,
            num_outer_iters=num_outer_iters,
            min_seg_points=min_seg_points,
            lam=lam,
            verbose=False,
            out_dirs=budget_dirs,
            use_mixed_degrees=use_mixed_degrees,
            max_error_budget=max_error_budget,
            refine_plausible_configs=refine_plausible_configs,
        )

        infer_csv = budget_dirs["inference"] / f"{act_name}_trainfit_degmax{degree_max}.csv"
        io_save_inference_csv(phase2_output_single, infer_csv)

        all_results.append(phase2_output_single)
        timing = phase2_output_single.get("timing_seconds", {}) or {}
        t_phase1 = float(timing.get("phase1", 0.0) or 0.0)
        t_phase2 = float(timing.get("phase2_total", 0.0) or 0.0)
        t_phase2_dp = float(timing.get("phase2_dp", 0.0) or 0.0)
        t_phase2_refine = float(timing.get("phase2_refine", 0.0) or 0.0)
        t_act_total = float(timing.get("activation_total", 0.0) or 0.0)

        deg_counts = helper_degrees_to_degree_counts(phase2_output_single.get("degrees_final", []), degree_max)
        deg_counts_str = "/".join(str(c) for c in deg_counts)
        dp_frontier = phase2_output_single.get("dp_frontier_points", []) or []
        pareto_feasible_list = phase2_output_single.get("pareto_feasible_configs", []) or []
        area_val = phase2_output_single.get("total_area")
        area_str = f"{area_val:.2f}" if area_val is not None else "N/A"
        mse_val = phase2_output_single.get("final_train_mse")
        mse_str = f"{mse_val:.2e}" if mse_val is not None else "N/A"

        log_file_only("")
        log_file_only(
            f"[Stage 1] {act_name}: degrees={phase2_output_single.get('degrees_final', [])}, "
            f"counts={deg_counts_str}, area={area_str}, MSE={mse_str}"
        )
        log_file_only(
            f"[Stage 1] {act_name}: DP end-state configs: {len(dp_frontier)} total, "
            f"{len(pareto_feasible_list)} feasible"
        )
        log_file_only(
            f"[Timing][Stage 1] {act_name}: "
            f"Phase 1={t_phase1:.3f}s | "
            f"Phase 2={t_phase2:.3f}s (DP={t_phase2_dp:.3f}s, refine={t_phase2_refine:.3f}s) | "
            f"Activation total={t_act_total:.3f}s"
        )

        segment_rows.append(
            {
                "Activation": act_name,
                "SegmentsRequested": num_segments,
                "SegmentsFinal": phase2_output_single["num_segments"],
                "DegreeMax": degree_max,
                "TotalParameters": int(phase2_output_single["total_parameters"]),
                "TotalArea": float(phase2_output_single["total_area"])
                if phase2_output_single.get("total_area", None) is not None
                else np.nan,
                "TrainMSE": phase2_output_single["final_train_mse"],
                "EvalMSE": phase2_output_single["final_eval_mse"],
                "MixedEnabled": bool(use_mixed_degrees),
                "RefinePlausible": bool(refine_plausible_configs),
                "Phase3Algo": str(phase3_algo),
                "Counts_c0c1c2c3": str(phase2_output_single.get("dp_chosen_counts", None)),
                "Phase1RuntimeSec": t_phase1,
                "Phase2RuntimeSec": t_phase2,
                "Phase2DPRuntimeSec": t_phase2_dp,
                "Phase2RefineRuntimeSec": t_phase2_refine,
                "ActivationRuntimeSec": t_act_total,
            }
        )

    stage1_elapsed_sec = time.perf_counter() - stage1_start_t
    log_file_only(f"[Timing][Stage 1] Total={stage1_elapsed_sec:.3f}s")

    max_degree_counts, phase2_upper_bound_area = helper_compute_phase2_upper_bound(
        all_results=all_results,
        degree_max=degree_max,
    )

    stage1_counts_str = "/".join(str(c) for c in max_degree_counts)
    log_file_only("")
    log_file_only("[Stage 1] Upper Bound (from Phase 1 + Phase 2 outputs):")
    log_file_only(f"[Stage 1]   counts={stage1_counts_str}, area={phase2_upper_bound_area:.2f}")

    phase2_outputs_by_activation = {res["activation_name"]: res for res in all_results}

    log_file_only("")
    log_file_only(f"{'=' * 60}")
    log_file_only("STAGE 2: Joint Common-Area Optimization (Phase 3)")
    log_file_only(f"{'=' * 60}")

    stage2_start_t = time.perf_counter()
    stage2_phase3_opt_start_t = time.perf_counter()
    phase3_output = phase3_optimize_common_area(
        all_results_by_activation=phase2_outputs_by_activation,
        degree_max=degree_max,
        max_rounds=50,
        verbose=True,
        algorithm=phase3_algo,
    )
    stage2_phase3_opt_elapsed_sec = time.perf_counter() - stage2_phase3_opt_start_t
    stage2_post_start_t = time.perf_counter()

    phase3_upper_bound = phase3_output["final_upper_bound"]
    phase3_upper_bound_area = phase3_output["final_upper_bound_area"]
    phase3_counts_str = "/".join(str(c) for c in phase3_upper_bound)
    area_reduction_pct = (
        (1 - phase3_upper_bound_area / phase2_upper_bound_area) * 100
        if phase2_upper_bound_area > 0
        else 0.0
    )
    log_file_only("")
    log_file_only("=== Stage 2 Upper Bound (Phase 3 final) ===")
    log_file_only(f"  Stage 1 (Phase 1+2): counts={stage1_counts_str}, area={phase2_upper_bound_area:.2f}")
    log_file_only(f"  Stage 2 (Phase 3):   counts={phase3_counts_str}, area={phase3_upper_bound_area:.2f}")
    log_file_only(f"  Area reduction: {area_reduction_pct:.2f}%")

    area_opt_dir = run_root / "area_err_optimization"
    area_opt_dir.mkdir(parents=True, exist_ok=True)

    io_save_phase3_summary(phase3_output, area_opt_dir, num_segments, degree_max)
    viz_phase3_optimization(phase3_output, area_opt_dir, num_segments, degree_max)
    viz_phase3_pareto_overlay(
        phase3_output,
        phase2_outputs_by_activation,
        area_opt_dir,
        num_segments,
        degree_max,
        max_error_budget,
        refine_plausible_configs,
    )
    viz_phase3_comparison(
        phase3_output,
        phase2_outputs_by_activation,
        area_opt_dir,
        num_segments,
        degree_max,
        max_error_budget,
        refine_plausible_configs,
    )
    viz_phase3_table_and_plot(phase3_output, area_opt_dir, num_segments)

    log_file_only("")
    log_file_only("=== Stage 2 Final Solutions (Phase 3 post-CD) ===")
    for act_name, sol in phase3_output["final_solutions"].items():
        mse_str = f"{sol['mse']:.2e}" if sol.get("mse") is not None else "N/A"
        counts_str = "/".join(str(c) for c in sol["counts"]) if sol.get("counts") else "N/A"
        log_file_only(
            f"  {act_name}: degrees={sol['degrees']}, counts={counts_str}, area={sol['area']:.2f}, MSE={mse_str}"
        )

    common_area_config_dir = budget_dirs.get("common_area_config")
    if common_area_config_dir is not None:
        for act_name, sol in phase3_output["final_solutions"].items():
            if "coeffs" in sol and "breakpoints" in sol:
                io_save_config_csv(
                    activation_name=act_name,
                    num_breakpoints=len(sol["breakpoints"]),
                    degree_max=degree_max,
                    breakpoints_x=sol["breakpoints"],
                    coeffs=sol["coeffs"],
                    out_dir=common_area_config_dir,
                    degrees=sol["degrees"],
                )

    n_eval_phase3 = 4096
    rows_by_activation = {row["Activation"]: row for row in segment_rows}

    for act_name, row in rows_by_activation.items():
        sol = phase3_output["final_solutions"].get(act_name)
        if sol is None:
            continue

        if sol.get("area") is not None:
            row["TotalArea"] = float(sol["area"])
        if sol.get("mse") is not None:
            row["TrainMSE"] = float(sol["mse"])
        row["Counts_c0c1c2c3"] = str(sol.get("counts", None))

        if sol.get("degrees") is not None:
            row["TotalParameters"] = int(sum((d + 1) for d in sol["degrees"]))

        if sol.get("coeffs") is not None and sol.get("breakpoints") is not None:
            act_fn = ACTIVATION_FUNCTIONS.get(act_name)
            if act_fn is not None:
                domain_min, domain_max = helper_activation_domain(act_name)
                with torch.no_grad():
                    x_eval_p3 = torch.linspace(domain_min, domain_max, n_eval_phase3, device=device, dtype=torch.float32)
                    y_eval_p3 = act_fn(x_eval_p3)
                    bp_x_tensor = torch.from_numpy(np.asarray(sol["breakpoints"])).float().to(device)
                    coeffs_tensors = [
                        c.to(device)
                        if torch.is_tensor(c)
                        else torch.tensor(c, device=device, dtype=torch.float32)
                        for c in sol["coeffs"]
                    ]
                    y_eval_hat, _, _ = helper_predict_from_coeffs(
                        x_eval_p3, bp_x_tensor, coeffs_tensors, sol["degrees"]
                    )
                    row["EvalMSE"] = float(torch.mean((y_eval_hat - y_eval_p3) ** 2).item())

    if save_per_budget_overfit_plots:
        viz_all_activations(
            all_results,
            num_breakpoints=num_breakpoints,
            out_dir=plots_dir,
            upper_bound_area=phase3_upper_bound_area,
            upper_bound_degree_counts=phase3_upper_bound,
            max_error_budget=max_error_budget,
            refine_plausible_configs=refine_plausible_configs,
        )

    stage2_post_elapsed_sec = time.perf_counter() - stage2_post_start_t
    stage2_elapsed_sec = time.perf_counter() - stage2_start_t
    segment_elapsed_sec = time.perf_counter() - segment_start_t

    for row in segment_rows:
        row["Stage1RuntimeSec"] = float(stage1_elapsed_sec)
        row["Stage2RuntimeSec"] = float(stage2_elapsed_sec)
        row["Stage2Phase3OptRuntimeSec"] = float(stage2_phase3_opt_elapsed_sec)
        row["Stage2PostRuntimeSec"] = float(stage2_post_elapsed_sec)
        row["SegmentRuntimeSec"] = float(segment_elapsed_sec)

    log_file_only(
        f"[Timing][Stage 2] Phase 3 optimize={stage2_phase3_opt_elapsed_sec:.3f}s | "
        f"postprocess={stage2_post_elapsed_sec:.3f}s | Total={stage2_elapsed_sec:.3f}s"
    )
    log_file_only(
        f"[Timing][Segment {num_segments}] Stage 1={stage1_elapsed_sec:.3f}s | "
        f"Stage 2={stage2_elapsed_sec:.3f}s | Total={segment_elapsed_sec:.3f}s"
    )

    return {
        "rows": segment_rows,
        "phase3_output": phase3_output,
        "timing_seconds": {
            "stage1_total": float(stage1_elapsed_sec),
            "stage2_total": float(stage2_elapsed_sec),
            "stage2_phase3_opt": float(stage2_phase3_opt_elapsed_sec),
            "stage2_post": float(stage2_post_elapsed_sec),
            "segment_total": float(segment_elapsed_sec),
        },
    }


# =============================================================================
# Main Workflow: Segment Sweep with 2-Stage Optimization
# =============================================================================
def run_segment_sweep(
    out_dirs,
    degree_max,
    n_train_samples,
    n_eval_samples,
    num_outer_iters,
    min_seg_points,
    lam,
    seg_min=1,
    seg_max=16,
    use_mixed_degrees=True,
    max_error_budget=None,
    save_per_budget_overfit_plots=True,
    refine_plausible_configs=False,
    phase3_algo="bestfirst",
    segment_budgets=None,
    clear_existing_runs=True,
    existing_phase3_outputs_by_segments=None,
):
    """
    Main workflow across segment budgets.

    Stage 1 (individual optimization):
      - Per activation function, run Phase 1 + Phase 2.

    Stage 2 (joint optimization):
      - Across all activations, run Phase 3 common-area optimization.
    """
    sweep_start_t = time.perf_counter()
    segment_schedule, seg_min_effective, seg_max_effective = helper_normalize_segment_schedule(
        seg_min=seg_min,
        seg_max=seg_max,
        segment_budgets=segment_budgets,
    )

    runs_root = Path(out_dirs["root"]) / "runs"
    if clear_existing_runs and runs_root.exists():
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    log_file_only("")
    log_file_only(f"{'=' * 80}")
    log_file_only("  CONFIGURATION")
    log_file_only(f"{'=' * 80}")
    if segment_budgets is not None:
        schedule_str = ", ".join(str(s) for s in segment_schedule)
        log_file_only(f"  Segments (trial list): {schedule_str}")
    else:
        log_file_only(f"  Segments: {seg_min_effective} to {seg_max_effective}")
    log_file_only(f"  Degree max: {degree_max}")
    log_file_only(
        f"  Max error budget: {max_error_budget:.2e}"
        if max_error_budget
        else "  Max error budget: NOT SET (will fail)"
    )
    log_file_only(
        f"  Refine all DP end states before prune (2nd CD): {'ON' if refine_plausible_configs else 'OFF'}"
    )
    log_file_only(f"  Clear existing runs: {'ON' if clear_existing_runs else 'OFF'}")
    log_file_only(f"{'=' * 80}")

    pareto_rows = []
    phase3_outputs_by_segments = dict(existing_phase3_outputs_by_segments or {})

    outer_pbar = tqdm(segment_schedule, desc="Budget sweep (segments)", ncols=120)
    for num_segments in outer_pbar:
        segment_result = helper_run_single_segment_budget(
            num_segments=num_segments,
            runs_root=runs_root,
            degree_max=degree_max,
            n_train_samples=n_train_samples,
            n_eval_samples=n_eval_samples,
            num_outer_iters=num_outer_iters,
            min_seg_points=min_seg_points,
            lam=lam,
            use_mixed_degrees=use_mixed_degrees,
            max_error_budget=max_error_budget,
            refine_plausible_configs=refine_plausible_configs,
            phase3_algo=phase3_algo,
            save_per_budget_overfit_plots=save_per_budget_overfit_plots,
        )
        pareto_rows.extend(segment_result["rows"])
        phase3_outputs_by_segments[int(num_segments)] = segment_result["phase3_output"]

    new_df = pd.DataFrame(pareto_rows)
    csv_path = Path(out_dirs["root"]) / "pareto_sweep_summary.csv"
    df = helper_merge_sweep_summary(
        csv_path=csv_path,
        new_df=new_df,
        append_existing=not clear_existing_runs,
    )
    df.to_csv(csv_path, index=False)
    log(f"\nSaved Pareto sweep summary to {csv_path}")

    if len(df) == 0:
        log("\nNo data collected. Check that seg_min <= seg_max.")
        return {
            "summary_df": df,
            "phase3_outputs_by_segments": phase3_outputs_by_segments,
            "runtime_csv_path": None,
        }

    should_write_aggregate = bool(phase3_outputs_by_segments) and len(segment_schedule) > 1
    if should_write_aggregate:
        log("\nGenerating aggregate Phase 3 summary...")

        aggregate_area_opt_dir = Path(out_dirs["root"]) / "area_err_optimization"
        aggregate_area_opt_dir.mkdir(parents=True, exist_ok=True)

        viz_aggregate_phase3_summary(
            phase3_outputs_by_segments,
            aggregate_area_opt_dir,
            degree_max,
            seg_min_effective,
            seg_max_effective,
        )
        io_save_aggregate_phase3_csv(
            phase3_outputs_by_segments,
            aggregate_area_opt_dir,
            degree_max,
        )

        log(f"Aggregate Phase 3 outputs saved to: {aggregate_area_opt_dir}")

    # -------------------------------------------------------------------------
    # Reference comparison data (sq-AAE from paper)
    # Compute hardware area for uniform-degree configs using our area model
    # Must convert degree_counts -> c_counts before calling helper_hardware_area
    # -------------------------------------------------------------------------
    is_halved = 2
    c_counts_16bp_1st = helper_degree_counts_to_c_counts([0, 15, 0, 0])
    c_counts_16bp_2nd = helper_degree_counts_to_c_counts([0, 0, 15, 0])
    c_counts_8bp_1st = helper_degree_counts_to_c_counts([0, 7 * is_halved, 0, 0])
    c_counts_8bp_2nd = helper_degree_counts_to_c_counts([0, 0, 7 * is_halved, 0])

    area_16bp_1st = helper_hardware_area(*c_counts_16bp_1st)
    area_16bp_2nd = helper_hardware_area(*c_counts_16bp_2nd)
    area_8bp_1st = helper_hardware_area(*c_counts_8bp_1st)
    area_8bp_2nd = helper_hardware_area(*c_counts_8bp_2nd)

    comparison_data = {
        "Tanh": [
            {"params": 15 * 2, "area": area_16bp_1st, "y": 4.26e-7, "label": "16BP-1st"},
            {"params": 15 * 3, "area": area_16bp_2nd, "y": 1.02e-8, "label": "16BP-2nd"},
            {"params": 7 * 2 * is_halved, "area": area_8bp_1st, "y": 1.37e-5, "label": "8BP-1st"},
            {"params": 7 * 3 * is_halved, "area": area_8bp_2nd, "y": 9.28e-7, "label": "8BP-2nd"},
        ],
        "Sigmoid": [
            {"params": 15 * 2, "area": area_16bp_1st, "y": 2.88e-7, "label": "16BP-1st"},
            {"params": 15 * 3, "area": area_16bp_2nd, "y": 6.50e-9, "label": "16BP-2nd"},
        ],
        "GELU": [
            {"params": 15 * 2, "area": area_16bp_1st, "y": 1.89e-7, "label": "16BP-1st"},
            {"params": 15 * 3, "area": area_16bp_2nd, "y": 9.07e-9, "label": "16BP-2nd"},
            {"params": 7 * 2 * is_halved, "area": area_8bp_1st, "y": 3.65e-6, "label": "8BP-1st"},
            {"params": 7 * 3 * is_halved, "area": area_8bp_2nd, "y": 3.78e-7, "label": "8BP-2nd"},
        ],
        "SiLU": [
            {"params": 7 * 2 * is_halved, "area": area_8bp_1st, "y": 4.27e-5, "label": "8BP-1st"},
            {"params": 7 * 3 * is_halved, "area": area_8bp_2nd, "y": 1.35e-6, "label": "8BP-2nd"},
        ],
    }

    pareto_dir = Path(out_dirs["root"]) / "pareto"
    if clear_existing_runs and pareto_dir.exists():
        shutil.rmtree(pareto_dir)
    pareto_dir.mkdir(parents=True, exist_ok=True)

    act_order = list(ACTIVATION_FUNCTIONS.keys())
    present_acts = [a for a in act_order if a in set(df["Activation"].unique())]
    seg_col = "SegmentsRequested" if "SegmentsRequested" in df.columns else (
        "SegmentsFinal" if "SegmentsFinal" in df.columns else None
    )
    plot_seg_min, plot_seg_max = helper_infer_segment_range(
        df=df,
        seg_col=seg_col,
        fallback_min=seg_min_effective,
        fallback_max=seg_max_effective,
    )

    def annotate_segment_labels(ax, label_points):
        if seg_col is None or not label_points:
            return

        fig = ax.figure
        # Needed so data->display transforms are stable when computing collisions.
        fig.canvas.draw()
        px_per_pt = fig.dpi / 72.0
        candidate_offsets_pt = [
            (4, 5), (4, -10), (-15, 5), (-15, -10),
            (9, 0), (-20, 0), (0, 11), (0, -13),
            (12, 8), (-24, 8), (12, -12), (-24, -12),
        ]
        placed_positions_px = []

        for p in sorted(label_points, key=lambda t: (float(t["x"]), float(t["y"]))):
            xv = float(p["x"])
            yv = float(p["y"])
            if not (np.isfinite(xv) and np.isfinite(yv) and yv > 0):
                continue

            pt_x_px, pt_y_px = ax.transData.transform((xv, yv))
            best = None
            for dx_pt, dy_pt in candidate_offsets_pt:
                tx_px = pt_x_px + dx_pt * px_per_pt
                ty_px = pt_y_px + dy_pt * px_per_pt
                collisions = 0
                min_sep = float("inf")
                for ex_px, ey_px in placed_positions_px:
                    dx_abs = abs(tx_px - ex_px)
                    dy_abs = abs(ty_px - ey_px)
                    if dx_abs < 20 and dy_abs < 12:
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

            placed_positions_px.append((best["tx_px"], best["ty_px"]))
            arrowprops = None
            if best["collisions"] > 0:
                arrowprops = dict(
                    arrowstyle="-",
                    color=p["color"],
                    lw=0.35,
                    alpha=0.45,
                    shrinkA=0,
                    shrinkB=2,
                )

            ax.annotate(
                f"S{int(p['seg'])}",
                xy=(xv, yv),
                xytext=(best["dx_pt"], best["dy_pt"]),
                textcoords="offset points",
                fontsize=7.2,
                color=p["color"],
                alpha=0.9,
                clip_on=True,
                bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=0.2),
                arrowprops=arrowprops,
            )

    def format_bounds_text(bounds_by_seg, value_fmt):
        if not bounds_by_seg:
            return "N/A"
        items = [f"S{int(seg):02d}={value_fmt(val)}" for seg, val in sorted(bounds_by_seg.items())]
        chunk_size = 4
        lines = []
        for i in range(0, len(items), chunk_size):
            lines.append(" | ".join(items[i:i + chunk_size]))
        return "\n".join(lines)

    upper_area_bounds_by_seg = {}
    upper_param_bounds_by_seg = {}
    bounds_source = None

    if phase3_outputs_by_segments:
        for seg, res in sorted(phase3_outputs_by_segments.items()):
            try:
                seg_i = int(seg)
            except (TypeError, ValueError):
                continue

            area_ub = res.get("final_upper_bound_area", None)
            if area_ub is not None and np.isfinite(area_ub):
                upper_area_bounds_by_seg[seg_i] = float(area_ub)

            ub_counts = res.get("final_upper_bound", None)
            if ub_counts is not None:
                try:
                    ub_counts = [int(c) for c in ub_counts]
                    upper_param_bounds_by_seg[seg_i] = int(sum((deg + 1) * c for deg, c in enumerate(ub_counts)))
                except (TypeError, ValueError):
                    pass

        if upper_area_bounds_by_seg and upper_param_bounds_by_seg:
            bounds_source = "Stage 2 (Phase 3)"

    if (not upper_area_bounds_by_seg or not upper_param_bounds_by_seg) and seg_col is not None and "Counts_c0c1c2c3" in df.columns:
        max_counts_by_seg = {}
        for _, row in df.iterrows():
            sv = row.get(seg_col, np.nan)
            if pd.isna(sv):
                continue
            try:
                seg_i = int(round(float(sv)))
            except (TypeError, ValueError):
                continue

            counts_raw = row.get("Counts_c0c1c2c3", None)
            if counts_raw is None or (isinstance(counts_raw, float) and np.isnan(counts_raw)):
                continue
            try:
                counts = ast.literal_eval(str(counts_raw))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(counts, (list, tuple)):
                continue

            counts = [int(c) for c in counts]
            if len(counts) < (degree_max + 1):
                counts = counts + [0] * ((degree_max + 1) - len(counts))
            counts = counts[: degree_max + 1]

            prev = max_counts_by_seg.get(seg_i, [0] * (degree_max + 1))
            max_counts_by_seg[seg_i] = [max(prev[i], counts[i]) for i in range(degree_max + 1)]

        for seg_i, max_counts in sorted(max_counts_by_seg.items()):
            c_counts = helper_degree_counts_to_c_counts(max_counts)
            upper_area_bounds_by_seg[seg_i] = float(helper_hardware_area(*c_counts))
            upper_param_bounds_by_seg[seg_i] = int(sum((deg + 1) * c for deg, c in enumerate(max_counts)))

        if upper_area_bounds_by_seg and upper_param_bounds_by_seg:
            bounds_source = "summary inference"

    def plot_overlay_metric(
        *,
        metric_name,
        x_col,
        ref_x_key,
        upper_bounds_by_seg,
        common_bound_label,
        x_label,
        title,
        bounds_note_title,
        bounds_text,
        out_path,
        csv_out_path,
    ):
        from matplotlib.lines import Line2D
        import matplotlib.transforms as mtransforms

        fig, ax = plt.subplots(figsize=(18.5, 11), dpi=220)

        def helper_adjust_ref_color(base_color, order_idx):
            rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
            if order_idx == 1:
                # 1st-order references are lighter.
                rgb = rgb + (1.0 - rgb) * 0.38
            elif order_idx == 2:
                # 2nd-order references are darker.
                rgb = rgb * 0.74
            return tuple(np.clip(rgb, 0.0, 1.0))

        def helper_ref_style(ref_cfg_label, base_color):
            marker_by_bp = {16: "D", 8: "^"}
            bp_idx = None
            order_idx = None
            tag = str(ref_cfg_label)
            if "BP-" in tag:
                bp_part, order_part = tag.split("BP-", 1)
                bp_digits = "".join(ch for ch in bp_part if ch.isdigit())
                order_digits = "".join(ch for ch in order_part if ch.isdigit())
                if bp_digits:
                    try:
                        bp_idx = int(bp_digits)
                    except ValueError:
                        bp_idx = None
                if order_digits:
                    try:
                        order_idx = int(order_digits)
                    except ValueError:
                        order_idx = None

            marker = marker_by_bp.get(bp_idx, "P")
            color = helper_adjust_ref_color(base_color, order_idx)
            size = 110 if bp_idx == 16 else 95 if bp_idx == 8 else 102
            return marker, color, size

        def helper_ref_meta(ref_cfg_label):
            bp_idx = None
            order_idx = None
            tag = str(ref_cfg_label)
            if "BP-" in tag:
                bp_part, order_part = tag.split("BP-", 1)
                bp_digits = "".join(ch for ch in bp_part if ch.isdigit())
                order_digits = "".join(ch for ch in order_part if ch.isdigit())
                if bp_digits:
                    try:
                        bp_idx = int(bp_digits)
                    except ValueError:
                        bp_idx = None
                if order_digits:
                    try:
                        order_idx = int(order_digits)
                    except ValueError:
                        order_idx = None
            return bp_idx, order_idx

        # Pre-collect subsets for stable range computation.
        sub_by_act = {}
        x_vals = []
        x_limit_candidates = []
        for act in present_acts:
            sub = df[df["Activation"] == act].copy()
            if x_col == "TotalArea":
                sub = sub[np.isfinite(sub["TotalArea"].values)]
            sub = sub[np.isfinite(sub[x_col].values) & np.isfinite(sub["TrainMSE"].values) & (sub["TrainMSE"].values > 0)]
            sub = sub.sort_values(x_col)
            if len(sub) == 0:
                continue
            sub_by_act[act] = sub
            x_vals.extend(sub[x_col].tolist())
            x_limit_candidates.extend(sub[x_col].tolist())

        if not sub_by_act:
            plt.close(fig)
            return

        eval_label_used = False
        act_colors = {}
        label_points = []
        legend_probe_points = []
        export_rows = []

        for act, sub in sub_by_act.items():
            line_train, = ax.plot(
                sub[x_col],
                sub["TrainMSE"],
                marker="o",
                linewidth=2,
                alpha=0.85,
                label=f"{act} (Ours – Train)",
            )
            act_colors[act] = line_train.get_color()
            ax.plot(
                sub[x_col],
                sub["EvalMSE"],
                linestyle="--",
                color=line_train.get_color(),
                linewidth=1.8,
                alpha=0.70,
                label="Ours – Eval (dashed)" if not eval_label_used else None,
            )
            eval_label_used = True
            legend_probe_points.extend(
                [
                    (float(xv), float(yv))
                    for xv, yv in zip(sub[x_col].values, sub["TrainMSE"].values)
                    if np.isfinite(xv) and np.isfinite(yv) and yv > 0
                ]
            )
            legend_probe_points.extend(
                [
                    (float(xv), float(yv))
                    for xv, yv in zip(sub[x_col].values, sub["EvalMSE"].values)
                    if np.isfinite(xv) and np.isfinite(yv) and yv > 0
                ]
            )

            for _, row in sub.iterrows():
                xv = row.get(x_col, np.nan)
                if not np.isfinite(xv):
                    continue
                xv_f = float(xv)

                seg_i = None
                if seg_col is not None:
                    sv = row.get(seg_col, np.nan)
                    if pd.notna(sv):
                        try:
                            seg_i = int(round(float(sv)))
                        except (TypeError, ValueError):
                            seg_i = None

                y_train = row.get("TrainMSE", np.nan)
                if np.isfinite(y_train) and float(y_train) > 0:
                    y_train_f = float(y_train)
                    export_rows.append(
                        {
                            "metric": metric_name,
                            "x_field": x_col,
                            "x": xv_f,
                            "y": y_train_f,
                            "series_type": "ours",
                            "curve": "train",
                            "activation": act,
                            "segment_budget": seg_i,
                            "ref_label": "",
                            "ref_bp": np.nan,
                            "ref_order": np.nan,
                            "bound_label": "",
                            "bound_source": "",
                            "bound_group_segments": "",
                        }
                    )
                    if seg_i is not None:
                        label_points.append(
                            {
                                "x": xv_f,
                                "y": y_train_f,
                                "seg": int(seg_i),
                                "color": line_train.get_color(),
                            }
                        )

                y_eval = row.get("EvalMSE", np.nan)
                if np.isfinite(y_eval) and float(y_eval) > 0:
                    export_rows.append(
                        {
                            "metric": metric_name,
                            "x_field": x_col,
                            "x": xv_f,
                            "y": float(y_eval),
                            "series_type": "ours",
                            "curve": "eval",
                            "activation": act,
                            "segment_budget": seg_i,
                            "ref_label": "",
                            "ref_bp": np.nan,
                            "ref_order": np.nan,
                            "bound_label": "",
                            "bound_source": "",
                            "bound_group_segments": "",
                        }
                    )

        seen_ref_labels = set()
        for act in sub_by_act.keys():
            if act not in comparison_data:
                continue
            for p in comparison_data[act]:
                ref_label = f"{act} (Ref – {p['label']})"
                label = None if ref_label in seen_ref_labels else ref_label
                seen_ref_labels.add(ref_label)
                marker, ref_color, ref_size = helper_ref_style(p["label"], act_colors.get(act, "gray"))

                ax.scatter(
                    p[ref_x_key],
                    p["y"],
                    marker=marker,
                    s=ref_size,
                    color=ref_color,
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=10,
                    label=label,
                )
                if np.isfinite(p[ref_x_key]):
                    x_limit_candidates.append(float(p[ref_x_key]))
                if np.isfinite(p[ref_x_key]) and np.isfinite(p["y"]) and float(p["y"]) > 0:
                    legend_probe_points.append((float(p[ref_x_key]), float(p["y"])))
                    ref_bp, ref_order = helper_ref_meta(p["label"])
                    export_rows.append(
                        {
                            "metric": metric_name,
                            "x_field": x_col,
                            "x": float(p[ref_x_key]),
                            "y": float(p["y"]),
                            "series_type": "ref",
                            "curve": "ref",
                            "activation": act,
                            "segment_budget": np.nan,
                            "ref_label": str(p["label"]),
                            "ref_bp": ref_bp if ref_bp is not None else np.nan,
                            "ref_order": ref_order if ref_order is not None else np.nan,
                            "bound_label": "",
                            "bound_source": "",
                            "bound_group_segments": "",
                        }
                    )

        if upper_bounds_by_seg:
            grouped_bounds = {}
            for seg_i, bound_val in sorted(upper_bounds_by_seg.items()):
                if not np.isfinite(bound_val):
                    continue
                if x_col == "TotalParameters":
                    key = int(round(float(bound_val)))
                else:
                    key = float(np.round(float(bound_val), 3))
                if key not in grouped_bounds:
                    grouped_bounds[key] = {"value": float(bound_val), "segments": []}
                grouped_bounds[key]["segments"].append(int(seg_i))

            grouped_entries = sorted(grouped_bounds.values(), key=lambda t: t["value"])
            x_with_bounds = x_vals + [g["value"] for g in grouped_entries]
            x_min_plot = float(np.nanmin(x_with_bounds))
            x_max_plot = float(np.nanmax(x_with_bounds))
            x_span_plot = max(x_max_plot - x_min_plot, 1e-12)
            x_limit_candidates.extend([g["value"] for g in grouped_entries if np.isfinite(g["value"])])
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            line_colors = plt.cm.cividis(np.linspace(0.12, 0.92, len(grouped_entries)))

            label_rows_y = [0.985, 0.935, 0.885, 0.835]
            last_x_norm_by_row = [-1e9] * len(label_rows_y)
            min_dx_norm = 0.045

            for idx, (entry, line_color) in enumerate(zip(grouped_entries, line_colors)):
                bound_val = entry["value"]
                ax.axvline(
                    x=bound_val,
                    color=line_color,
                    linestyle=(0, (3, 3)),
                    alpha=0.65,
                    linewidth=1.2,
                )

                seg_tokens = [f"S{s:02d}" for s in entry["segments"]]
                if len(seg_tokens) > 3:
                    seg_label = ", ".join(seg_tokens[:3]) + ", ..."
                else:
                    seg_label = ", ".join(seg_tokens)

                x_norm = (float(bound_val) - x_min_plot) / x_span_plot
                row_idx = None
                for r_i in range(len(label_rows_y)):
                    if x_norm - last_x_norm_by_row[r_i] >= min_dx_norm:
                        row_idx = r_i
                        break
                if row_idx is None:
                    row_idx = int(np.argmin(last_x_norm_by_row))
                last_x_norm_by_row[row_idx] = x_norm

                ax.text(
                    bound_val,
                    label_rows_y[row_idx],
                    seg_label,
                    transform=trans,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=7.2,
                    color=line_color,
                    clip_on=False,
                    bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.4),
                )

                seg_tokens = [f"S{s:02d}" for s in entry["segments"]]
                seg_group_str = "|".join(seg_tokens)
                for seg_i in entry["segments"]:
                    export_rows.append(
                        {
                            "metric": metric_name,
                            "x_field": x_col,
                            "x": float(bound_val),
                            "y": np.nan,
                            "series_type": "common_bound",
                            "curve": "vertical",
                            "activation": "",
                            "segment_budget": int(seg_i),
                            "ref_label": "",
                            "ref_bp": np.nan,
                            "ref_order": np.nan,
                            "bound_label": common_bound_label,
                            "bound_source": bounds_source if bounds_source is not None else "",
                            "bound_group_segments": seg_group_str,
                        }
                    )

        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("MSE (log scale)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # Keep a tight x-domain (no large artificial right-side lane).
        finite_x = np.asarray([v for v in x_limit_candidates if np.isfinite(v)], dtype=float)
        if finite_x.size > 0:
            x_data_min = float(np.nanmin(finite_x))
            x_data_max = float(np.nanmax(finite_x))
            x_span = max(x_data_max - x_data_min, 1e-12)
            x_left_pad = 0.04 * x_span
            x_right_data_pad = 0.02 * x_span
            x_upper = x_data_max + x_right_data_pad
            if x_col == "TotalArea":
                # Prefer clean right ticks (e.g., 50k) without overextending.
                x_upper = max(x_upper, float(np.ceil(x_data_max / 1000.0) * 1000.0))
            ax.set_xlim(
                x_data_min - x_left_pad,
                x_upper,
            )

        annotate_segment_labels(ax, label_points)

        note = "Labels: S# = segment budget (shown on every train point)"
        if bounds_source is not None:
            note += f"\n{bounds_note_title} ({bounds_source})\n{bounds_text}"

        note_artist = ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.85,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.5),
        )

        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l and l not in by_label:
                by_label[l] = h

        by_label[f"{common_bound_label}"] = Line2D(
            [0], [0], color="dimgray", linestyle=(0, (3, 3)), linewidth=1.2
        )
        by_label["Ref BP marker: 16BP"] = Line2D(
            [0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="black", color="black", markersize=7
        )
        by_label["Ref BP marker: 8BP"] = Line2D(
            [0], [0], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor="black", color="black", markersize=7
        )
        by_label["Ref order shade: 1st=lighter"] = Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#cfcfcf", markeredgecolor="black", color="black", markersize=6
        )
        by_label["Ref order shade: 2nd=darker"] = Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#6f6f6f", markeredgecolor="black", color="black", markersize=6
        )

        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=10,
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.88),
            borderaxespad=0.2,
            framealpha=0.92,
            markerscale=0.58,
            labelspacing=0.28,
            handletextpad=0.45,
            borderpad=0.35,
            columnspacing=0.75,
        )

        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.92)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        if csv_out_path is not None:
            export_df = pd.DataFrame(export_rows)
            export_df.to_csv(csv_out_path, index=False)

    area_bounds_text = format_bounds_text(upper_area_bounds_by_seg, lambda v: f"{v:.1f}")
    out_png = pareto_dir / f"pareto_area_max{degree_max}_seg{plot_seg_min}-{plot_seg_max}.png"
    out_csv_area = pareto_dir / f"pareto_area_points_max{degree_max}_seg{plot_seg_min}-{plot_seg_max}.csv"
    plot_overlay_metric(
        metric_name="AREA",
        x_col="TotalArea",
        ref_x_key="area",
        upper_bounds_by_seg=upper_area_bounds_by_seg,
        common_bound_label="Common Area",
        x_label="Hardware Area Cost (μm²)",
        title=f"Pareto Frontier (Overlay, AREA) – degree_max={degree_max}, segments={plot_seg_min}..{plot_seg_max})",
        bounds_note_title="Upper area bound",
        bounds_text=area_bounds_text,
        out_path=out_png,
        csv_out_path=out_csv_area,
    )
    log(f"Saved AREA Pareto plot: {out_png}")
    log(f"Saved AREA plot points CSV: {out_csv_area}")

    param_bounds_text = format_bounds_text(upper_param_bounds_by_seg, lambda v: f"{int(round(v))}")
    out_png_params = pareto_dir / f"pareto_params_max{degree_max}_seg{plot_seg_min}-{plot_seg_max}.png"
    out_csv_params = pareto_dir / f"pareto_params_points_max{degree_max}_seg{plot_seg_min}-{plot_seg_max}.csv"
    plot_overlay_metric(
        metric_name="PARAMS",
        x_col="TotalParameters",
        ref_x_key="params",
        upper_bounds_by_seg=upper_param_bounds_by_seg,
        common_bound_label="Common Config",
        x_label="Total Parameters = segments × (degree + 1)",
        title=f"Pareto Frontier (Overlay, PARAMS) – degree_max={degree_max}, segments={plot_seg_min}..{plot_seg_max})",
        bounds_note_title="Upper param bound",
        bounds_text=param_bounds_text,
        out_path=out_png_params,
        csv_out_path=out_csv_params,
    )
    log(f"Saved PARAMS Pareto plot: {out_png_params}")
    log(f"Saved PARAMS plot points CSV: {out_csv_params}")

    sweep_elapsed_sec = time.perf_counter() - sweep_start_t
    runtime_csv_path = io_save_runtime_matrix_csv(
        summary_df=df,
        out_root=out_dirs["root"],
        append_existing=not clear_existing_runs,
        run_elapsed_sec=sweep_elapsed_sec,
    )
    if runtime_csv_path is not None:
        log(f"Saved runtime matrix CSV: {runtime_csv_path}")
        log_file_only(f"[Timing][Sweep] wall-clock={sweep_elapsed_sec:.3f}s")

    return {
        "summary_df": df,
        "phase3_outputs_by_segments": phase3_outputs_by_segments,
        "runtime_csv_path": runtime_csv_path,
    }


def run_binary_search_mode(
    out_dirs,
    degree_max,
    n_train_samples,
    n_eval_samples,
    num_outer_iters,
    min_seg_points,
    lam,
    seg_max,
    max_error_budget,
    refine_plausible_configs=False,
    phase3_algo="bestfirst",
):
    """
    Binary search to find the minimum number of segments where all functions
    can achieve the target error budget AFTER Stage 2 (Phase 3) optimization.

    Algorithm:
    1. Binary search between 1 and seg_max using Stage 2 criterion
       (each tested segment runs full Stage 1 + Stage 2 and is saved immediately)

    Args:
        seg_max: Maximum number of segments to try
        max_error_budget: Required - the target MSE budget
        Other args: Same as run_segment_sweep

    Returns:
        int: Minimum number of segments found, or -1 if seg_max is insufficient
    """
    if max_error_budget is None:
        raise ValueError("Binary search mode requires --max-error to be specified")

    log(f"")
    log(f"{'='*60}")
    log(f"Binary Search Mode: Finding minimum segments for error budget")
    log(f"{'='*60}")
    log(f"  Target MSE budget: {max_error_budget:.2e}")
    log(f"  Maximum segments: {seg_max}")
    log(f"")

    trial_segments = []
    trial_segment_set = set()
    feasible_by_segment = {}
    trial_phase3_outputs = {}

    def record_trial_segment(seg):
        seg_i = int(seg)
        if seg_i not in trial_segment_set:
            trial_segment_set.add(seg_i)
            trial_segments.append(seg_i)

    def run_trial(seg):
        seg_i = int(seg)
        if seg_i in feasible_by_segment:
            return feasible_by_segment[seg_i]

        clear_existing_runs = len(feasible_by_segment) == 0

        sweep_result = run_segment_sweep(
            out_dirs=out_dirs,
            degree_max=degree_max,
            n_train_samples=n_train_samples,
            n_eval_samples=n_eval_samples,
            num_outer_iters=num_outer_iters,
            min_seg_points=min_seg_points,
            lam=lam,
            seg_min=seg_i,
            seg_max=seg_i,
            use_mixed_degrees=True,
            max_error_budget=max_error_budget,
            save_per_budget_overfit_plots=True,
            refine_plausible_configs=refine_plausible_configs,
            phase3_algo=phase3_algo,
            segment_budgets=[seg_i],
            clear_existing_runs=clear_existing_runs,
        )

        trial_df = sweep_result.get("summary_df")
        if isinstance(trial_df, pd.DataFrame) and len(trial_df) > 0 and "SegmentsRequested" in trial_df.columns:
            seg_rows = trial_df[trial_df["SegmentsRequested"] == seg_i]
            if len(seg_rows) > 0:
                seg_row = seg_rows.iloc[-1]
                t_s1 = float(seg_row.get("Stage1RuntimeSec", np.nan))
                t_s2 = float(seg_row.get("Stage2RuntimeSec", np.nan))
                t_s2_opt = float(seg_row.get("Stage2Phase3OptRuntimeSec", np.nan))
                t_seg = float(seg_row.get("SegmentRuntimeSec", np.nan))
                if np.isfinite(t_seg):
                    log_file_only(
                        f"[Timing][Binary search trial seg={seg_i}] "
                        f"Stage 1={t_s1:.3f}s | Stage 2={t_s2:.3f}s "
                        f"(Phase 3 optimize={t_s2_opt:.3f}s) | Total={t_seg:.3f}s"
                    )

        seg_phase3 = sweep_result.get("phase3_outputs_by_segments", {}).get(seg_i, None)
        if seg_phase3 is not None:
            trial_phase3_outputs[seg_i] = seg_phase3

        feasible = True
        final_solutions = seg_phase3.get("final_solutions", {}) if seg_phase3 is not None else {}
        for act_name in ACTIVATION_FUNCTIONS.keys():
            sol = final_solutions.get(act_name, {})
            mse = sol.get("mse", None)
            if mse is None or mse > max_error_budget:
                feasible = False
                break

        feasible_by_segment[seg_i] = feasible
        return feasible

    # Step 1: Binary search for minimum (Stage 2 criterion)
    log(f"Step 1: Binary search for minimum segments (Stage 2 / Phase 3 criterion)...")

    lo = 1
    hi = seg_max
    best_seg = None

    while lo <= hi:
        mid = (lo + hi) // 2

        log(f"  Testing {mid} segments (search range: [{lo}, {hi}])...")
        log_file_only(f"")
        log_file_only(f"Binary search: testing {mid} segments (range [{lo}, {hi}])")
        record_trial_segment(mid)

        feasible = run_trial(mid)

        if feasible:
            log(f"    ✓ FEASIBLE → trying fewer segments")
            log_file_only(f"  ✓ {mid} segments: FEASIBLE")
            best_seg = mid
            hi = mid - 1  # Try fewer segments
        else:
            log(f"    ✗ NOT feasible → need more segments")
            log_file_only(f"  ✗ {mid} segments: NOT feasible")
            lo = mid + 1  # Need more segments

    log(f"")
    log(f"Binary search complete!")
    tested_sorted = sorted(trial_segments)
    tested_str = ", ".join(str(s) for s in tested_sorted)
    if best_seg is None:
        log(f"  No feasible segment budget found in [1, {seg_max}] for error budget {max_error_budget:.2e}")
        log(f"Binary search result: no feasible solution (tested trials: {tested_str})")
        return -1

    # Generate aggregate Phase 3 summary across tested trial budgets.
    if trial_phase3_outputs:
        aggregate_area_opt_dir = Path(out_dirs["root"]) / "area_err_optimization"
        aggregate_area_opt_dir.mkdir(parents=True, exist_ok=True)
        trial_segs_sorted = sorted(trial_phase3_outputs.keys())
        viz_aggregate_phase3_summary(
            trial_phase3_outputs,
            aggregate_area_opt_dir,
            degree_max,
            trial_segs_sorted[0],
            trial_segs_sorted[-1],
        )
        io_save_aggregate_phase3_csv(
            trial_phase3_outputs,
            aggregate_area_opt_dir,
            degree_max,
        )

    log(f"  Minimum segments needed: {best_seg}")
    log(f"")
    log(f"Binary search result: minimum segments = {best_seg} (tested trials: {tested_str})")

    return best_seg


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """
    Main entry point for piecewise polynomial activation function fitting.

    The algorithm has 2 stages for each segment count:

    STAGE 1 (individual optimization): Phase 1 + Phase 2
      - Phase 1: Breakpoint optimization (hard-coded uniform degree_max fit)
      - Phase 2: Mixed-degree DP selection under error budget

    STAGE 2 (joint optimization): Phase 3
      - Finds one shared hardware configuration that works across all activations
      - Optimizes the common upper bound on degree counts

    Two modes:
      - Sweep mode (default): Run Stage 1 + Stage 2 for seg_min..seg_max
      - Binary search mode (--binary-search): Find minimum segments to meet error budget
        using Stage 2 (Phase 3) final solutions as the stop criterion
    """
    parser = argparse.ArgumentParser(
        description="Piecewise Polynomial Fitting with 2-Stage Optimization (Stage 1: Phase 1+2, Stage 2: Phase 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seg-min", type=int, default=1,
                        help="Minimum number of segments")
    parser.add_argument("--seg-max", type=int, default=16,
                        help="Maximum number of segments")
    parser.add_argument("--max-error", type=float, required=True,
                        help="Max allowed MSE budget (required)")
    parser.add_argument("--degree-max", type=int, default=3,
                        help="Maximum polynomial degree (must be <= 3 for area model)")
    parser.add_argument("--refine-plausible", action="store_true",
                        help="Run 2nd CD refinement for each plausible config (slower but more accurate)")
    parser.add_argument("--binary-search", action="store_true",
                        help="Binary search mode: find minimum segments to meet error budget using Stage 2 (Phase 3) final solutions (requires --max-error)")
    parser.add_argument("--phase3-algo", type=str, default="bestfirst",
                        choices=["iterative", "tree", "bruteforce", "bestfirst"],
                        help="Stage 2 (Phase 3) search algorithm: iterative (greedy, fast), tree (DFS with pruning), "
                             "bruteforce (sort all by area, optimal), bestfirst (priority queue, optimal)")
    args = parser.parse_args()

    # ==========================================================================
    # Configuration
    # ==========================================================================
    seg_min = args.seg_min
    seg_max = args.seg_max
    max_error_budget = args.max_error
    degree_max = args.degree_max
    refine_plausible_configs = args.refine_plausible
    binary_search_mode = args.binary_search
    phase3_algo = args.phase3_algo

    # Validate
    if seg_min > seg_max:
        parser.error(f"--seg-min ({seg_min}) cannot be greater than --seg-max ({seg_max}). Did you swap them?")

    # ==========================================================================
    # Setup output directories and logging
    # ==========================================================================
    out_root_name = helper_format_output_root_name(
        degree_max=degree_max,
        max_error_budget=max_error_budget,
        num_functions=len(ACTIVATION_FUNCTIONS),
    )
    out_dirs = io_prepare_output_dirs(out_root=out_root_name, fresh=True)

    log_dir = out_dirs["root"] / "logs"
    log_path = setup_logging(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    log(f"Outputs: {out_dirs['root']}")
    log(f"Log file: {log_path}")

    log_file_only("=" * 100)
    log_file_only("PIECEWISE POLYNOMIAL FITTING - 2-Stage Optimization (Stage 1: Phase 1+2, Stage 2: Phase 3)")
    log_file_only("=" * 100)
    log_file_only(f"Device: {device}")
    if device.type == "cuda":
        log_file_only(f"GPU: {torch.cuda.get_device_name(0)}")

    # Algorithm parameters
    n_train_samples = 1024
    n_eval_samples = 1024
    num_outer_iters = None
    min_seg_points = 10
    lam = 0.0

    log(f"")
    log(f"Configuration:")
    log(f"  seg_min={seg_min}, seg_max={seg_max}, degree_max={degree_max}")
    log(f"  max_error_budget={max_error_budget:.2e}")
    log(f"  binary_search_mode={binary_search_mode}")
    log(f"  refine_plausible_configs={refine_plausible_configs}")
    log(f"  stage2_algo(phase3)={phase3_algo}")
    log(f"")

    # ==========================================================================
    # Run the 2-stage algorithm
    # ==========================================================================
    if binary_search_mode:
        # Binary search: find minimum segments to meet error budget
        min_seg_found = run_binary_search_mode(
            out_dirs=out_dirs,
            degree_max=degree_max,
            n_train_samples=n_train_samples,
            n_eval_samples=n_eval_samples,
            num_outer_iters=num_outer_iters,
            min_seg_points=min_seg_points,
            lam=lam,
            seg_max=seg_max,
            max_error_budget=max_error_budget,
            refine_plausible_configs=refine_plausible_configs,
            phase3_algo=phase3_algo,
        )

        if min_seg_found < 0:
            log_file_only("=" * 80)
            log_file_only("BINARY SEARCH FAILED - max segments insufficient")
            log_file_only("=" * 80)
        else:
            log_file_only("=" * 80)
            log_file_only(f"BINARY SEARCH COMPLETE - minimum segments = {min_seg_found}")
            log_file_only("=" * 80)
    else:
        # Normal sweep mode
        run_segment_sweep(
            out_dirs=out_dirs,
            degree_max=degree_max,
            n_train_samples=n_train_samples,
            n_eval_samples=n_eval_samples,
            num_outer_iters=num_outer_iters,
            min_seg_points=min_seg_points,
            lam=lam,
            seg_min=seg_min,
            seg_max=seg_max,
            use_mixed_degrees=True,
            max_error_budget=max_error_budget,
            save_per_budget_overfit_plots=True,
            refine_plausible_configs=refine_plausible_configs,
            phase3_algo=phase3_algo,
        )

    log_file_only("=" * 80)
    log_file_only("DONE")
    log_file_only("=" * 80)
    log("Done! See log file for details.")


if __name__ == "__main__":
    main()
