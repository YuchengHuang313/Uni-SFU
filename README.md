# Uni-SFU: Algorithm-HW Co-Design for Universal SFUs

This repository contains the public reference implementation of the offline
piecewise-polynomial approximation and joint hardware-area search used in
**“Uni-SFU: Algorithm-HW Co-Design for Universal SFUs via Mixed-Degree
Piecewise Approximation.”**

The paper was accepted at CODES 2026, part of Embedded Systems Week
(ESWEEK 2026). The public manuscript is available as
[arXiv:2608.11577](https://arxiv.org/abs/2608.11577).

Paper authors, in publication order: Miao Sun, Yucheng Huang, Mingcong Cao,
Jaehyun Park, Partha Pratim Pande, and Umit Y. Ogras. See [AUTHORS.md](AUTHORS.md)
for authorship and attribution information.

If you use this software, please cite the paper. Machine-readable citation
metadata is provided in [CITATION.cff](CITATION.cff).

```bibtex
@article{sun2026unisfu,
  title   = {Uni-SFU: Algorithm-HW Co-Design for Universal SFUs via Mixed-Degree Piecewise Approximation},
  author  = {Sun, Miao and Huang, Yucheng and Cao, Mingcong and Park, Jaehyun and Pande, Partha Pratim and Ogras, Umit Y.},
  journal = {arXiv preprint arXiv:2608.11577},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.11577}
}
```

The released software covers the approximation search, joint area
optimization, coefficient export, and result post-processing. The RTL,
technology libraries, and end-to-end DNN evaluation infrastructure described
in the paper are outside the scope of this software release.

## Project Files

| File | Purpose |
|---|---|
| `optimize_piecewise_activations.py` | Main Phase 1/2/3 experiment. A normal invocation is a **fresh run** and deletes an existing result directory with the same generated name. |
| `resume_activation_sweep.py` | Safely resumes an interrupted segment sweep at a segment boundary. It preserves completed segment folders, reconstructs their global summary state, runs only the requested remaining segment range, and then generates combined final outputs. |
| `run_error_budget_sweep.sh` | Runs `optimize_piecewise_activations.py` sequentially for one or more error budgets. Segment range, Python executable, refinement, and search algorithm are controlled with environment variables. |
| `evaluate_piecewise_configs.py` | Re-evaluates exported polynomial configuration CSVs, verifies local-to-global coefficient conversion, and writes a sibling `<results_root>_eval` tree. |
| `combine_pareto_frontiers.py` | Finds `pareto_sweep_summary.csv` files from multiple result roots and produces a combined area-vs-MSE plot plus merged/frontier CSV and text exports. |
| `compare_flex_sfu_baselines.py` | Builds a paper-oriented comparison CSV from the generated Pareto point CSVs and Flex-SFU reference cases. |
| `environment.yml` | Reproducible CPU-only Conda environment for running all project entry points. |

Files named `combined_pareto_*` and result directories named `maxdeg*_maxerr*_funcs*_results` are generated outputs, not experiment entry points.

## Environment

Create the CPU-only project environment directly from the checked-in Conda
specification:

```bash
cd /path/to/Uni-SFU
conda env create -f environment.yml
conda activate uni-sfu
```

The code currently selects `torch.device("cpu")`, so the environment deliberately
uses `pytorch-cpu` and does not install CUDA libraries. After activation, use:

```bash
PYTHON=python
```

If the environment already exists after the specification changes, synchronize
it with `conda env update -f environment.yml --prune`.

## Paper Configuration

The release defaults to the six activation functions evaluated in the paper:
GELU, SiLU, Sigmoid, Tanh, Softplus, and ELU. All six use the paper's
`[-8, 8]` approximation domain. These definitions are intentionally kept near
the top of `optimize_piecewise_activations.py` so that new target sets can be
introduced without changing the optimization pipeline.

The checked-in area equation is the regression model used by the software
search. Replacing it with a model from another process node only requires
updating `helper_hardware_area()`.

## Quick Start

```bash
$PYTHON optimize_piecewise_activations.py \
  --max-error 2e-7 \
  --seg-min 1 \
  --seg-max 16 \
  --degree-max 3 \
  --phase3-algo bestfirst \
  --refine-plausible
```

> **Destructive fresh-run behavior:** `optimize_piecewise_activations.py` derives a deterministic output name from the degree limit, error budget, and number of configured activations. It calls `io_prepare_output_dirs(..., fresh=True)`, so starting the same configuration again removes the existing same-named result directory first. Use `resume_activation_sweep.py` for an interrupted sweep whose completed segment results must be retained.

### Activation and domain configuration

There is no activation-list CLI option. Edit these definitions in `optimize_piecewise_activations.py` before starting an experiment:

- `ACTIVATION_FUNCTIONS`: activations included in Stage 1 and Stage 2.
- `ACTIVATION_DOMAINS`: per-activation input ranges.
- `DEFAULT_ACTIVATION_DOMAIN`: fallback range for activations without an override.

The public release uses GELU, SiLU, Sigmoid, Tanh, Softplus, and ELU on
`[-8, 8]`. Keep these definitions unchanged when resuming an existing result
root because the activation set is part of the experiment identity.

### Running several error budgets

`run_error_budget_sweep.sh` invokes the main program once per error budget:

```bash
PYTHON_BIN=python \
SEG_MIN=1 \
SEG_MAX=16 \
DEGREE_MAX=3 \
PHASE3_ALGO=bestfirst \
REFINE_PLAUSIBLE=1 \
BINARY_SEARCH=0 \
./run_error_budget_sweep.sh 2e-7 5e-7 1e-6
```

`REFINE_PLAUSIBLE` defaults to `0`; set it to `1` when the expensive second coordinate-descent refinement is required. `BINARY_SEARCH` also defaults to `0`.

## Algorithm Overview

The algorithm has 3 phases that run for each segment count:

### Phase 1: Breakpoint Optimization (per activation)

**Goal:** Find optimal breakpoint locations for a given number of segments.

- Uses coordinate descent to optimize breakpoint positions
- All segments use uniform `degree_max` (e.g., all cubic)
- Output: Optimized breakpoint indices that minimize fitting error

**Pseudo-code (high level):**

```text
Input: x_train, y_train, num_segments, degree_max
Initialize breakpoints uniformly across x_train
Repeat for a fixed number of outer iterations:
  For each breakpoint (excluding endpoints):
    Try a small neighborhood of candidate positions
    Fit this new trial breakpoint location with max degree
    Keep the position that minimizes total MSE
Return optimized breakpoint indices
```

### Phase 2: Mixed-Degree Selection via DP (per activation)

**Goal:** Select per-segment polynomial degrees to minimize hardware area while meeting the error budget.

- Input: Breakpoints from Phase 1
- Uses dynamic programming to enumerate all degree combinations
- Builds Pareto frontier of (area, error) trade-offs
- Selects minimum-area configuration that satisfies `max_error_budget`
- Optional (`--refine-plausible`): run 2nd CD for each Pareto-optimal config, then re-check feasibility/frontier
- Output: Per-activation chosen min-area config + Pareto-optimal frontier (feasible subset is used by Phase 3)

**DP formulation (conceptual):**

For segment $i$ and degree $d$, precompute:
- $e_{i,d}$ = MSE contribution of fitting segment $i$ with degree $d$
- $\Delta c_d$ = coefficient-count increments from choosing degree $d$

Let $S_k$ be the Pareto set of partial solutions after the first $k$ segments. Each state is a pair $(E, \mathbf{c})$ where $E$ is total MSE and $\mathbf{c}$ is the degree-count vector. Let $D_{\max}$ denote the maximum degree and $B_{\max}$ the MSE budget.

Transition:

$$
S_{k+1} = \mathrm{Pareto}\Big(\{(E + e_{k+1,d},\ \mathbf{c} + \Delta c_d) \|\ (E,\mathbf{c}) \in S_k,\ d \in [0, D_{\max}]\}\Big)
$$

At the end, filter $S_N$ by $E \le B_{\max}$ and choose the configuration with minimum hardware area.

**Meaning of each symbol:**

- $k$: segment index (0-based)
- $S_k$: Pareto set after processing the first $k$ segments
- $(E,\mathbf{c})$: a partial solution with total MSE $E$ and coefficient-count vector $\mathbf{c}=(c1,c2,c3)$
- $e_{k+1,d}$: MSE contribution of fitting segment $k{+}1$ with degree $d$
- $\Delta c_d$: how counts change when degree $d$ is chosen (for this code: $d=0\to(0,0,0)$, $d=1\to(1,0,0)$, $d=2\to(1,1,0)$, $d=3\to(1,1,1)$)
- $D_{\max}$: maximum degree allowed
- $B_{\max}$: MSE budget (= max\_error\_budget)
- $\mathrm{Pareto}(\cdot)$: remove dominated points (higher MSE and higher area) to keep only the frontier

**Pseudo-code (high level):**

```text
Input: breakpoints, x_train, y_train, degree_max, max_error_budget
For each segment and each degree in [0..degree_max]:
  Fit polynomial, record MSE contribution and degree-count cost
Run DP over segments to enumerate all (total MSE, degree-count) combos
Compute Pareto frontier (remove dominated points)
Select lowest-area config with MSE <= max_error_budget
Optionally (--refine-plausible):
  Refine each Pareto-optimal config with CD, recompute MSE, re-check feasibility, re-extract frontier
Return: chosen min-area config + Pareto frontier (with feasible subset)
```

**Worked DP example:**

Assume **3 segments** and `degree_max = 2`. The DP tracks states by $(c1,c2,c3)$ and stores the **minimum total MSE** for each state.

Per-segment MSE table (precomputed in code):

| Segment | d=0 MSE | d=1 MSE | d=2 MSE | $(\Delta c1,\Delta c2,\Delta c3)$ |
|---------|---------|---------|---------|-----------------------------------|
| 0 | 10 | 6 | 5 | d=0→(0,0,0), d=1→(1,0,0), d=2→(1,1,0) |
| 1 | 9 | 5 | 4 | d=0→(0,0,0), d=1→(1,0,0), d=2→(1,1,0) |
| 2 | 7 | 4 | 3 | d=0→(0,0,0), d=1→(1,0,0), d=2→(1,1,0) |

Initialization:

$$
S_0 = \{(c1,c2,c3)=(0,0,0) \mapsto 0\}
$$

After segment 0:

From (0,0,0) with MSE=0:

- d=0 → (0,0,0): 0+10=10
- d=1 → (1,0,0): 0+6=6
- d=2 → (1,1,0): 0+5=5

$$
S_1 = \{(0,0,0)\mapsto 10,\ (1,0,0)\mapsto 6,\ (1,1,0)\mapsto 5\}
$$

After segment 1\:

From $(0,0,0)$ with MSE=10:

- d=0 → $(0,0,0)$: $10+9=19$
- d=1 → $(1,0,0)$: $10+5=15$
- d=2 → $(1,1,0)$: $10+4=14$

From $(1,0,0)$ with MSE=6:

- d=0 → $(1,0,0)$: $6+9=15$ (ties 15)
- d=1 → $(2,0,0)$: $6+5=11$
- d=2 → $(2,1,0)$: $6+4=10$

From $(1,1,0)$ with MSE=5:

- d=0 → $(1,1,0)$: $5+9=14$ (ties 14)
- d=1 → $(2,1,0)$: $5+5=10$ (ties 10)
- d=2 → $(2,2,0)$: $5+4=9$

Final $S_2$:

$$
S_2 = \{(0,0,0)\mapsto 19,\ (1,0,0)\mapsto 15,\ (1,1,0)\mapsto 14,\ (2,0,0)\mapsto 11,\ (2,1,0)\mapsto 10,\ (2,2,0)\mapsto 9\}
$$

After segment 2 (explicit expansion style):

From $(0,0,0)$ with MSE=19:

- d=0 → $(0,0,0)$: $19+7=26$
- d=1 → $(1,0,0)$: $19+4=23$
- d=2 → $(1,1,0)$: $19+3=22$

From $(1,0,0)$ with MSE=15:

- d=0 → $(1,0,0)$: $15+7=22$
- d=1 → $(2,0,0)$: $15+4=19$
- d=2 → $(2,1,0)$: $15+3=18$

From $(1,1,0)$ with MSE=14:

- d=0 → $(1,1,0)$: $14+7=21$
- d=1 → $(2,1,0)$: $14+4=18$ (ties 18)
- d=2 → $(2,2,0)$: $14+3=17$

From $(2,0,0)$ with MSE=11:

- d=0 → $(2,0,0)$: $11+7=18$ (better than 19)
- d=1 → $(3,0,0)$: $11+4=15$
- d=2 → $(3,1,0)$: $11+3=14$

From $(2,1,0)$ with MSE=10:

- d=0 → $(2,1,0)$: $10+7=17$ (better than 18)
- d=1 → $(3,1,0)$: $10+4=14$ (ties 14)
- d=2 → $(3,2,0)$: $10+3=13$

From $(2,2,0)$ with MSE=9:

- d=0 → $(2,2,0)$: $9+7=16$ (better than 17)
- d=1 → $(3,2,0)$: $9+4=13$ (ties 13)
- d=2 → $(3,3,0)$: $9+3=12$

Final $S_3$:

$$
S_3 = \{(0,0,0)\mapsto 26,\ (1,0,0)\mapsto 22,\ (1,1,0)\mapsto 21,\ (2,0,0)\mapsto 18,\ (2,1,0)\mapsto 17,\ (2,2,0)\mapsto 16,\ (3,0,0)\mapsto 15,\ (3,1,0)\mapsto 14,\ (3,2,0)\mapsto 13,\ (3,3,0)\mapsto 12\}
$$

Suppose the MSE budget is **12**. Then only states with total MSE $\le 12$ are feasible. Among those, the algorithm computes the area for each state and picks the minimum-area feasible config.

### Phase 3: Common-Area Optimization (across all activations)

**Goal:** Find a single hardware configuration (shared coefficient counts) that works for ALL activation functions with minimum area.

The current hardware area model is:

```text
area = 60.90*c0 + 57.06*c1 + 56.37*c2 + 57.07*c3
       + 162.49 + 960.0*max_degree
```

Here, `max_degree` is the highest degree with a nonzero count (0, 1, 2, or 3). Update `helper_hardware_area()` if the hardware model changes.

**Coefficient counts from degree counts:**
- c0 = b0 + b1 + b2 + b3 (constant term in all polynomials)
- c1 = b1 + b2 + b3 (linear term in degree ≥1)
- c2 = b2 + b3 (quadratic term in degree ≥2)
- c3 = b3 (cubic term in degree 3 only)

Higher-degree segments add more coefficient-count terms, and the highest degree used also determines the global `960.0*max_degree` term.

**Search bounds:**
- `search_min[i]` = max over activations of (min count[i] among that activation's feasible configs)
- `search_max[i]` = max over activations of (max count[i] among that activation's feasible configs)

Any budget below `search_min` is guaranteed infeasible; any budget above `search_max` is wasteful.

**Pseudo-code (high level):**

```text
Input: per-activation Pareto feasible configs
Compute search_min[i] = max over activations of min count[i]
Compute search_max[i] = max over activations of max count[i]
Run chosen search algorithm to find minimum-area budget that is feasible for ALL activations
Pick, for each activation, a config that fits under the final budget
Re-select lowest-MSE config under that same budget
Run mandatory breakpoint refinement on these final Phase 3 configs (post-CD solutions)
Return final shared budget and post-CD per-activation solutions
```

## Phase 3 Algorithm Options

Use `--phase3-algo` to select the search algorithm:

### `bestfirst` (default, recommended)

Priority queue / Dijkstra-style search that explores budgets in area order.

- Starts at `search_min`, expands neighbors via priority queue ordered by area
- **Stops at first feasible budget** (guaranteed to be optimal by area order)
- Time: O(k log k) where k = rank of optimal solution
- Space: O(frontier size)
- **Best choice for production use**

**Pseudo-code:**

```text
Push search_min into min-heap by area
While heap not empty:
  pop lowest-area budget
  if feasible for all activations: return budget
  push neighbors (increment one dimension) within search_max
```

### `bruteforce`

Generate all budget combinations, sort by area, check feasibility in order.

- Generates all N = ∏(search_max[i] - search_min[i] + 1) combinations
- Sorts by area, then checks feasibility from lowest to highest
- **Stops at first feasible** (optimal)
- Time: O(N log N) for sort + O(k) feasibility checks
- Space: O(N) to store all budgets
- Simple and predictable baseline

**Pseudo-code:**

```text
Generate all budgets in [search_min..search_max]
Sort by area ascending
For each budget in sorted order:
  if feasible for all activations: return budget
```

### `tree`

Proper tree-structured DFS with partial feasibility pruning.

- Tree structure: Level 1 assigns b3, Level 2 assigns b2, ..., leaves are complete [b0,b1,b2,b3]
- At each internal node, prunes if no activation config satisfies the partial assignment
- **Must explore all feasible leaves** to find minimum area
- Time: O(feasible leaves + pruned nodes)
- Space: O(depth) = O(4)
- Useful for understanding pruning effectiveness

**Pseudo-code:**

```text
DFS over a proper tree:
  Level 1 assigns b3, Level 2 assigns b2, Level 3 assigns b1, Level 4 assigns b0
At each internal node:
  if no activation config can satisfy assigned constraints: prune subtree
At each leaf (full [b0,b1,b2,b3]):
  if feasible: update best if area is lower
Return best feasible budget after DFS completes
```

### `iterative`

Greedy iterative refinement starting from Phase 2 solutions.

- Starts with upper bound from Phase 2 min-area solutions
- Each round tries to reduce one degree count by finding alternative configs
- **May get stuck in local minima** (not guaranteed optimal)
- Time: O(rounds × degrees × activations)
- Space: O(1)
- Fast approximation for very large search spaces

**Pseudo-code:**

```text
Start from Phase-2 min-area solutions
Repeat for max_rounds:
  Try reducing one degree count by swapping one activation's config
  If the shared upper bound area improves, accept the change
Stop when no improvement in a full pass
```

## Comparison

| Algorithm | Optimal? | Early Stop? | Best For |
|-----------|----------|-------------|----------|
| `bestfirst` | ✅ Yes | Yes | **Production use** |
| `bruteforce` | ✅ Yes | Yes | Simple baseline |
| `tree` | ✅ Yes | No | Debugging/analysis |
| `iterative` | ❌ No | Yes | Quick approximation |

## Command Line Options

```bash
python optimize_piecewise_activations.py [OPTIONS]

Options:
  --max-error FLOAT       Maximum allowed MSE (required)
  --seg-min INT           Minimum segments to sweep (default: 1)
  --seg-max INT           Maximum segments to sweep (default: 16)
  --degree-max INT        Maximum polynomial degree (default: 3)
  --phase3-algo ALGO      Phase 3 algorithm: iterative, tree, bruteforce, bestfirst (default: bestfirst)
  --refine-plausible      Run 2nd CD refinement for each plausible (Pareto-optimal) config
  --binary-search         Find minimum segments via binary search using Phase 3 final solutions as criterion
```

## Resuming an Interrupted Sweep

Use `resume_activation_sweep.py` when some segment counts finished successfully but the process stopped before the full sweep and aggregate outputs were written.

The resume entry point performs the following steps:

1. Opens the existing deterministic result root without deleting it.
2. Treats a segment as complete only when both of these files exist:
   - `runs/seg_XX/area_err_optimization/phase3_summary_XXseg.txt`
   - `runs/seg_XX/area_err_optimization/phase2_vs_phase3_comparison_XXseg.csv`
3. Reconstructs the completed segments' Phase 3 state, global `pareto_sweep_summary.csv`, evaluation MSE values, and recorded timings.
4. Runs the requested remaining segment range with existing runs preserved.
5. Regenerates the aggregate Phase 3 CSV/plot, Pareto outputs, and runtime table using both the reconstructed and newly completed segments.

### Important limitations

- Resume works at **segment boundaries**, not within Phase 1/2/3 of an unfinished segment. If seg15 stopped halfway through Exp refinement, seg15 must be run again from its beginning.
- Move an unfinished `runs/seg_XX` directory aside before resuming that segment. This preserves the interrupted artifacts and prevents stale files from being mixed into the rerun.
- Use the same `--max-error`, `--degree-max`, `--refine-plausible`, Phase 3 algorithm, activation list, activation domains, and relevant source code as the original run.
- Run from the `Uni-SFU` directory. Result paths are relative to the current working directory.
- Set `--seg-min` to the first incomplete segment. Do not include already completed segments unless you intentionally want to recompute them.
- A VS Code/SSH disconnect does not stop a `tmux` job, but shutting down or restarting WSL does.

### Example: resume seg15 through seg16

Suppose seg1–14 are complete and seg15 is incomplete:

```bash
cd /path/to/Uni-SFU

# Preserve the incomplete segment before recreating seg15.
mv maxdeg3_maxerr2p00e-07_funcs6_results/runs/seg_15 \
   maxdeg3_maxerr2p00e-07_funcs6_results/runs/seg_15_interrupted_YYYYMMDD_HHMMSS

python -u resume_activation_sweep.py \
  --max-error 2e-7 \
  --seg-min 15 \
  --seg-max 16 \
  --degree-max 3 \
  --phase3-algo bestfirst \
  --refine-plausible
```

The current resume CLI is:

```text
--max-error FLOAT       Required; must match the interrupted run
--seg-min INT           First segment to rerun (default: 15)
--seg-max INT           Last segment to run (default: 16)
--degree-max INT        Maximum polynomial degree (default: 3)
--phase3-algo ALGO      iterative, tree, bruteforce, or bestfirst
--refine-plausible      Enable the expensive second-CD refinement
```

### Detached execution with tmux

The following form continues running after VS Code disconnects from WSL:

```bash
tmux new-session -d -s activation_sweep_resume \
  "cd /path/to/Uni-SFU && \
   exec conda run --no-capture-output -n uni-sfu \
     python -u resume_activation_sweep.py \
     --max-error 2e-7 --seg-min 15 --seg-max 16 --degree-max 3 \
     --phase3-algo bestfirst --refine-plausible \
     > resume_seg15_16_console.log 2>&1"
```

Monitor or attach to it with:

```bash
tmux list-sessions
tmux attach -t activation_sweep_resume
tail -f maxdeg3_maxerr2p00e-07_funcs6_results/logs/resume_*.log
```

Detach from an attached tmux session without stopping it by pressing `Ctrl-B`, then `D`.

## Output

The deterministic result-root format is:

```text
maxdeg{degree_max}_maxerr{formatted_error}_funcs{activation_count}_results/
```

For example, degree 3, error budget `2e-7`, and the six paper activations
produce `maxdeg3_maxerr2p00e-07_funcs6_results/`.

```text
maxdeg..._results/
├── logs/                              detailed fresh-run or resume logs
├── pareto_sweep_summary.csv           cross-segment, per-activation final results
├── runtime/
│   └── runtime_by_segment.csv         per-activation/segment timing matrix
├── pareto/                            aggregate area/parameter Pareto plots and point CSVs
├── area_err_optimization/             aggregate Phase 3 plot and CSV summaries
└── runs/
    └── seg_XX/
        ├── plots/                     per-segment fit visualization
        ├── inference/                 fitted values on the training grid
        ├── min_area_config/           Phase 2 selected configuration
        ├── pareto_optimal_all/        all exported Pareto configurations
        ├── pareto_optimal_feasible/   error-budget-feasible Pareto configurations
        ├── common_area_config/        final Phase 3 post-CD configurations
        └── area_err_optimization/     per-segment Phase 3 summaries and plots
```

The two files under each completed segment that are especially important for resume are `phase3_summary_XXseg.txt` and `phase2_vs_phase3_comparison_XXseg.csv`.

## Post-processing Utilities

### Re-evaluate exported configurations

To evaluate only the final common-area configs for **every available segment
budget**, run:

```bash
$PYTHON evaluate_piecewise_configs.py \
  maxdeg3_maxerr2p00e-07_funcs6_results \
  --common-only \
  --n-samples 8192
```

Add `--segments` to restrict that pass to one or more budgets. For example, the
8-segment common configs only:

```bash
$PYTHON evaluate_piecewise_configs.py \
  maxdeg3_maxerr2p00e-07_funcs6_results \
  --common-only \
  --segments 8 \
  --n-samples 8192
```

That command creates:

- `maxdeg3_maxerr2p00e-07_funcs6_results_eval/seg_08/common_config/eval.csv`
- `maxdeg3_maxerr2p00e-07_funcs6_results_eval/seg_08/common_config/global_x_coefficients.csv`

To evaluate every exported Pareto config across every available segment budget,
omit both filters:

```bash
$PYTHON evaluate_piecewise_configs.py \
  maxdeg3_maxerr2p00e-07_funcs6_results \
  --n-samples 4096
```

The full mode reads `pareto_optimal_all/` as its input set, then routes matching
results into `pareto_optimal_all_eval/`, `pareto_optimal_feasible_eval/`,
`common_config/`, and `min_area_config/`. It can evaluate and plot thousands of
configs, so `--common-only` is the practical choice when only the final common
solutions are needed.

`global_x_coefficients.csv` has one row per activation and segment, with
left/right breakpoints, degree, and ascending-power `coeff_xN` columns for
`y = coeff_x0 + coeff_x1*x + coeff_x2*x^2 + ...`. `eval.csv` contains metrics
from both the original local-coordinate evaluation and the converted global-x
evaluation, plus `local_global_max_abs_diff`. A conversion is accepted only
when that maximum difference is at most `1e-9`.

The evaluation interval is read from each config's first and last breakpoints,
so the six paper activations on `[-8,8]` are handled without a separate domain
setting. The conversion and check use float64 arithmetic; they do not model
fixed-point rounding, saturation, integer grid codes, or the exact hardware
datapath.

The complete sibling `<results_root>_eval` tree is recreated fresh on every
invocation. Do not run full and common-only evaluations concurrently against the
same results root, because they write to that same output tree.

### Combine Pareto frontiers from several runs

```bash
$PYTHON combine_pareto_frontiers.py \
  /path/to/Uni-SFU \
  --curve train
```

The script recursively discovers `pareto_sweep_summary.csv` files and writes `combined_pareto_area_overlay.png`, merged point data, frontier-only data, and a readable frontier summary by default.

### Build the Flex-SFU comparison table

```bash
$PYTHON compare_flex_sfu_baselines.py \
  maxdeg3_maxerr2p00e-07_funcs6_results/pareto/pareto_area_points_max3_seg1-16.csv \
  --params-csv maxdeg3_maxerr2p00e-07_funcs6_results/pareto/pareto_params_points_max3_seg1-16.csv \
  --summary-csv maxdeg3_maxerr2p00e-07_funcs6_results/area_err_optimization/aggregate_phase3_summary.csv \
  --out-csv maxdeg3_maxerr2p00e-07_funcs6_results/pareto/compare.csv
```

This helper is tailored to the 8-BP/16-BP Flex-SFU comparisons, matched to this project's 7-segment and 15-segment points.

## Testing

After creating the Conda environment, run the lightweight unit smoke test:

```bash
python -m unittest discover -s tests -v
```

The test checks the paper activation set and the degree-count/area-model
conversions without starting a full optimization sweep.

## License

This software is released under the [MIT License](LICENSE).
