# Quantum Game Theory: Nash Equilibria via Matrix Product States

Research codebase for computing Nash equilibria in multi-player quantum games using tensor network optimization methods.

## Overview

This repository implements algorithms for finding Nash equilibria in quantum Prisoner's Dilemma games with 2-6 players. States are represented as periodic Matrix Product States (MPS), and equilibria are found using differential best response dynamics combined with gradient-based optimization.

### Key Features

- **Hybrid optimization**: Gradient ascent (Adam) + differential best response
- **Scalable tensor networks**: MPS representation with controlled bond dimension
- **Distributed computing**: MPI-based grid search and W&B sweep management
- **Dual backends**: NumPy (CPU, production) and PyTorch (GPU-capable, research)
- **Comprehensive testing**: 12+ test modules validating numerical accuracy and convergence

---

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup Jupyter kernel (for notebooks)
pip install ipykernel
python -m ipykernel install --user --name=nash-venv --display-name "Python (nash-venv)"
```

---

## Quick Start

### 1. Run Nash Equilibrium Solver

```python
from opt_mps_fiducial_state import train, TrainerConfig, NEFinderConfig, DataConfig

# Configure training
trainer_cfg = TrainerConfig()
trainer_cfg.mps_bond_dim = 2           # Bond dimension χ
trainer_cfg.lr = 3.2e-3                # Learning rate (scale ∝ χ²)
trainer_cfg.n_optimizer_steps = 30000  # Training iterations

solver_cfg = NEFinderConfig()
solver_cfg.alpha = 10                  # Nash solver learning rate
solver_cfg.expl_threshold = 1e-3       # Convergence threshold

# Run optimization
train(trainer_cfg=trainer_cfg, solver_cfg=solver_cfg, data_cfg=DataConfig())
```

Results save to `nash_data/` with filenames encoding hyperparameters.

### 2. Run Distributed Grid Search

```bash
# Launch 8 MPI workers for 3-qubit state space exploration
mpirun -n 8 python qpd3.py
```

### 3. Analyze Results

```python
from load_results import load_result, analyze_run

# Load saved equilibrium
result = load_result('nash_data/qpd_opt_chi2_lr0.0032_*.pkl')

# Analyze equilibrium properties
analyze_run(result)
```

---

## Core Scripts

### Main Entry Points

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| **opt_mps_fiducial_state.py** | PyTorch-based Nash equilibrium trainer (GPU-capable) | `TrainerConfig`, `NEFinderConfig`, `DataConfig` |
| **solver.py** | NumPy-based Nash solver (CPU, for production/validation) | Similar configs, NumPy arrays |
| **qpd3.py** / **qpd4.py** | MPI-distributed grid search over state parameters | MPI rank/size |
| **run_sweep.py** | W&B hyperparameter sweep runner | W&B sweep config YAML |

### Core Libraries

| Module | Purpose | Implementation |
|--------|---------|----------------|
| **mps_utils.py** | MPS state manipulation (canonical forms, random states, GHZ) | NumPy (PRODUCTION) |
| **misc.py** / **misc_torch.py** | MPS/MPO tensor network algorithms (overlap, compression, operator application) | NumPy / PyTorch |
| **game.py** / **game_torch.py** | Quantum Prisoner's Dilemma payoff definitions (2-6 players) | NumPy / PyTorch |
| **nash_utils.py** | Analysis tools (exploitability, entanglement parameters, visualization) | NumPy |

### Utilities

| Script | Purpose |
|--------|---------|
| **load_results.py** | Load, analyze, and compare experiment results |
| **cat_pkl.py** | Concatenate pickle files from distributed runs |
| **setup_cluster_sweep.py** | Generate cluster job scripts and W&B sweeps |

---

## Architecture

```
┌─────────────────────────────────────────┐
│     OPTIMIZATION WORKFLOWS              │
├─────────────────────────────────────────┤
│  opt_mps_fiducial_state.py (PyTorch)   │  ← Main trainer (GPU)
│  solver.py (NumPy)                      │  ← Production solver (CPU)
│  qpd3.py (MPI)                          │  ← Distributed grid search
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│       CORE LIBRARIES                    │
├─────────────────────────────────────────┤
│  MPS Utils    │  Tensor Algos │  Games  │
│  mps_utils.py │  misc.py      │ game.py │
│               │  misc_torch   │ game_   │
│               │               │ torch   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│     ANALYSIS & MANAGEMENT               │
├─────────────────────────────────────────┤
│  nash_utils.py  │  load_results.py      │
│  cat_pkl.py     │  setup_cluster_sweep  │
└─────────────────────────────────────────┘
```

### Backend Strategy

- **MPS operations**: Pure NumPy (CPU) to avoid Apple MPS backend bugs with complex `torch.einsum`
- **Optimization**: PyTorch (GPU-capable) for automatic differentiation
- **Game definitions**: Dual NumPy/PyTorch implementations

---

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest test_*.py -v

# Specific test categories
pytest test_misc_torch.py -v         # NumPy ↔ PyTorch equivalence
pytest test_solver_4qubit.py -v      # 4-qubit solver tests
pytest test_nash_convergence_4qubit.py -v  # Convergence validation
pytest test_consistencies.py -v      # Cross-module consistency
```

### Test Coverage

- **Numerical accuracy**: NumPy/PyTorch equivalence (`test_misc_torch.py`)
- **Canonical forms**: MPS normalization and gauge fixing (`test_solver.py`)
- **Multi-qubit**: 4, 5, 6 player systems (`test_solver_4qubit.py`, `test_5players.py`)
- **Convergence**: Nash equilibrium solver validation (`test_nash_convergence_4qubit.py`)
- **Reproducibility**: Random seed consistency (`test_seed.py`)
- **Persistence**: Save/load integrity (`test_save_load.py`)

---

## Key Concepts

### Nash Equilibrium Finding

Uses **differential best response dynamics**:
1. Compute payoff gradient w.r.t. local unitary operations
2. Extract unitary updates via SVD of gradient matrix
3. Apply updates simultaneously (synchronous)
4. Iterate until exploitability < threshold

### Exploitability

- **Local exploitability**: Energy gain from infinitesimal unitary perturbations (convergence metric)
- **Global exploitability**: Max gain from any single-qubit unitary (validation metric)
- State is ε-Nash equilibrium if global exploitability < ε

### MPS Representation

Periodic boundary conditions:
- 3 tensors `[A₀, A₁, A₂]`, each shape `(phys=2, χ_L, χ_R)`
- Full state: `trace(A₀ A₁ A₂)` after contraction
- Bond dimension χ controls entanglement capacity (typical: χ=2)

### Entanglement Parameters

5 invariants characterizing tripartite entanglement:
- **I₁, I₂, I₃**: Single-party purities
- **I₄**: Two-party correlation
- **I₅**: Three-party entanglement (3-tangle)

---

## Configuration

### Learning Rate Scaling

Scale learning rate ∝ χ² when changing bond dimension:
```python
# χ=2: lr = 3.2e-3
# χ=4: lr = 3.2e-3 × (4/2)² = 1.28e-2
```

### Nash Solver Parameters

```python
solver_cfg = NEFinderConfig()
solver_cfg.alpha = 10              # Learning rate for unitary updates
solver_cfg.expl_threshold = 1e-3   # Convergence tolerance
solver_cfg.max_steps = 10000       # Max iterations
solver_cfg.n_restarts = 20         # Random restarts (avoid local optima)
```

### Device Selection

```python
# Auto-detect best available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Note: MPS utilities always use NumPy (CPU) to avoid complex number bugs
```

---

## Hyperparameter Sweeps with W&B

The repository provides two scripts for managing distributed hyperparameter sweeps using Weights & Biases:

### Workflow Overview

```
1. Create sweep          →  2. Generate commands    →  3. Submit to cluster
   (setup_cluster_sweep)     (commands.txt)             (multirun system)

4. Monitor progress      →  5. Join from other machines (optional)
   (W&B dashboard)           (run_sweep.py)
```

### Script 1: `experiments/setup_cluster_sweep.py`

Creates a W&B sweep and generates a commands file for cluster multirun systems.

**Purpose**:
- Initialize a new W&B sweep from YAML or Python config
- Save sweep ID for later reference
- Generate `commands.txt` file for distributed execution on compute clusters

**Basic Usage**:

```bash
# From YAML configuration
python experiments/setup_cluster_sweep.py \
    --config configs/sweep_configs/sweep_config.yaml \
    --num-workers 10 \
    --count-per-worker 10

# From Python config (Bayesian optimization)
python experiments/setup_cluster_sweep.py \
    --create-sweep \
    --method bayes \
    --num-workers 20 \
    --count-per-worker 5

# From Python config (Grid search)
python experiments/setup_cluster_sweep.py \
    --create-sweep \
    --method grid \
    --num-workers 25 \
    --count-per-worker 3
```

**Key Arguments**:

| Argument | Description | Default |
|----------|-------------|---------|
| `--config PATH` | Path to YAML sweep config | None |
| `--create-sweep` | Use Python config instead of YAML | False |
| `--method {bayes,grid}` | Sweep method (when using Python config) | `bayes` |
| `--name NAME` | Custom sweep name (shown in W&B) | Auto-generated |
| `--project PROJECT` | W&B project name | `nash-equilibrium` |
| `--entity ENTITY` | W&B entity (team) name | Your username |
| `--num-workers N` | Number of parallel workers | 10 |
| `--count-per-worker N` | Runs per worker | 10 |
| `--output FILE` | Output commands file | `commands.txt` |
| `--save-id FILE` | File to save sweep ID | `sweep_id.txt` |

**Example Output**:

```
✓ Sweep created successfully!
Sweep ID:   abc123def
Sweep Name: seed-sweep-v1
Project:    nash-equilibrium
URL:        https://wandb.ai/username/nash-equilibrium/sweeps/abc123def

✓ Sweep ID saved to: sweep_id.txt
✓ Generating commands file: commands.txt
  Workers: 10
  Runs per worker: 10
  Total runs: up to 100

Next Steps:
1. Review the commands file:
   cat commands.txt

2. Submit to your cluster's multirun system:
   multirun commands.txt

3. Monitor sweep progress:
   https://wandb.ai/username/nash-equilibrium/sweeps/abc123def
```

**Generated `commands.txt`**:

```bash
/usr/bin/python3 run_sweep.py --sweep-id abc123def --count 10
/usr/bin/python3 run_sweep.py --sweep-id abc123def --count 10
...
```

Each line is a command that can be executed on a separate cluster node.

### Script 2: `experiments/run_sweep.py`

Runs W&B sweep agents to execute hyperparameter search runs.

**Purpose**:
- Execute sweep runs (either create new sweep or join existing)
- Used by cluster workers via commands.txt
- Can also be run manually for local sweeps

**Basic Usage**:

```bash
# Join existing sweep (typical cluster usage)
python experiments/run_sweep.py \
    --sweep-id abc123def \
    --count 10

# Create new sweep and run (local development)
python experiments/run_sweep.py \
    --config configs/sweep_configs/sweep_config.yaml \
    --count 20

# Create from Python config
python experiments/run_sweep.py \
    --create-sweep \
    --method bayes \
    --count 15
```

**Key Arguments**:

| Argument | Description | Default |
|----------|-------------|---------|
| `--config PATH` | Path to YAML sweep config | None |
| `--create-sweep` | Create sweep from Python config | False |
| `--method {bayes,grid}` | Sweep method (bayes or grid) | `bayes` |
| `--sweep-id ID` | Existing sweep ID to join | None |
| `--count N` | Number of runs to execute | 10 |
| `--project PROJECT` | W&B project name | `nash-equilibrium` |
| `--entity ENTITY` | W&B entity (team) name | None |

**Sweep Configuration Examples**:

The script includes built-in Python configs for common scenarios:

**Bayesian Optimization** (`SWEEP_CONFIG`):
- Method: `bayes`
- Swept parameters: `eps`, `subroutine_lr`, `seed`
- Metric: Maximize `welfare`
- Early termination: Hyperband

**Grid Search** (`SWEEP_CONFIG_GRID`):
- Method: `grid`
- Total combinations: 3 seeds × 7 eps × 5 lr = 105 runs
- Discrete parameter values

### Complete Workflow Example

**Step 1**: Create sweep for 100 total runs (10 workers × 10 runs each)

```bash
python experiments/setup_cluster_sweep.py \
    --create-sweep \
    --method bayes \
    --name "chi8-5player-sweep" \
    --project nash-equilibrium \
    --num-workers 10 \
    --count-per-worker 10
```

**Step 2**: Review generated commands

```bash
cat commands.txt
# Output:
# /usr/bin/python3 run_sweep.py --sweep-id abc123def --count 10
# /usr/bin/python3 run_sweep.py --sweep-id abc123def --count 10
# ... (10 lines total)
```

**Step 3**: Submit to cluster multirun system

```bash
# On SLURM-based clusters with multirun
multirun commands.txt

# Or manually submit each worker as a separate job
for cmd in $(cat commands.txt); do
    sbatch --wrap="$cmd"
done
```

**Step 4**: Monitor progress at W&B dashboard

```
https://wandb.ai/username/nash-equilibrium/sweeps/abc123def
```

**Step 5** (Optional): Join sweep from another machine

```bash
# Read sweep ID from file
SWEEP_ID=$(cat sweep_id.txt)

# Run additional workers
python experiments/run_sweep.py \
    --sweep-id $SWEEP_ID \
    --count 20
```

### YAML Configuration Format

Create custom sweep configs in `configs/sweep_configs/`:

```yaml
method: bayes  # or 'grid', 'random'
metric:
  name: welfare
  goal: maximize

parameters:
  # Fixed parameters
  chi:
    value: 8
  num_players:
    value: 5

  # Swept parameters (Bayesian)
  eps:
    distribution: log_uniform_values
    min: 0.001
    max: 0.1

  subroutine_lr:
    distribution: log_uniform_values
    min: 0.01
    max: 0.6

  # Discrete choices
  seed:
    values: [42, 123, 456, 789, 1337]

# Optional early termination
early_terminate:
  type: hyperband
  min_iter: 10
  eta: 2
```

For grid search, use discrete `values` instead of `distribution`:

```yaml
method: grid
parameters:
  eps:
    values: [0.001, 0.01, 0.1]
  subroutine_lr:
    values: [0.01, 0.1, 0.3]
  seed:
    values: [42, 123, 456]
# Total runs: 3 × 3 × 3 = 27
```

### Tips

**Grid Search Coverage**:
- Ensure `num_workers × count_per_worker ≥ total_grid_size`
- The script warns if you're under-sampling the grid
- Example: 3×5×5=75 combinations requires at least 75 total runs

**Bayesian Optimization**:
- More runs = better exploration (typically 50-200)
- Use early termination to prune unpromising runs
- Monitor parallel workers on W&B dashboard

**Distributed Execution**:
- Each worker independently pulls sweep configs from W&B
- No inter-worker communication needed
- Workers can be added/removed dynamically

**Resuming Sweeps**:
- Sweep ID is saved in `sweep_id.txt`
- Can join sweep anytime with `--sweep-id`
- Progress persists across workers and sessions

---

## File Naming Conventions

**Results**: `qpd_opt_chi{χ}_lr{lr}_steps{N}_alpha{α}_expl{ε}_{timestamp}.pkl`

**Modules**:
- `*_torch.py`: PyTorch implementations
- `*.py` (no suffix): NumPy or standalone scripts
- `test_*.py`: Pytest test modules

---

## Known Issues

**Apple MPS Backend Bug**: PyTorch's Metal Performance Shaders backend has bugs with complex `torch.einsum` operations. Use `mps_utils.py` (NumPy, CPU) instead of `mps_utils_torch.py` for production code.

**Nash Solver Convergence**: Differential best response may converge to local optima. The solver attempts multiple random restarts (default: 20) and returns the best result.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_nash_mps,
  title = {Quantum Game Theory: Nash Equilibria via Matrix Product States},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/nash}
}
```

---

## License

[Add your license here]
