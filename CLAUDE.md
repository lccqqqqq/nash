# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for computing Nash equilibria in quantum games using tensor network methods (Matrix Product States). The primary application is analyzing multi-player quantum Prisoner's Dilemma variants.

**Key Technologies:**
- NumPy for MPS tensor operations (primary, avoids device-specific bugs)
- PyTorch for optimization and automatic differentiation
- MPI (via mpi4py) for distributed computing
- Jupyter notebooks for interactive analysis and visualization
- Weights & Biases (wandb) for experiment tracking

## Environment Setup

**Python Environment:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Jupyter Kernel Setup:**
After activating the virtual environment, register it as a Jupyter kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=nash-venv --display-name "Python (nash-venv)"
```

Then select the `.venv` or `nash-venv` kernel in Jupyter notebooks.

**Running Tests:**
```bash
# Run MPS/MPO implementation tests
pytest test_misc_torch.py -v -s

# Or run directly
python test_misc_torch.py
```

## Code Architecture

### Core Modules

**misc.py & misc_torch.py**
- Twin implementations (NumPy and PyTorch) of Matrix Product State (MPS) and Matrix Product Operator (MPO) operations
- Key functions:
  - `group_legs()` / `ungroup_legs()`: Tensor index manipulation
  - `mps_2form()`: Canonical form conversion (A/B forms)
  - `mps_overlap()`: Inner product computation
  - `compress()`: SVD-based bond dimension reduction
  - `mpo_on_mps()`: Operator application
- `misc_torch.py` is the active implementation used in optimization
- `test_misc_torch.py` verifies numerical equivalence between implementations

**opt_mps_fiducial_state.py**
- Main training script implementing hybrid optimization for quantum Nash equilibria
- Algorithm: Gradient ascent (Adam) + differential best response dynamics
- Key components:
  - `find_nash_eq()`: Iterative Nash equilibrium solver using differential best response
  - `batch_compute_exploitability()`: Global exploitability calculation via unitary sampling
  - `compute_energy()`: Expected payoff computation
  - `train()`: Main training loop with automatic Nash refinement
  - Configuration classes: `DataConfig`, `NEFinderConfig`, `TrainerConfig`

**qpd3.py**
- MPI-distributed grid search over initial state configurations
- Functions:
  - `spherical_meshgrid()`: Generate parameter grids on high-dimensional spheres
  - `pure_state_grid()`: 3-qubit state parametrization grids
  - `get_canonical_form_state()`: Convert parameters to quantum states
  - `equilibrium_finding()`: Single Nash equilibrium search
  - `process_grid()`: Distributed computation with checkpoint saving

**mps_utils.py & mps_utils_torch.py**
- **`mps_utils.py`** (PRODUCTION): Pure NumPy implementation of MPS utilities
  - Avoids Apple MPS backend bugs with complex number operations
  - All operations run correctly on CPU
  - Key functions: `to_canonical_form()`, `get_rand_mps()`, `to_comp_basis()`, `get_product_state()`, `get_ghz_state()`
  - API uses `dtype: np.dtype` instead of `device` parameter
- **`mps_utils_torch.py`** (DEPRECATED): Original PyTorch implementation
  - Kept for reference only
  - Has device management but suffers from MPS complex einsum bugs on Apple Silicon
  - Do not use for production code

**Important - MPS Backend Bug:**
Apple's Metal Performance Shaders (MPS) backend has bugs with complex number operations in `torch.einsum` and related functions. When using `dtype=torch.complex64/128`, einsum contractions produce incorrect results on `device='mps'`. This is why `mps_utils.py` uses pure NumPy instead of PyTorch. If you need GPU acceleration with complex numbers, use CUDA devices, not MPS.

### Jupyter Notebooks

**opt_fiducial_state.ipynb**
- Primary analysis notebook for 3-player quantum games
- Loads and visualizes training results
- Entanglement parameter analysis

**qpd.ipynb**
- Quantum Prisoner's Dilemma exploration
- Classical vs quantum game theory comparisons

**prisoner.ipynb**
- Classical game theory foundations
- Nash equilibrium basics

**more_players.ipynb**
- Extensions to >3 player scenarios

## Common Development Tasks

**Running Optimization:**
```python
# In Python or notebook
from opt_mps_fiducial_state import train, TrainerConfig, NEFinderConfig, DataConfig

# Configure training
trainer_cfg = TrainerConfig()
trainer_cfg.mps_bond_dim = 2           # MPS bond dimension (χ)
trainer_cfg.lr = 3.2e-3                # Learning rate (scales with χ²)
trainer_cfg.n_optimizer_steps = 30000  # Training iterations
trainer_cfg.use_wandb = True           # Enable W&B logging

solver_cfg = NEFinderConfig()
solver_cfg.alpha = 10                  # Nash solver learning rate
solver_cfg.expl_threshold = 1e-3       # Exploitability tolerance

data_cfg = DataConfig()

# Run training
train(trainer_cfg=trainer_cfg, solver_cfg=solver_cfg, data_cfg=data_cfg)
```

Results are automatically saved to `nash_data/` with descriptive filenames containing hyperparameters and timestamps.

**Running Distributed Grid Search:**
```bash
# Example: 8 MPI workers
mpirun -n 8 python qpd3.py
```
Results are saved as checkpoints in `qpd3_results_*` directories with per-worker files.

**Device Selection:**
The codebase uses a **hybrid approach**:
- **MPS operations (NumPy)**: Run on CPU to avoid complex number bugs
  - All functions in `mps_utils.py` use NumPy
  - No GPU acceleration, but correct results with complex dtypes
- **Optimization (PyTorch)**: Can use GPU when available
  - **Mac (Apple Silicon)**: Can use MPS for real-valued operations only
  - **Linux/Windows with NVIDIA GPU**: Can use CUDA
  - **Otherwise**: Falls back to CPU

**Important for Mac users**: The MPS utilities (`mps_utils.py`) run on CPU using NumPy. This avoids Apple's MPS backend bugs with complex `torch.einsum` operations. If you need GPU acceleration for MPS operations with complex numbers, use a CUDA-enabled machine instead of Apple Silicon.

## Key Concepts

**Nash Equilibrium Finding:**
The algorithm uses **differential best response dynamics**:
1. Each player computes the gradient of their payoff with respect to local unitary operations
2. Extract unitary updates via SVD of the gradient
3. Apply updates simultaneously (synchronous)
4. Iterate until local exploitability < threshold

**Exploitability:**
- **Local exploitability**: Energy increase from infinitesimal unitary perturbations (used for convergence)
- **Global exploitability**: Maximum gain from any single-qubit unitary deviation (used for validation)
- A state is an ε-Nash equilibrium if global exploitability < ε

**MPS Representation:**
States are represented as periodic MPS (periodic boundary conditions):
- 3 tensors: `[A₀, A₁, A₂]` each of shape `(phys=2, χ_L, χ_R)`
- Full state: trace over bond indices after contraction
- Bond dimension χ controls entanglement capacity (χ=2 is typical)

**Entanglement Parameters:**
The code computes 5 invariants (I₁-I₅) characterizing tripartite entanglement:
- I₁, I₂, I₃: Single-party purities
- I₄: Two-party correlation measure
- I₅: Three-party entanglement (related to 3-tangle)

**Hamiltonian Structure:**
The default 3-player quantum Prisoner's Dilemma uses diagonal payoff matrices stored as shape `(2,2,2,2,2,2)` tensors (6 indices for 3 players × 2 measurement outcomes each).

## File Naming Conventions

- `mps_utils.py`: Pure NumPy MPS utilities (PRODUCTION)
- `mps_utils_torch.py`: PyTorch MPS utilities (DEPRECATED - reference only)
- `*_torch.py`: PyTorch implementations (e.g., `misc_torch.py`)
- `*.py` (without suffix): Standalone scripts or NumPy implementations
- `test_*.py`: Test files using pytest
- Saved results: `qpd_opt_chi{χ}_lr{lr}_steps{N}_alpha{α}_expl{ε}_{timestamp}.pkl`

## Important Notes

- **Use NumPy API for MPS utilities**: When calling functions from `mps_utils.py`, use NumPy dtypes (e.g., `dtype=np.complex64`) instead of PyTorch dtypes and device parameters. No `device` parameter is needed.
  ```python
  # Correct (NumPy API)
  from mps_utils import get_rand_mps
  Psi = get_rand_mps(L=3, chi=2, dtype=np.complex64, seed=42)

  # Wrong (old PyTorch API - don't use)
  Psi = get_rand_mps(L=3, chi=2, default_dtype=torch.complex64, device='mps')
  ```
- **Learning rate scaling**: For gradient ascent, scale lr ∝ χ² when changing bond dimension
- **Nash solver can fail**: The differential BR dynamics may converge to local optima. The code attempts multiple random restarts (default: 20) and keeps the best result.
- **Canonical form is critical**: Always convert MPS to canonical form before running Nash equilibrium solver for numerical stability
- The code works specifically for 3-site systems; generalizing to arbitrary lengths requires modifying Einstein summation specifications in `batch_compute_energy()`
