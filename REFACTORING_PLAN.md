# Nash Equilibrium Algorithms Refactoring Plan

## Overview

Refactoring to clearly separate the two core Nash equilibrium algorithms while using a simple function-based design suitable for research code.

## Core Algorithms

**Two Fundamental Algorithms (both in `solver.py`)**:

1. **`find_nash_eq1()`** - Intra-Orbit Nash Finding
   - Finds Nash equilibria using local unitary transformations
   - Preserves entanglement structure (stays within orbit)
   - Uses differential best response dynamics

2. **`train()`** - Inter-Orbit Optimization
   - Optimizes fiducial state across different orbits
   - Changes entanglement structure via gradient descent on MPS parameters
   - Calls `find_nash_eq1()` for refinement within each orbit

## Proposed Directory Structure

```
nash/
├── solver.py                 # Core algorithms: find_nash_eq1(), train()
│
├── src/                      # Logically important source code
│   ├── __init__.py
│   ├── mps_utils.py          # MPS manipulation (canonical form, random states)
│   ├── tensor_ops.py         # Tensor network operations (from misc.py)
│   ├── game.py               # Game definitions (Hamiltonians, payoffs)
│   └── entanglement.py       # Entanglement parameter calculations
│
├── tests/                    # Pytest test suite
│   ├── __init__.py
│   ├── test_solver.py        # Core solver tests
│   ├── test_solver_4qubit.py # 4-qubit system tests
│   ├── test_misc_torch.py    # NumPy/PyTorch equivalence tests
│   ├── test_game.py          # Game definition tests
│   ├── test_consistencies.py # Cross-module consistency tests
│   ├── test_5players.py      # Multi-player (5+) tests
│   ├── test_seed.py          # Reproducibility tests
│   └── test_save_load.py     # Persistence tests
│
├── utils/                    # Engineering utilities
│   ├── __init__.py
│   ├── data_io.py            # Loading/saving results, pickle handling
│   ├── mpi_utils.py          # MPI distributed computing helpers
│   └── wandb_utils.py        # Weights & Biases logging helpers
│
├── notebooks/                # Interactive exploration (Jupyter)
│   ├── qpd_data_analysis.ipynb
│   ├── qpd_4players.ipynb
│   ├── qpd.ipynb
│   └── ...
│
├── configs/                  # Configuration files
│   ├── default_3player.yaml
│   ├── default_4player.yaml
│   └── sweep_configs/
│       ├── sweep_config.yaml
│       ├── sweep_config_lronly.yaml
│       └── sweep_example_simple.yaml
│
├── experiments/              # Job submission and sweep scripts
│   ├── run_3player.sh        # SLURM job script for 3-player
│   ├── run_sweep.py          # W&B sweep runner
│   └── setup_cluster_sweep.py # Cluster sweep setup utility
│
├── README.md
├── requirements.txt
└── CLAUDE.md
```

## Core `solver.py` Design

```python
"""
Core Nash equilibrium algorithms for quantum games.

Two main algorithms:
1. find_nash_eq1() - Find NE within a unitary orbit (local search)
2. train() - Optimize fiducial state across orbits (global search)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

from src.mps_utils import to_canonical_form, to_comp_basis, get_rand_mps
from src.tensor_ops import apply_unitary, mps_overlap
from src.game import get_default_H
from src.entanglement import compute_entanglement_params


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SolverConfig:
    """Configuration for intra-orbit Nash equilibrium finding."""
    max_iter: int = 10000
    alpha: float = 0.01           # Learning rate for unitary updates
    convergence_threshold: float = 1e-7
    expl_threshold: float = 5e-4  # Global exploitability threshold
    use_tqdm: bool = False
    trace_history: bool = False


@dataclass
class TrainConfig:
    """Configuration for inter-orbit optimization."""
    n_steps: int = 2000
    lr: float = 0.01
    mps_bond_dim: int = 2

    # Nash refinement
    solver_config: SolverConfig = None  # Uses default if None
    max_nash_attempts: int = 20

    # Logging
    use_wandb: bool = True
    wandb_project: str = 'quantum-nash-optimization'
    log_interval: int = 100
    save_dir: str = 'results'


# ============================================================================
# Intra-Orbit Nash Finding
# ============================================================================

def find_nash_eq1(
    Psi: list[np.ndarray] | np.ndarray,
    H: list[np.ndarray],
    config: SolverConfig = None,
) -> dict:
    """
    Find Nash equilibrium within a unitary orbit using differential best response.

    This algorithm applies local unitary transformations to each player's subsystem
    until no player can improve their payoff. The entanglement structure is preserved.

    Args:
        Psi: Initial state (MPS tensors or computational basis array)
        H: List of Hamiltonian tensors, one per player
        config: Solver configuration (uses defaults if None)

    Returns:
        dict with keys:
            - 'state': Final MPS tensors at Nash equilibrium
            - 'energies': Payoffs for each player
            - 'converged': Whether algorithm converged
            - 'num_iters': Number of iterations taken
            - 'local_expl': Final local exploitability
            - 'global_expl': Global exploitability (if validated)
            - 'history': Energy trajectory (if trace_history=True)

    Algorithm:
        1. Compute energy gradient w.r.t. local unitaries for each player
        2. Extract unitary update via SVD: U = (V @ Wh).T.conj()
        3. Apply all unitaries simultaneously (synchronous update)
        4. Converge when local exploitability < threshold
    """
    if config is None:
        config = SolverConfig()

    # ... implementation ...


def compute_exploitability(psi: np.ndarray, H: list[np.ndarray], player_idx: int) -> float:
    """
    Compute global exploitability for a single player.

    Uses differential evolution to find the maximum payoff gain from
    unilateral deviation in the exp(iY) direction.

    Args:
        psi: Quantum state in computational basis
        H: List of Hamiltonians
        player_idx: Index of player to compute exploitability for

    Returns:
        Maximum payoff gain from deviation (0 if at equilibrium)
    """
    # ... implementation ...


def compute_energies(Psi: list[np.ndarray] | np.ndarray, H: list[np.ndarray]) -> np.ndarray:
    """
    Compute expected payoffs for all players.

    Args:
        Psi: State (MPS or computational basis)
        H: Stacked Hamiltonian

    Returns:
        Array of energies, one per player
    """
    # ... implementation ...


# ============================================================================
# Inter-Orbit Optimization
# ============================================================================

def train(
    H: list[np.ndarray] = None,
    config: TrainConfig = None,
    initial_state: list[np.ndarray] = None,
) -> dict:
    """
    Optimize fiducial state across orbits to find high-welfare Nash equilibria.

    This algorithm uses gradient descent on MPS parameters to explore different
    entanglement structures, with Nash refinement at each step.

    Args:
        H: Game Hamiltonians (uses default QPD if None)
        config: Training configuration
        initial_state: Starting MPS (random if None)

    Returns:
        dict with keys:
            - 'final_state': Best Nash equilibrium found
            - 'trajectory': DataFrame with full optimization history
            - 'best_welfare': Maximum welfare achieved
            - 'entanglement_params': Entanglement invariants at each step

    Algorithm:
        For each optimization step:
        1. Compute energies and gradients w.r.t. MPS parameters
        2. Update MPS via Adam optimizer (maximize welfare)
        3. Convert to canonical form
        4. Refine to Nash equilibrium using find_nash_eq1()
        5. Log metrics and save checkpoints
    """
    if config is None:
        config = TrainConfig()
    if H is None:
        H = get_default_H(n_players=3)

    # ... implementation ...


# ============================================================================
# Convenience Functions
# ============================================================================

def find_all_equilibria(
    Psi: list[np.ndarray],
    H: list[np.ndarray],
    n_restarts: int = 20,
    config: SolverConfig = None,
) -> list[dict]:
    """
    Find multiple Nash equilibria using random restarts.

    Useful for exploring the equilibrium landscape within an orbit.
    """
    # ... implementation ...


def validate_nash_equilibrium(
    Psi: list[np.ndarray],
    H: list[np.ndarray],
    threshold: float = 1e-3,
) -> tuple[bool, np.ndarray]:
    """
    Validate that a state is a Nash equilibrium.

    Returns:
        (is_nash, exploitabilities): Tuple of validation result and per-player exploitabilities
    """
    # ... implementation ...
```

## Source Code Organization (`src/`)

### `src/mps_utils.py`
```python
"""Matrix Product State utilities."""

def get_rand_mps(L: int, chi: int, dtype=np.complex128, seed=None) -> list[np.ndarray]:
    """Generate random MPS with given bond dimension."""

def to_canonical_form(Psi: list[np.ndarray], center: int = 0) -> list[np.ndarray]:
    """Convert MPS to mixed canonical form."""

def to_comp_basis(Psi: list[np.ndarray]) -> np.ndarray:
    """Contract MPS to full state vector."""

def from_comp_basis(psi: np.ndarray, chi: int) -> list[np.ndarray]:
    """Convert state vector to MPS via SVD."""

def get_product_state(angles: list[float]) -> list[np.ndarray]:
    """Create product state MPS from Bloch sphere angles."""

def get_ghz_state(L: int) -> list[np.ndarray]:
    """Create GHZ state as MPS."""

def apply_random_unitaries(Psi: list[np.ndarray]) -> list[np.ndarray]:
    """Apply random local unitaries to each site."""
```

### `src/tensor_ops.py`
```python
"""Tensor network operations (extracted from misc.py)."""

def apply_unitary(U: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply unitary to physical leg of MPS tensor."""

def mps_overlap(Psi1: list, Psi2: list) -> complex:
    """Compute inner product of two MPS."""

def compress(Psi: list, chi_max: int) -> list:
    """Compress MPS to given bond dimension via SVD."""

def mps_2form(Psi: list, center: int = 0) -> list:
    """Convert to mixed canonical form centered at given site."""
```

### `src/game.py`
```python
"""Quantum game definitions."""

def get_default_H(n_players: int = 3, game: str = 'qpd') -> list[np.ndarray]:
    """Get default Hamiltonian for quantum Prisoner's Dilemma."""

def canonical_qpd(n_players: int) -> list[np.ndarray]:
    """Canonical QPD payoff structure."""

def get_hamiltonian_from_payoffs(payoffs: np.ndarray) -> list[np.ndarray]:
    """Convert payoff tensor to Hamiltonian list."""
```

### `src/entanglement.py`
```python
"""Entanglement parameter calculations."""

def compute_entanglement_params(psi: np.ndarray, option: str = 'I') -> np.ndarray:
    """
    Compute 5 entanglement invariants for 3-qubit state.

    Args:
        psi: State vector (shape 2x2x2 or flattened)
        option: 'I' for invariants, 'J' for derived parameters

    Returns:
        Array of 5 parameters: [I1, I2, I3, I4, I5]
        - I1, I2, I3: Single-party purities
        - I4: Two-party correlation
        - I5: Three-tangle (genuine tripartite entanglement)
    """

def compute_purity(rho: np.ndarray) -> float:
    """Compute purity Tr(rho^2) of density matrix."""

def partial_trace(psi: np.ndarray, keep: list[int]) -> np.ndarray:
    """Compute reduced density matrix by tracing out specified subsystems."""
```

## Utilities Organization (`utils/`)

### `utils/mpi_utils.py`
```python
"""MPI utilities for distributed computing."""

def get_mpi_info() -> tuple[int, int]:
    """Return (rank, size) for current MPI process."""

def distribute_work(items: list, rank: int, size: int) -> list:
    """Distribute work items across MPI ranks."""

def gather_results(local_results: list, comm) -> list:
    """Gather results from all ranks to rank 0."""
```

### `utils/data_io.py`
```python
"""Data loading and saving utilities."""

def save_results(results: dict, filepath: str):
    """Save results to pickle file with metadata."""

def load_results(filepath: str) -> dict:
    """Load results from pickle file."""

def concatenate_results(filepaths: list[str]) -> dict:
    """Concatenate results from multiple files (e.g., MPI outputs)."""

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dictionary to pandas DataFrame."""
```

### `utils/wandb_utils.py`
```python
"""Weights & Biases logging utilities."""

def init_wandb(config: dict, project: str, run_name: str = None):
    """Initialize W&B run with configuration."""

def log_step(step: int, metrics: dict):
    """Log metrics for a training step."""

def log_equilibrium(eq_result: dict, step: int):
    """Log Nash equilibrium result."""

def finish_run():
    """Finish W&B run and upload final artifacts."""
```

## Configuration Files (`configs/`)

### `configs/default_3player.yaml`
```yaml
# Default configuration for 3-player quantum Prisoner's Dilemma

solver:
  max_iter: 10000
  alpha: 0.01
  convergence_threshold: 1e-7
  expl_threshold: 5e-4

train:
  n_steps: 5000
  lr: 3.2e-3
  mps_bond_dim: 2
  max_nash_attempts: 20
  log_interval: 100

game:
  n_players: 3
  type: qpd
```

## Experiment Scripts (`experiments/`)

### `experiments/run_3player.sh`
```bash
#!/bin/bash
#SBATCH --job-name=nash_3p
#SBATCH --output=logs/%j.out
#SBATCH --time=4:00:00
#SBATCH --mem=8G

source .venv/bin/activate
python -c "
from solver import train
from configs import load_config

config = load_config('configs/default_3player.yaml')
train(config=config)
"
```

## Migration Plan

### Phase 1: Reorganize Files ✅ COMPLETE

1. Create directory structure
2. Move existing code to appropriate locations:
   - `find_nash_eq1()` and `train()` → keep in `solver.py` (already there)
   - MPS utilities → `src/mps_utils.py`
   - Tensor operations → `src/tensor_ops.py`
   - Game definitions → `src/game.py`
   - Entanglement calculations → `src/entanglement.py`
   - Tests → `tests/`
   - Notebooks → `notebooks/`
   - Sweep scripts → `experiments/`
   - Sweep configs → `configs/sweep_configs/`
3. Move deprecated code to `old/`:
   - `opt_mps_fiducial_state.py` (PyTorch trainer, superseded by solver.py)
   - `misc_torch.py`, `mps_utils_torch.py`, `game_torch.py` (deprecated PyTorch versions)

### Phase 2: NOT NEEDED

The core algorithms `find_nash_eq1()` and `train()` are already consolidated in `solver.py`.
No further code consolidation required.

## Benefits

1. **Clear separation**: Core algorithms in one file (`solver.py`)
2. **Simpler design**: Functions instead of classes
3. **Research-friendly**: Easy to modify and experiment
4. **Reproducible**: Config files + experiment scripts
5. **Modular**: Source code separated from utilities
6. **Testable**: Notebooks for interactive testing and visualization

## Example Usage

```python
# Simple intra-orbit Nash finding
from solver import find_nash_eq1, SolverConfig
from src.mps_utils import get_rand_mps
from src.game import get_default_H

Psi = get_rand_mps(L=3, chi=2)
H = get_default_H(n_players=3)

result = find_nash_eq1(Psi, H)
print(f"Converged: {result['converged']}")
print(f"Energies: {result['energies']}")
print(f"Exploitability: {result['global_expl']}")


# Full optimization across orbits
from solver import train, TrainConfig

config = TrainConfig(
    n_steps=5000,
    lr=3.2e-3,
    mps_bond_dim=2,
)
result = train(config=config)
print(f"Best welfare: {result['best_welfare']}")
```
