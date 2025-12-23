# Nash Equilibrium Algorithms Refactoring Plan

## Overview

This document outlines a refactoring plan to clearly separate and highlight the two core Nash equilibrium finding algorithms in the quantum game theory codebase.

## Current Architecture Issues

1. **Mixed responsibilities**: The `train()` function combines inter-orbit optimization with intra-orbit refinement
2. **Unclear algorithm boundaries**: The two core algorithms (within/across orbits) are not clearly distinguished
3. **Code duplication**: Similar logic exists in `opt_mps_fiducial_state.py` and `solver.py`
4. **Terminology confusion**: "Nash equilibrium finding" sometimes refers to local search, sometimes to global optimization

## Proposed Architecture

### Core Concepts

**Orbit**: The set of all quantum states reachable from a given state by applying local unitary transformations to each player's subsystem. States in the same orbit have identical entanglement structure.

**Two Fundamental Algorithms**:

1. **Intra-Orbit Nash Finding** (Local Search)
   - **Input**: A quantum state (MPS representation)
   - **Output**: A Nash equilibrium within the same unitary orbit
   - **Method**: Differential best response dynamics using local unitary transformations
   - **Current implementation**: `find_nash_eq()`, `find_nash_eq1()`
   - **Constraint**: Preserves entanglement structure (stays within orbit)

2. **Inter-Orbit Optimization** (Global Search)
   - **Input**: Starting state + objective function (e.g., maximize welfare)
   - **Output**: Optimized state in a different orbit
   - **Method**: Gradient-based optimization (Adam) on MPS parameters
   - **Current implementation**: Gradient ascent loop in `train()`
   - **Freedom**: Can change entanglement structure (moves between orbits)

### Proposed Module Structure

```
nash/
├── core_algorithms/
│   ├── __init__.py
│   ├── intra_orbit.py       # IntraOrbitNashFinder class
│   ├── inter_orbit.py       # InterOrbitOptimizer class
│   └── configs.py           # Configuration classes
├── opt_mps_fiducial_state.py  # Refactored to use core algorithms
├── solver.py                   # Refactored to use core algorithms
└── ...
```

## Detailed Design

### 1. `core_algorithms/intra_orbit.py`

```python
class IntraOrbitNashFinder:
    """
    Finds Nash equilibria within a unitary orbit using differential best response dynamics.

    This algorithm applies local unitary transformations to each player's subsystem
    until no player can improve their payoff through further local unitaries.

    The algorithm preserves the entanglement structure of the input state.
    """

    def __init__(self, config: IntraOrbitConfig):
        self.max_iter = config.max_iter
        self.alpha = config.alpha  # Learning rate for unitary updates
        self.convergence_threshold = config.convergence_threshold
        self.trace_history = config.trace_history

    def find_equilibrium(
        self,
        Psi: list[Tensor],  # MPS tensors
        H: list[Tensor],    # Hamiltonian for each player
    ) -> NashEquilibriumResult:
        """
        Apply differential best response dynamics to find a Nash equilibrium.

        Algorithm:
        1. For each iteration:
           - Compute energy gradient w.r.t. local unitaries for each player
           - Extract unitary updates via SVD of gradient matrix
           - Apply all unitaries simultaneously (synchronous update)
        2. Converge when local exploitability < threshold

        Returns:
            NashEquilibriumResult containing:
            - final_state: Nash equilibrium state
            - energies: Payoffs for each player
            - converged: Whether algorithm converged
            - num_iterations: Number of iterations taken
            - local_exploitability: Final exploitability measure
            - history: Trajectory (if trace_history=True)
        """
        pass

    def compute_local_exploitability(self, Psi, H) -> float:
        """
        Compute local exploitability from infinitesimal unitary perturbations.

        This is the convergence criterion for differential best response.
        """
        pass

    def compute_unitary_update(self, Psi, H, player_idx) -> Tensor:
        """
        Compute unitary update for a single player using gradient + SVD.

        Returns a unitary matrix that increases player's payoff.
        """
        pass
```

### 2. `core_algorithms/inter_orbit.py`

```python
class InterOrbitOptimizer:
    """
    Optimizes quantum states across different unitary orbits using gradient descent.

    This algorithm can change the entanglement structure of the state by directly
    optimizing the MPS tensor parameters.
    """

    def __init__(self, config: InterOrbitConfig):
        self.n_steps = config.n_steps
        self.lr = config.lr
        self.optimizer_type = config.optimizer_type  # 'adam', 'sgd', etc.
        self.objective = config.objective  # 'welfare', 'nash_welfare', etc.
        self.use_nash_refinement = config.use_nash_refinement

        # Reference to intra-orbit finder for refinement
        self.nash_finder = IntraOrbitNashFinder(config.nash_config) if self.use_nash_refinement else None

    def optimize(
        self,
        initial_state: list[Tensor],
        H: list[Tensor],
        objective_fn: Callable = None,
    ) -> OptimizationResult:
        """
        Optimize MPS state to maximize objective function.

        Algorithm:
        1. For each optimization step:
           - Compute objective (e.g., sum of energies)
           - Backpropagate gradients to MPS parameters
           - Update parameters via optimizer (Adam/SGD)
           - [Optional] Refine to Nash equilibrium within new orbit
        2. Return trajectory of states and objective values

        Returns:
            OptimizationResult containing:
            - final_state: Optimized state
            - trajectory: States at each iteration
            - objective_values: Objective function values
            - nash_equilibria: Nash equilibria at each step (if refinement enabled)
        """
        pass

    def compute_objective(self, Psi, H, objective_type='welfare') -> Tensor:
        """
        Compute objective function to be maximized.

        Options:
        - 'welfare': Sum of all players' energies
        - 'nash_welfare': Product of energies (Nash social welfare)
        - 'min_energy': Minimum player energy (egalitarian)
        - 'custom': User-provided objective function
        """
        pass

    def refine_to_nash(self, Psi, H) -> list[Tensor]:
        """
        Refine current state to Nash equilibrium using IntraOrbitNashFinder.

        This projects the gradient-optimized state onto the nearest Nash equilibrium
        within its current orbit.
        """
        if self.nash_finder is None:
            return Psi
        result = self.nash_finder.find_equilibrium(Psi, H)
        return result.final_state
```

### 3. `core_algorithms/configs.py`

```python
@dataclass
class IntraOrbitConfig:
    """Configuration for intra-orbit Nash equilibrium finding."""
    max_iter: int = 10000
    alpha: float = 10.0  # Learning rate for unitary updates
    convergence_threshold: float = 1e-6  # Local exploitability threshold
    trace_history: bool = False
    validate_with_global_expl: bool = True
    global_expl_samples: int = 10000
    global_expl_threshold: float = 1e-3

@dataclass
class InterOrbitConfig:
    """Configuration for inter-orbit optimization."""
    n_steps: int = 2000
    lr: float = 0.01
    optimizer_type: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
    objective: str = 'welfare'  # 'welfare', 'nash_welfare', 'min_energy'

    # Nash refinement settings
    use_nash_refinement: bool = True
    nash_config: IntraOrbitConfig = field(default_factory=IntraOrbitConfig)
    nash_refinement_frequency: int = 1  # Refine every N steps
    max_nash_attempts: int = 20  # Multiple restarts for Nash finder

@dataclass
class HybridOptimizationConfig:
    """Configuration for hybrid optimization (inter + intra)."""
    inter_orbit_config: InterOrbitConfig
    intra_orbit_config: IntraOrbitConfig

    # Data configuration
    mps_bond_dim: int = 2
    initial_state: list[Tensor] | None = None

    # Logging
    use_wandb: bool = True
    wandb_project: str = 'quantum-nash-optimization'
    log_interval: int = 1
```

### 4. Result Data Classes

```python
@dataclass
class NashEquilibriumResult:
    """Result from intra-orbit Nash equilibrium finding."""
    final_state: list[Tensor]
    energies: list[float]
    converged: bool
    num_iterations: int
    local_exploitability: float
    global_exploitability: list[float] | None = None
    history: dict | None = None  # Energy trajectories, etc.

@dataclass
class OptimizationResult:
    """Result from inter-orbit optimization."""
    final_state: list[Tensor]
    trajectory: list[list[Tensor]]
    objective_values: list[float]
    energies_history: list[list[float]]
    nash_equilibria: list[NashEquilibriumResult] | None = None
    entanglement_params: list[np.ndarray] | None = None
```

## Migration Strategy

### Phase 1: Create Core Algorithms Module (No Breaking Changes)

1. Create `core_algorithms/` directory
2. Implement `IntraOrbitNashFinder` by extracting logic from `find_nash_eq()`
3. Implement `InterOrbitOptimizer` by extracting logic from `train()`
4. Keep existing functions as thin wrappers calling the new classes

**Benefit**: New code is cleaner, old code still works

### Phase 2: Refactor Main Scripts

1. Update `opt_mps_fiducial_state.py`:
   ```python
   def train(config: HybridOptimizationConfig):
       optimizer = InterOrbitOptimizer(config.inter_orbit_config)
       result = optimizer.optimize(
           initial_state=get_initial_state(config),
           H=config.H,
       )
       return result
   ```

2. Update `solver.py`:
   ```python
   def solve_nash_equilibrium(Psi, H, config: IntraOrbitConfig):
       finder = IntraOrbitNashFinder(config)
       result = finder.find_equilibrium(Psi, H)
       return result
   ```

3. Update tests to use new APIs

### Phase 3: Add New Capabilities

Once core algorithms are separated, easily add:

1. **Different inter-orbit optimizers**: Not just Adam, but also:
   - CMA-ES for derivative-free optimization
   - Natural gradient methods
   - Trust region methods

2. **Different intra-orbit finders**: Not just differential BR, but also:
   - Fictitious play
   - Replicator dynamics
   - Best response iterations

3. **Hybrid strategies**:
   - Alternating between inter and intra-orbit optimization
   - Multi-start global optimization
   - Curriculum learning (start with simple games)

## Benefits of Refactoring

1. **Conceptual clarity**: The two core algorithms are clearly distinguished
2. **Modularity**: Each algorithm can be tested, improved, and replaced independently
3. **Reusability**: Core algorithms can be used in different contexts (not just `train()`)
4. **Extensibility**: Easy to add new optimization methods or Nash finding algorithms
5. **Better testing**: Each component can be unit tested separately
6. **Scientific clarity**: Aligns code structure with mathematical concepts (orbits)

## Open Questions for Discussion

1. **Naming**: Do you prefer `IntraOrbit`/`InterOrbit` or `LocalSearch`/`GlobalSearch`?
2. **Return types**: Should we use dataclasses or dictionaries for results?
3. **Backward compatibility**: Should we keep old function signatures as aliases?
4. **Validation**: Should Nash refinement validate exploitability by default?
5. **Multi-player generalization**: Should we design for arbitrary number of players from the start?

## Example Usage (After Refactoring)

```python
from core_algorithms import IntraOrbitNashFinder, InterOrbitOptimizer
from core_algorithms.configs import IntraOrbitConfig, InterOrbitConfig

# Example 1: Find Nash equilibrium within current orbit
config = IntraOrbitConfig(alpha=10, convergence_threshold=1e-6)
finder = IntraOrbitNashFinder(config)
result = finder.find_equilibrium(Psi, H)
print(f"Converged: {result.converged}, Exploitability: {result.local_exploitability}")

# Example 2: Optimize across orbits with Nash refinement
inter_config = InterOrbitConfig(
    n_steps=5000,
    lr=3.2e-3,
    use_nash_refinement=True,
    nash_config=IntraOrbitConfig(alpha=10, convergence_threshold=1e-6)
)
optimizer = InterOrbitOptimizer(inter_config)
result = optimizer.optimize(initial_state=Psi_init, H=H)

# Example 3: Hybrid optimization (legacy interface)
hybrid_config = HybridOptimizationConfig(
    inter_orbit_config=inter_config,
    intra_orbit_config=config,
    mps_bond_dim=2,
)
train(hybrid_config)  # Refactored to use core algorithms
```

## Next Steps

1. Review and approve this refactoring plan
2. Decide on naming conventions and API details
3. Implement Phase 1 (core algorithms module)
4. Test with existing experiments
5. Gradually migrate existing scripts (Phase 2)
6. Add new capabilities (Phase 3)
