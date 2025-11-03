
## Executive Summary

This program implements a hybrid optimization algorithm for finding Nash equilibria in quantum games using Matrix Product State (MPS) representations. The algorithm is designed for 3-player quantum games where players share a tripartite quantum state and each player applies local unitaries to maximize their expected payoff.

### Main Algorithm Overview

The optimization consists of two nested phases:

1. **Outer Loop - Gradient Ascent Phase**:
   - Optimizes the MPS representation of the quantum state to maximize total welfare (sum of all players' energies)
   - Uses Adam optimizer with gradient ascent (negated loss) on the MPS tensor parameters
   - Each player's MPS tensor is optimized independently

2. **Inner Loop - Nash Equilibrium Refinement Phase**:
   - After each gradient step, finds a Nash equilibrium via differential best response dynamics
   - Applies local unitary transformations to each player's tensor
   - Verifies global exploitability across a range of alternative strategies
   - Attempts multiple times to find a valid Nash equilibrium (low exploitability)

The algorithm tracks entanglement parameters, welfare metrics, and exploitability throughout training, with optional Weights & Biases (wandb) logging.

### Key Concepts

- **MPS (Matrix Product State)**: A tensor network representation of quantum states that efficiently captures entanglement with bond dimension χ
- **Nash Equilibrium**: A strategy profile where no player can unilaterally improve their payoff
- **Exploitability**: A measure of how much a player can improve by deviating from their current strategy
- **Differential Best Response Dynamics**: An iterative method that applies small unitary updates based on energy gradients
- **Welfare**: The sum of all players' expected payoffs

---

## Configuration Classes

### `DataConfig`
Configuration for Hamiltonian and data storage.

**Attributes:**
- `default_dtype` (torch.dtype): Data type for tensors (default: float64)
- `device` (torch.device): Computation device (CUDA if available, else CPU)
- `H` (List[torch.Tensor]): List of Hamiltonian tensors for each player, shape (2,2,2,2,2,2)
- `H_all_in_one` (torch.Tensor): Stacked Hamiltonian tensor, shape (3,2,2,2,2,2,2)
- `save_dir` (str): Directory to save results (default: 'nash_data')
- `save_name` (str): Filename for saved results (deprecated, now auto-generated)

### `NEFinderConfig`
Configuration for Nash equilibrium finder using differential best response dynamics.

**Attributes:**
- `max_iter` (int): Maximum iterations for best response dynamics (default: 10000)
- `alpha` (float): Learning rate for unitary updates (default: 10)
- `convergence_threshold` (float): Threshold for local exploitability convergence (default: 1e-6)
- `symmetric` (bool): DEPRECATED - Whether to use same random unitary for all players
- `trace_history` (bool): Whether to record energy history during dynamics
- `expl_threshold` (float): Global exploitability threshold to accept Nash equilibrium (default: 1e-3)
- `expl_num_samples` (int): Number of samples for exploitability computation (default: 10000)
- `max_num_attempts` (int): Maximum attempts to find valid Nash equilibrium (default: 20)

### `TrainerConfig`
Configuration for the main training loop.

**Attributes:**
- `start_state` (List[torch.Tensor] | None): Initial MPS tensors (None for random initialization)
- `mps_bond_dim` (int): Bond dimension χ of the MPS (default: 3)
- `n_optimizer_steps` (int): Number of gradient ascent steps (default: 2000)
- `lr` (float): Learning rate for Adam optimizer (default: 0.01)
- `use_wandb` (bool): Enable Weights & Biases logging (default: True)
- `wandb_project` (str): W&B project name (default: 'quantum-nash-optimization')
- `wandb_run_name` (str | None): Custom run name (auto-generated if None)

---

## Core Functions

### `get_default_H`
```python
def get_default_H(option: str = 'H', default_dtype: torch.dtype = torch.float64,
                  device: torch.device = ...) -> List[torch.Tensor] | torch.Tensor
```

Returns the default Hamiltonian for the 3-player quantum game (Prisoner's Dilemma variant).

**Parameters:**
- `option` (str): Return format - 'H' for list, 'H_all_in_one' for stacked tensor
- `default_dtype` (torch.dtype): Data type for tensors
- `device` (torch.device): Computation device

**Returns:**
- List[torch.Tensor] or torch.Tensor: Hamiltonian(s) representing payoff matrices

**Implementation:**
- Creates three diagonal payoff matrices (one per player)
- Each matrix has shape (8,) representing outcomes for 2^3 = 8 possible measurement results
- Reshapes to (2,2,2,2,2,2) for tensor contraction with quantum states
- Payoff structure encodes a quantum Prisoner's Dilemma game

---

### MPS Manipulation Functions

### `get_state_from_tensors`
```python
def get_state_from_tensors(A_list: List[torch.Tensor], bc: str = 'PBC') -> torch.Tensor
```

Converts MPS tensor list into the full quantum state vector (wavefunction).

**Parameters:**
- `A_list` `(List[torch.Tensor])`: MPS tensors of shape` (phys, χ_L, χ_R)`
- `bc` `(str)`: Boundary conditions - 'PBC' (periodic) or 'OBC' (open, not implemented)

**Returns:**
- `torch.Tensor`: Normalized quantum state of shape (2, 2, 2) for 3-player system

**Implementation:**
1. Converts tensors to canonical form using `mps_2form()`
2. Contracts tensors sequentially using Einstein summation
3. For PBC: traces over bond indices to close the loop
4. Normalizes the resulting state vector

**Note:** Currently only supports 3-site systems with PBC.

---

### `normalize_mps_tensor`
```python
def normalize_mps_tensor(A: torch.Tensor) -> torch.Tensor
```

Normalizes an MPS tensor by the dominant eigenvalue of its transfer matrix.

**Parameters:**
- `A` (torch.Tensor): MPS tensor of shape `(phys, χ_L, χ_R)`

**Returns:**
- torch.Tensor: Normalized MPS tensor

**Implementation:**
- Computes the transfer matrix T = Σ_phys A[phys] ⊗ A*[phys]
- Finds the largest eigenvalue λ_max
- Scales tensor by 1/√λ_max to ensure proper normalization in thermodynamic limit

---

### compute_transfer_matrix
```python
def compute_transfer_matrix(A: torch.Tensor) -> torch.Tensor
```

Computes the transfer matrix of an MPS tensor.

**Parameters:**
- `A` (torch.Tensor): MPS tensor of shape (phys, χ_L, χ_R)

**Returns:**
- torch.Tensor: Transfer matrix of shape (χ_L × χ_L, χ_R × χ_R)

**Implementation:**
- Contracts tensor with its conjugate: T = Σ_phys A ⊗ A*
- Reshapes to matrix form for eigenvalue computation
- Used for normalization and computing expectation values in infinite MPS

---

### apply_unitary
```python
def apply_unitary(unitary: torch.Tensor, A: torch.Tensor) -> torch.Tensor
```

Applies a unitary gate to the physical leg of an MPS tensor.

**Parameters:**
- `unitary` (torch.Tensor): Unitary matrix of shape (2, 2)
- `A` (torch.Tensor): MPS tensor of shape (phys, χ_L, χ_R)

**Returns:**
- torch.Tensor: Transformed MPS tensor with same shape

**Implementation:**
- Contracts unitary with physical index: A'[new_phys] = Σ_phys U[new_phys, phys] A[phys]
- Represents local quantum operations (gates) applied by each player

---

## Nash Equilibrium Finding

### find_nash_eq
```python
def find_nash_eq(Psi: List[torch.Tensor], H: List[torch.Tensor],
                 max_iter: int = 10000, alpha: float = 10,
                 convergence_threshold: float = 1e-6, symmetric: bool = False,
                 trace_history: bool = False) -> Dict
```

Finds Nash equilibrium using differential best response dynamics.

**Parameters:**
- `Psi` (List[torch.Tensor]): Current MPS tensors for 3 players
- `H` (List[torch.Tensor]): Hamiltonian tensors for each player
- `max_iter` (int): Maximum number of iterations
- `alpha` (float): Learning rate for unitary updates
- `convergence_threshold` (float): Local exploitability threshold for convergence
- `symmetric` (bool): DEPRECATED parameter
- `trace_history` (bool): Whether to record energy at each iteration

**Returns:**
- Dict with keys:
  - `converged` (bool): Whether dynamics converged
  - `energy` (List[float]): Final energies for each player
  - `state` (List[torch.Tensor]): Final MPS tensors
  - `num_iters` (int): Number of iterations taken
  - `exploitability` (float): Final local exploitability
  - `energy_history` (List[List[float]]): Energy trajectory (if trace_history=True)

**Algorithm:**
1. **For each iteration:**
   - Compute current energies E_old for all players
   - For each player i:
     - Compute gradient dE/dU_i of player i's energy w.r.t. their unitary
     - Construct matrix: M = I - α * (dE/dU_i) / ||dE/dU_i||
     - Extract unitary via SVD: U_i = (UVh)^T* from M = USVh
   - Apply all unitaries simultaneously to MPS tensors
   - Compute new energies E_new

2. **Convergence check:**
   - Local exploitability = Σ_i max(E_new[i] - E_old[i], 0)
   - Converged if local exploitability < threshold

**Key Idea:** Each player simultaneously applies a small unitary rotation that increases their payoff, following the energy gradient on the unitary manifold.

---

## Energy and Exploitability Computation

### compute_energy
```python
def compute_energy(Psi: List[torch.Tensor], H: torch.Tensor) -> torch.Tensor
```

Computes expected payoff (energy) for all players given current state.

**Parameters:**
- `Psi` (List[torch.Tensor]): MPS tensors
- `H` (torch.Tensor): Stacked Hamiltonian of shape (n_players, 2,2,2,2,2,2)

**Returns:**
- torch.Tensor: Energy for each player, shape (n_players,)

**Implementation:**
- Converts MPS to full state ψ
- Computes ⟨ψ|H_i|ψ⟩ for each player i
- Uses Einstein summation for efficient tensor contraction
- Returns real part (energies are real by construction)

---

### batch_compute_energy
```python
def batch_compute_energy(Psi: List[torch.Tensor], H: torch.Tensor,
                         Psi_batch: torch.Tensor, active_site: int) -> torch.Tensor
```

Efficiently computes energies for multiple variations of a single player's tensor.

**Parameters:**
- `Psi` (List[torch.Tensor]): Fixed MPS tensors for all players
- `H` (torch.Tensor): Stacked Hamiltonian
- `Psi_batch` (torch.Tensor): Batch of alternative tensors for active player, shape (batch, phys, χ_L, χ_R)
- `active_site` (int): Index of player whose tensor is being varied

**Returns:**
- torch.Tensor: Energies for each alternative strategy, shape (batch,)

**Implementation:**
- Constructs batch of quantum states with one player's tensor varied
- Computes energies for all batch elements in parallel
- Properly normalizes each state in the batch
- Used for exploitability computation

**Note:** Currently specialized for 3-site systems.

---

### batch_compute_exploitability
```python
def batch_compute_exploitability(Psi: List[torch.Tensor], H: torch.Tensor,
                                  num_samples: int = 1000) -> torch.Tensor
```

Computes global exploitability by testing alternative unitary strategies.

**Parameters:**
- `Psi` (List[torch.Tensor]): Current MPS tensors (proposed Nash equilibrium)
- `H` (torch.Tensor): Stacked Hamiltonian
- `num_samples` (int): Number of alternative unitaries to test per player

**Returns:**
- torch.Tensor: Maximum exploitability for each player, shape (n_players,)

**Algorithm:**
1. Generate `num_samples` single-qubit unitaries via rotation gates:
   - U(θ) = [[cos θ, sin θ], [-sin θ, cos θ]]
   - Sample θ uniformly in [0, π]
2. For each player and each unitary:
   - Apply unitary to player's MPS tensor
   - Compute resulting energy
3. Exploitability[i] = max(E_alternative[i] - E_current[i], 0)

**Interpretation:** How much can each player gain by unilaterally deviating from the current strategy? Low exploitability indicates a good Nash equilibrium.

---

## Entanglement and Analysis

### `compute_ent_params_from_state`
```python
def compute_ent_params_from_state(state: torch.Tensor, option: str = 'I') -> torch.Tensor
```

Computes entanglement parameters characterizing the quantum state structure.

**Parameters:**
- `state` (torch.Tensor): Quantum state of shape (2,2,2) or flattened (8,)
- `option` (str): Return 'I' invariants or 'J' parameters

**Returns:**
- torch.Tensor: Entanglement parameters, shape (5,)

**Computed Parameters:**

**For option='I' (invariants):**
- `I1`: Tr(ρ_1²) - Single-party purity for player 1
- `I2`: Tr(ρ_2²) - Single-party purity for player 2
- `I3`: Tr(ρ_3²) - Single-party purity for player 3
- `I4`: Tr((ρ_1 ⊗ ρ_2) ρ_12) - Two-party correlation measure
- `I5`: |det₃(ψ)|² - Three-party entanglement measure (generalized concurrence)

**For option='J' (derived parameters):**
- `J1, J2, J3`: Transformed purity measures
- `J4`: √I5 - Concurrence
- `J5`: Higher-order correlation measure

**Implementation:**
- Computes reduced density matrices for all subsystems
- Uses Levi-Civita tensor (eps) for determinant computation
- Parameters are entanglement monotones useful for classifying quantum correlations

---

### post_process
```python
def post_process(df: pd.DataFrame | List[Dict]) -> pd.DataFrame
```

Post-processes results DataFrame with derived metrics.

**Parameters:**
- `df` (pd.DataFrame | List[Dict]): Raw results data

**Returns:**
- pd.DataFrame: Enhanced DataFrame with additional columns

**Added Columns:**
- `welfare`: Sum of all players' energies (total payoff)
- `tot_expl`: Sum of all players' exploitabilities
- `ent_params`: Entanglement parameters (I1-I5) for each state

**Usage:** Call this after loading saved results to add analysis metrics.

---

## Main Training Loop

### train
```python
def train(trainer_cfg: TrainerConfig = TrainerConfig(),
          solver_cfg: NEFinderConfig = NEFinderConfig(),
          data_cfg: DataConfig = DataConfig()) -> None
```

Main training loop implementing the hybrid optimization algorithm.

**Parameters:**
- `trainer_cfg` (TrainerConfig): Training hyperparameters
- `solver_cfg` (NEFinderConfig): Nash equilibrium finder configuration
- `data_cfg` (DataConfig): Hamiltonian and data storage configuration

**Algorithm:**

**Initialization:**
1. Create MPS tensors (random or from start_state)
2. Initialize Adam optimizers for each player's tensor
3. Setup Weights & Biases logging (if enabled)

**Main Loop (for n_optimizer_steps):**
1. **Gradient Ascent Phase:**
   - Compute energies E for all players
   - Backpropagate -E (negated for maximization)
   - Update MPS tensors via Adam optimizer

2. **Nash Equilibrium Refinement:**
   - Convert tensors to canonical form
   - Attempt to find Nash equilibrium up to `max_num_attempts` times:
     - Run differential best response dynamics
     - Compute global exploitability
     - Keep best result (lowest exploitability)
     - Stop if exploitability < threshold

3. **Logging and Storage:**
   - Update MPS parameters with best Nash equilibrium found
   - Compute entanglement parameters
   - Record: energies, exploitabilities, state tensors, welfare
   - Log to Weights & Biases (if enabled)

**Output:**
- Saves results to pickle file with auto-generated name:
  - Format: `qpd_opt_chi{χ}_lr{lr}_steps{N}_alpha{α}_expl{ε}_{timestamp}.pkl`
  - Contains DataFrame with full training trajectory

**Key Design Choices:**
- **Simultaneous updates:** All players' gradients computed before any updates applied
- **Multiple attempts:** Nash equilibrium solver may get stuck in local optima, so multiple attempts with best result selection
- **Canonical form:** Converts to canonical MPS form before Nash equilibrium finding for numerical stability

---

## Usage Example

```python
from opt_mps_fiducial_state import train, TrainerConfig, NEFinderConfig, DataConfig

# Configure training
trainer_cfg = TrainerConfig(
    mps_bond_dim=8,           # Bond dimension
    lr=3e-5,                  # Learning rate
    n_optimizer_steps=20000,  # Number of gradient steps
    use_wandb=True            # Enable logging
)

solver_cfg = NEFinderConfig(
    alpha=10,                 # Nash equilibrium learning rate
    expl_threshold=1e-3,      # Exploitability threshold
    max_num_attempts=20       # Max attempts per step
)

data_cfg = DataConfig()       # Use default Hamiltonian

# Run training
train(trainer_cfg=trainer_cfg, solver_cfg=solver_cfg, data_cfg=data_cfg)
```

## Output Data Structure

The saved pickle file contains a pandas DataFrame with the following columns:

- `energy`: List[float] - Energies for each player
- `converged`: bool - Whether Nash dynamics converged
- `state`: np.ndarray - MPS tensors, shape (3, 2, χ, χ)
- `num_iters`: int - Iterations taken by Nash equilibrium finder
- `local_expl`: float - Local exploitability from differential dynamics
- `global_expl`: np.ndarray - Global exploitability for each player
- `state_`: np.ndarray - Full quantum state vector, shape (2,2,2)
- `welfare`: float - Sum of all players' energies
- `tot_expl`: float - Sum of all exploitabilities
- `ent_params`: np.ndarray - Entanglement parameters (I1-I5)

Each row corresponds to one optimization step.

---

## References and Further Reading

This implementation combines concepts from:
- **Matrix Product States (MPS)**: Efficient representation of quantum many-body states
- **Game Theory**: Nash equilibria and exploitability in multi-agent games
- **Quantum Games**: Games where players share entangled quantum resources
- **Differential Game Dynamics**: Continuous-time best response dynamics for finding equilibria

The algorithm aims to find quantum states that maximize welfare while maintaining Nash equilibrium properties, relevant for studying cooperation and entanglement in quantum game theory.

