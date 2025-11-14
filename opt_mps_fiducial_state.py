import os
# Enable MPS fallback for unsupported operations (e.g., linalg_qr)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from misc_torch import *
import numpy as np
import pandas as pd
import torch as t
import einops
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Complex, Float
from dataclasses import dataclass
from tqdm import tqdm
import wandb
from datetime import datetime

from typing import List, Tuple, Dict, Any


def get_default_H(option: str = 'H', default_dtype: t.dtype = t.float32, device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))):
    """
    Returns the default Hamiltonian for the 3-player quantum game (Prisoner's Dilemma variant).

    Args:
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        default_dtype: Data type for tensors (default: float64)
        device: Computation device (CUDA if available, else CPU)

    Returns:
        List[Tensor] or Tensor: Hamiltonian(s) representing payoff matrices
            - If 'H': List of 3 tensors, each shape (2,2,2,2,2,2)
            - If 'H_all_in_one': Single stacked tensor, shape (3,2,2,2,2,2,2)

    Implementation:
        Creates three diagonal payoff matrices (one per player) encoding a quantum
        Prisoner's Dilemma game. Each matrix represents outcomes for 2^3 = 8
        possible measurement results.
    """
    H = [t.diag(t.tensor([6., 3., 3., 0., 10., 6., 6., 2.], dtype=default_dtype, device=device)),
        t.diag(t.tensor([6., 3., 10., 6., 3., 0., 6., 2.], dtype=default_dtype, device=device)),
        t.diag(t.tensor([6., 10., 3., 6., 3., 6., 0., 2.], dtype=default_dtype, device=device))]
    H = [h.reshape(2, 2, 2, 2, 2, 2) for h in H]

    if option == 'H':
        return H
    elif option == 'H_all_in_one':
        H_all_in_one = t.stack(H)
        return H_all_in_one
    else:
        raise ValueError("Invalid option")

@dataclass
class DataConfig:
    default_dtype: t.dtype = t.float32
    device: t.device = t.device('cuda') if t.cuda.is_available() else (t.device('mps') if t.backends.mps.is_available() else t.device('cpu'))
    H = get_default_H(option='H', default_dtype=default_dtype, device=device)
    H_all_in_one = get_default_H(option='H_all_in_one', default_dtype=default_dtype, device=device)
    save_dir: str = 'nash_data'
    save_name: str = 'qpd_opt_results.pkl'

@dataclass
class NEFinderConfig:
    max_iter: int = 10000   
    alpha: float = 10 # learning rate
    convergence_threshold: float = 1e-6 # threshold for convergence in terms of local exploitability
    symmetric: bool = False # DEPRECATED: whether to use the same random unitary for all players
    trace_history: bool = False # whether to record the energy history
    expl_threshold: float = 1e-3 # threshold for global exploitability to accept a Nash equilibrium
    expl_num_samples: int = 10000 # number of samples to compute exploitability
    max_num_attempts: int = 20 # maximum number of attempts to find a Nash equilibrium

@dataclass
class TrainerConfig:
    start_state: list[t.Tensor] | None = None # initial state to start from
    mps_bond_dim: int = 3
    n_optimizer_steps: int = 2000 # number of optimizer steps to take
    lr: float = 0.01 # learning rate
    use_wandb: bool = True # whether to use wandb for logging
    wandb_project: str = 'quantum-nash-optimization' # wandb project name
    wandb_run_name: str | None = None # wandb run name



# Device selection: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
# MPS fallback is enabled above for unsupported operations
if t.cuda.is_available():
    device = t.device('cuda')
    print("Using NVIDIA GPU (CUDA)")
elif t.backends.mps.is_available():
    device = t.device('mps')
    print("Using Apple Silicon GPU (MPS) with CPU fallback for unsupported ops")
else:
    device = t.device('cpu')
    print("Using CPU")
default_dtype = t.float32

def get_random_near_id_unitary(eps=5e-2):
    """
    Generates a unitary matrix close to identity.

    Args:
        eps: Perturbation strength (default: 5e-2)

    Returns:
        Tensor: 2×2 unitary matrix approximately equal to identity

    Implementation:
        U = (1 - ε²/2)I + ε·iσ_y where σ_y is Pauli-Y matrix
    """
    return t.eye(2, dtype=default_dtype, device=device) * t.cos(eps) + t.sin(eps) * t.tensor([[0, 1], [-1, 0]], dtype=default_dtype, device=device)
    

def get_state_from_tensors(A_list: list[t.Tensor], bc: str = 'PBC'):
    """
    Converts MPS tensor list into the full quantum state vector (wavefunction).

    Args:
        A_list: List of MPS tensors of shape (phys, χ_L, χ_R)
        bc: Boundary conditions - 'PBC' (periodic) or 'OBC' (open, not implemented)

    Returns:
        Tensor: Normalized quantum state of shape (2, 2, 2) for 3-player system

    Implementation:
        1. Converts tensors to canonical form using mps_2form()
        2. Contracts tensors sequentially using Einstein summation
        3. For PBC: traces over bond indices to close the loop
        4. Normalizes the resulting state vector

    Note:
        Currently only supports 3-site systems with periodic boundary conditions.
    """
    Psi = mps_2form(A_list)
    L = len(Psi)
    if bc == "PBC":
        psi = Psi[0]
        for A in Psi[1:]:
            psi = einops.einsum(psi, A, "... chi_l bond, phys bond chi_r -> ... phys chi_l chi_r")

        psi = t.diagonal(psi, dim1=-2, dim2=-1).sum(dim=-1)
        MPS_norm = t.sqrt(t.tensordot(psi, psi.conj(), dims=(list(range(L)), list(range(L)))))
    elif bc == "OBC":
        raise NotImplementedError("OBC not implemented yet")
    return psi / MPS_norm

def normalize_mps_tensor(A: t.Tensor):
    """
    Normalizes an MPS tensor by the dominant eigenvalue of its transfer matrix.

    Args:
        A: MPS tensor of shape (phys, χ_L, χ_R)

    Returns:
        Tensor: Normalized MPS tensor with same shape

    Implementation:
        - Computes the transfer matrix T = Σ_phys A[phys] ⊗ A*[phys]
        - Finds the largest eigenvalue λ_max
        - Scales tensor by 1/√λ_max to ensure proper normalization
    """
    T = compute_transfer_matrix(A)
    eigvals = t.linalg.eigvals(T)
    max_eigval = t.abs(eigvals).max()
    A = A / t.sqrt(max_eigval)
    return A

def compute_transfer_matrix(A: t.Tensor):
    """
    Computes the transfer matrix of an MPS tensor.

    Args:
        A: MPS tensor of shape (phys, χ_L, χ_R)

    Returns:
        Tensor: Transfer matrix of shape (χ_L × χ_L, χ_R × χ_R)

    Implementation:
        Contracts tensor with its conjugate: T = Σ_phys A ⊗ A*
        Reshapes to matrix form for eigenvalue computation.
        Used for normalization and computing expectation values in infinite MPS.
    """
    T = einops.einsum(
        A, A.conj(), "phys chi_l chi_r, phys chi_lc chi_rc -> chi_l chi_r chi_lc chi_rc"
    )
    T = einops.rearrange(T, "chi_l chi_r chi_lc chi_rc -> (chi_l chi_lc) (chi_r chi_rc)")
    return T


def apply_unitary(unitary, A):
    """
    Applies a unitary gate to the physical leg of an MPS tensor.

    Args:
        unitary: Unitary matrix of shape (2, 2)
        A: MPS tensor of shape (phys, χ_L, χ_R)

    Returns:
        Tensor: Transformed MPS tensor with same shape

    Implementation:
        Contracts unitary with physical index: A'[new_phys] = Σ_phys U[new_phys, phys] A[phys]
        Represents local quantum operations (gates) applied by each player.
    """
    A = einops.einsum(
        A, unitary, "phys chi_l chi_r, new_phys phys -> new_phys chi_l chi_r"
    )
    return A

def find_nash_eq(Psi, H, max_iter=10000, alpha=10, convergence_threshold=1e-6, symmetric=False, trace_history=False):
    """
    Finds Nash equilibrium using differential best response dynamics.

    Args:
        Psi: List[Tensor] - MPS tensors of shape (phys, χ_L, χ_R) for each player
        H: List[Tensor] - Hamiltonian tensors for each player, shape (2,2,2,2,2,2)
        max_iter: Maximum number of iterations (default: 10000)
        alpha: Learning rate for unitary updates (default: 10)
        convergence_threshold: Local exploitability threshold for convergence (default: 1e-6)
        symmetric: DEPRECATED parameter (not used)
        trace_history: Whether to record energy at each iteration (default: False)

    Returns:
        dict with keys:
            - 'converged' (bool): Whether dynamics converged
            - 'energy' (List[float]): Final energies for each player
            - 'state' (List[Tensor]): Final MPS tensors
            - 'num_iters' (int): Number of iterations taken
            - 'exploitability' (float): Final local exploitability
            - 'energy_history' (List[List[float]]): Energy trajectory (if trace_history=True)

    Algorithm:
        1. For each iteration:
           - Compute current energies E_old for all players
           - For each player: compute gradient dE/dU and extract unitary via SVD
           - Apply all unitaries simultaneously to MPS tensors
           - Compute new energies E_new
        2. Convergence: local exploitability = Σ_i max(E_new[i] - E_old[i], 0) < threshold

    Key Idea:
        Each player simultaneously applies a small unitary rotation that increases
        their payoff, following the energy gradient on the unitary manifold.
    """
    L = len(Psi)
    _iter = 0
    converged = False
    
    # Initialize with random unitaries
    # U0 = t.tensor(np.linalg.qr(np.random.randn(2, 2))[0], dtype=default_dtype, device=device)
    # for i in range(L):
        # U = t.tensor(np.linalg.qr(np.random.randn(2, 2))[0], dtype=default_dtype, device=device)
        # unitary = U if not symmetric else U0
        # unitary = get_random_near_id_unitary()
        # Psi[i] = apply_unitary(unitary, Psi[i])

    if trace_history:
        E_history = []
    

    while _iter <= max_iter and not converged:
        unitaries = []
        psi = get_state_from_tensors(Psi)
        E_old = []
        E_new = []

        # Compute energy BEFORE update and generate unitaries
        for i in range(L):
            j = (i - 1) % L
            k = (i + 1) % L
            dE = t.tensordot(H[i], psi, dims=([4, 5, 3], [1, 2, 0]))
            dE = t.tensordot(psi.conj(), dE, dims=([j, k], [j, k]))
            E_old.append(t.trace(dE).real.item())
            
            # Compute unitary from SVD
            dE = t.eye(2, dtype=default_dtype, device=device) - alpha * dE / t.linalg.norm(dE)
            U, S, Vh = t.linalg.svd(dE)
            unitaries.append((U @ Vh).T.conj())

        if trace_history:
            E_history.append(E_old)

        # Apply unitaries
        for i in range(L):
            Psi[i] = apply_unitary(unitaries[i], Psi[i])

        # Compute energy AFTER update for convergence check
        psi = get_state_from_tensors(Psi)
        for i in range(L):
            j = (i - 1) % L
            k = (i + 1) % L
            dE = t.tensordot(H[i], psi, dims=([4, 5, 3], [1, 2, 0]))
            dE = t.tensordot(psi.conj(), dE, dims=([j, k], [j, k]))
            E_new.append(t.trace(dE).real.item())

        local_max_epl = sum([max(E_new[i] - E_old[i], 0) for i in range(L)])
        converged = local_max_epl < convergence_threshold
        _iter = _iter + 1

    if not converged:
        print(f"Warning: the differential BR dynamics did not converge up to threshold {convergence_threshold}")

    result = {
        'converged': converged,
        'energy': E_new,
        'state': Psi,
        'num_iters': _iter,
        'exploitability': local_max_epl,
    }
    if trace_history:
        result['energy_history'] = E_history

    return result

def compute_energy(Psi: list[t.Tensor], H: t.Tensor):
    """
    Computes expected payoff (energy) for all players given current state.

    Args:
        Psi: List of MPS tensors
        H: Stacked Hamiltonian of shape (n_players, 2, 2, 2, 2, 2, 2)

    Returns:
        Tensor: Energy for each player, shape (n_players,)

    Implementation:
        - Converts MPS to full state ψ
        - Computes ⟨ψ|H_i|ψ⟩ for each player i
        - Uses Einstein summation for efficient tensor contraction
        - Returns real part (energies are real by construction)

    Note: Works for 3-site systems (not thermodynamic limit).
    """
    if isinstance(H, list):
        H = t.stack(H)
        
    psi = get_state_from_tensors(Psi)
    coord_str = 'a1 a2 a3'
    coord_str_conj = 'b1 b2 b3'
    contraction_specification = "".join([coord_str, ', batch ', coord_str, ' ', coord_str_conj, ', ', coord_str_conj, ' -> batch'])
    # print(f"Contraction specification: {contraction_specification}")
    E = einops.einsum(psi, H, psi.conj(), contraction_specification)
    return t.real(E)

def batch_compute_energy(Psi: list[t.Tensor], H: t.Tensor, Psi_batch: Float[t.Tensor, "batch phys chi_l chi_r"], active_site: int):
    """
    Efficiently computes energies for multiple variations of a single player's tensor.

    Args:
        Psi: Fixed MPS tensors for all players
        H: Stacked Hamiltonian tensor
        Psi_batch: Batch of alternative tensors for active player, shape (batch, phys, χ_L, χ_R)
        active_site: Index of player whose tensor is being varied

    Returns:
        Tensor: Energies for each alternative strategy, shape (batch,)

    Implementation:
        - Constructs batch of quantum states with one player's tensor varied
        - Computes energies for all batch elements in parallel
        - Properly normalizes each state in the batch
        - Used for exploitability computation

    Note: Currently specialized for 3-site systems.
    """
    def next_site(site: int):
        return (site + 1) % len(Psi)
    
    H_active_site = H[active_site]
    active_site_inds = f"batch a{active_site} b{active_site} b{next_site(active_site)}"
    inactive_site_inds = [
        f"a{i} b{i} b{next_site(i)}" for i in range(len(Psi)) if i != active_site
    ]
    contraction_spec = "".join(
        [active_site_inds, ", "] + [inactive_site_ind + ", " for inactive_site_ind in inactive_site_inds[:-1]] + [inactive_site_inds[-1]] + 
        ["-> batch "] + [f"a{i} " for i in range(len(Psi))]
    )
    # print(f"Contraction specification: {contraction_spec}")
    psi_batch = einops.einsum(Psi_batch, *[Psi[i] for i in range(len(Psi)) if i != active_site], contraction_spec)
    
    # The following parts work for 3 sites only
    coord_str = 'batch a1 a2 a3'
    coord_str_conj = 'batch b1 b2 b3'
    H_inds = 'a1 a2 a3 b1 b2 b3'
    contraction_spec = f"{coord_str}, {H_inds}, {coord_str_conj} -> batch"
    norm_spec = f"{coord_str}, {coord_str} -> batch"
    E = einops.einsum(psi_batch, H_active_site, psi_batch.conj(), contraction_spec)
    norm = einops.einsum(psi_batch, psi_batch.conj(), norm_spec)
    return E / norm

def batch_compute_exploitability(Psi, H, num_samples: int = 1000):
    """
    Computes global exploitability by testing alternative unitary strategies.

    Args:
        Psi: Current MPS tensors (proposed Nash equilibrium)
        H: Stacked Hamiltonian tensor
        num_samples: Number of alternative unitaries to test per player (default: 1000)

    Returns:
        Tensor: Maximum exploitability for each player, shape (n_players,)

    Algorithm:
        1. Generate num_samples single-qubit rotation unitaries: U(θ) = [[cos θ, sin θ], [-sin θ, cos θ]]
        2. For each player and each unitary:
           - Apply unitary to player's MPS tensor
           - Compute resulting energy
        3. Exploitability[i] = max(E_alternative[i] - E_current[i], 0)

    Interpretation:
        How much can each player gain by unilaterally deviating from the current strategy?
        Low exploitability indicates a good Nash equilibrium.
    """
    params = t.linspace(0, t.pi, num_samples, dtype=default_dtype, device=device)
    batch_U = t.stack([t.cos(params), t.sin(params), -t.sin(params), t.cos(params)])
    batch_U = einops.rearrange(batch_U, "(d1 d2) n_sample -> n_sample d1 d2", d1=2, d2=2)
    batch_E = []
    for site in range(len(Psi)):
        Psi_batch = einops.einsum(
            batch_U, Psi[site], "n_sample d1 d2, d2 chi_l chi_r -> n_sample d1 chi_l chi_r"
        )
        batch_E.append(batch_compute_energy(Psi, H, Psi_batch, site))

    batch_E: Float[t.Tensor, "n_player n_sample"] = t.stack(batch_E)
    original_E = einops.repeat(compute_energy(Psi, H), "n_player -> n_player n_sample", n_sample=num_samples)
    expl = t.real(batch_E - original_E).max(dim=1).values
    return expl

def compute_ent_params_from_state(state, option = 'I'):
    """
    Computes entanglement parameters characterizing the quantum state structure.

    Args:
        state: Quantum state of shape (2,2,2) or flattened (8,)
        option: Return 'I' invariants or 'J' parameters (default: 'I')

    Returns:
        Tensor: Entanglement parameters, shape (5,)

    For option='I' (invariants):
        - I1: Tr(ρ_1²) - Single-party purity for player 1
        - I2: Tr(ρ_2²) - Single-party purity for player 2
        - I3: Tr(ρ_3²) - Single-party purity for player 3
        - I4: Tr((ρ_1 ⊗ ρ_2) ρ_12) - Two-party correlation measure
        - I5: |det₃(ψ)|² - Three-party entanglement (generalized concurrence)

    For option='J' (derived parameters):
        - J1, J2, J3: Transformed purity measures
        - J4: √I5 - Concurrence
        - J5: Higher-order correlation measure

    Implementation:
        Computes reduced density matrices for all subsystems and uses
        Levi-Civita tensor for determinant computation. Parameters are
        entanglement monotones useful for classifying quantum correlations.
    """
    if state.ndim == 1:
        state = state.reshape(2, 2, 2)
    
    rho_1 = einops.einsum(state, state.conj(), 'x i j, y i j -> x y')
    rho_2 = einops.einsum(state, state.conj(), 'i x j, i y j -> x y')
    rho_3 = einops.einsum(state, state.conj(), 'i j x, i j y -> x y')
    rho_12 = einops.einsum(state, state.conj(), 'x1 x2 i, y1 y2 i -> x1 y1 x2 y2')
    rho_12 = einops.rearrange(rho_12, 'x1 y1 x2 y2 -> (x1 x2) (y1 y2)')
    
    I1 = t.trace(t.linalg.matrix_power(rho_1, 2))
    I2 = t.trace(t.linalg.matrix_power(rho_2, 2))
    I3 = t.trace(t.linalg.matrix_power(rho_3, 2))
    # I4 = 3 * np.trace(np.kron(rho_1, rho_2) @ rho_12) - np.trace(np.linalg.matrix_power(rho_1, 3)) - np.trace(np.linalg.matrix_power(rho_2, 3))
    I4 = t.trace(t.kron(rho_1, rho_2) @ rho_12)
    
    eps = t.tensor([[0, 1], [-1, 0]], dtype=default_dtype, device=device)
    det3 = 1/2 * einops.einsum(
        eps, eps, eps, eps, eps, eps, state, state, state, state,
        'i1 j1, i2 j2, k1 l1, k2 l2, i3 k3, j3 l3, i1 i2 i3, j1 j2 j3, k1 k2 k3, l1 l2 l3 ->'
    )
    # I5 = 4 * np.abs(det3) ** 2
    I5 = t.abs(det3) ** 2

    if option == 'I':
        return t.stack([I1, I2, I3, I4, I5])
    elif option == 'J':
        J1 = 1/4 * (1 + I1 - I2 - I3 - 2 * t.sqrt(I5))
        J2 = 1/4 * (1 - I1 + I2 - I3 - 2 * t.sqrt(I5))
        J3 = 1/4 * (1 - I1 - I2 + I3 - 2 * t.sqrt(I5))
        J4 = t.sqrt(I5)
        J5 = 1/4 * (3 - 3 * I1 - 3 * I2 - I3 + 4 * I4 - 2 * t.sqrt(I5))
        return t.stack([J1, J2, J3, J4, J5])
    else:
        raise ValueError("Invalid option")
    
def post_process(df: pd.DataFrame | list[dict]):
    """
    Post-processes results DataFrame with derived metrics.

    Args:
        df: Raw results data (DataFrame or list of dicts)

    Returns:
        DataFrame: Enhanced DataFrame with additional columns

    Added Columns:
        - welfare: Sum of all players' energies (total payoff)
        - tot_expl: Sum of all players' exploitabilities
        - ent_params: Entanglement parameters (I1-I5) for each state

    Usage:
        Call this after loading saved results to add analysis metrics.
    """
    if isinstance(df, list):
        df = pd.DataFrame(df)
    # Assume each entry in 'energy' and 'global_expl' columns is a list or array
    df['welfare'] = df['energy'].apply(lambda x: sum(x))
    df['tot_expl'] = df['global_expl'].apply(lambda x: sum(x))

    # compute the entanglement parameters
    df['ent_params'] = df['state_'].apply(lambda x: compute_ent_params_from_state(x, option='I'))
    return df


def train(trainer_cfg = TrainerConfig(), solver_cfg = NEFinderConfig(), data_cfg = DataConfig()):
    """
    Main training loop implementing the hybrid optimization algorithm.

    Args:
        trainer_cfg: Training hyperparameters (TrainerConfig)
        solver_cfg: Nash equilibrium finder configuration (NEFinderConfig)
        data_cfg: Hamiltonian and data storage configuration (DataConfig)

    Algorithm:
        Initialization:
            1. Create MPS tensors (random or from start_state)
            2. Initialize Adam optimizers for each player's tensor
            3. Setup Weights & Biases logging (if enabled)

        Main Loop (for n_optimizer_steps):
            1. Gradient Ascent Phase:
               - Compute energies E for all players
               - Backpropagate -E (negated for maximization)
               - Update MPS tensors via Adam optimizer

            2. Nash Equilibrium Refinement:
               - Convert tensors to canonical form
               - Attempt to find Nash equilibrium up to max_num_attempts times
               - Keep best result (lowest exploitability)
               - Stop if exploitability < threshold

            3. Logging and Storage:
               - Update MPS parameters with best Nash equilibrium found
               - Compute entanglement parameters
               - Record: energies, exploitabilities, state tensors, welfare
               - Log to Weights & Biases (if enabled)

    Output:
        Saves results to pickle file with auto-generated name:
        Format: qpd_opt_chi{χ}_lr{lr}_steps{N}_alpha{α}_expl{ε}_{timestamp}.pkl
        Contains DataFrame with full training trajectory.

    Key Design Choices:
        - Simultaneous updates: All players' gradients computed before any updates applied
        - Multiple attempts: Nash equilibrium solver may get stuck in local optima
        - Canonical form: Converts to canonical MPS form before Nash finding for stability
    """
    if trainer_cfg.start_state is None:
        print(f"Initial state not specified. Starting from random state with chi = {trainer_cfg.mps_bond_dim}")
        Psi = [t.randn(2, trainer_cfg.mps_bond_dim, trainer_cfg.mps_bond_dim, dtype=default_dtype, device=device) for _ in range(3)]
    else:
        Psi = trainer_cfg.start_state
        assert len(Psi) == 3, "Initial state must have 3 tensors"
        assert all(psi.shape[0] == 2 for psi in Psi), "Initial state must have shape (2, chi, chi)"
        assert all(psi.shape[1] == psi.shape[2] for psi in Psi), "Initial state must have square tensors"
        print(f"Starting from specified initial state with chi = {Psi[0].shape[1]}")

    Psi_params = [nn.Parameter(Psi[i]) for i in range(len(Psi))]
    optimizer_list = [optim.Adam([Psi_params[i]], lr=trainer_cfg.lr) for i in range(len(Psi))]

    # Initialize wandb if enabled
    if trainer_cfg.use_wandb:
        # Generate run name if not provided
        if trainer_cfg.wandb_run_name is None:
            run_name = (
                f"chi{trainer_cfg.mps_bond_dim}_"
                f"lr{trainer_cfg.lr:.0e}_"
                f"steps{trainer_cfg.n_optimizer_steps}_"
                f"alpha{solver_cfg.alpha}_"
                f"expl{solver_cfg.expl_threshold:.0e}"
            )
        else:
            run_name = trainer_cfg.wandb_run_name

        wandb.init(
            project=trainer_cfg.wandb_project,
            name=run_name,
            config={
                'mps_bond_dim': trainer_cfg.mps_bond_dim,
                'n_optimizer_steps': trainer_cfg.n_optimizer_steps,
                'lr': trainer_cfg.lr,
                'ne_max_iter': solver_cfg.max_iter,
                'ne_alpha': solver_cfg.alpha,
                'ne_convergence_threshold': solver_cfg.convergence_threshold,
                'expl_threshold': solver_cfg.expl_threshold,
                'expl_num_samples': solver_cfg.expl_num_samples,
                'max_num_attempts': solver_cfg.max_num_attempts,
            }
        )

    df = []
    for _ in tqdm(range(trainer_cfg.n_optimizer_steps)):
        E = compute_energy(Psi_params, data_cfg.H_all_in_one)
        for i in range(E.shape[0]):
            optimizer_list[i].zero_grad()
            (-E[i]).backward(retain_graph=True)  # Maximize energy by negating the loss
        # First accumulate the gradients, then update the tensors simultaneously
        for i in range(len(Psi)):
            optimizer_list[i].step()
        
        with t.no_grad():
            Psi_canonical_tensors = mps_2form(Psi_params)

            best_result = None
            best_global_expl = None
            best_expl = t.inf
            for trail in range(solver_cfg.max_num_attempts):
                result = find_nash_eq(Psi_canonical_tensors, data_cfg.H, max_iter=solver_cfg.max_iter, alpha=solver_cfg.alpha, convergence_threshold=solver_cfg.convergence_threshold, symmetric=solver_cfg.symmetric, trace_history=solver_cfg.trace_history)
                ne_state = result['state']
                for i in range(len(Psi)):
                    Psi_params[i].data = ne_state[i]

                global_expl = batch_compute_exploitability(ne_state, data_cfg.H_all_in_one, num_samples=solver_cfg.expl_num_samples)
                if global_expl.sum() < best_expl:
                    best_global_expl = global_expl
                    best_expl = global_expl.sum()
                    best_result = result
                if best_expl < solver_cfg.expl_threshold:
                    print(f"Global exploitability {best_expl} is less than threshold {solver_cfg.expl_threshold}. Stopping training.")
                    break
            
            if trail == solver_cfg.max_num_attempts - 1:
                print(f"Did not find a legit NE within {solver_cfg.max_num_attempts} attempts, best exploitability: {best_expl}")
            
            for i in range(len(Psi)):
                Psi_params[i].data = best_result['state'][i]

        ## Logging
        state_ = get_state_from_tensors(best_result['state'])
        print(best_result['exploitability'])
        ent_params = compute_ent_params_from_state(state_, option='I')
        df.append({
            'energy': best_result['energy'],
            'converged': best_result['converged'],
            'state': t.stack(best_result['state']).detach().cpu().numpy(),
            'num_iters': best_result['num_iters'],
            'local_expl': best_result['exploitability'],
            'global_expl': best_global_expl.detach().cpu().numpy(),
            'state_': state_.detach().cpu().numpy(),
            'welfare': sum(best_result['energy']),
            'tot_expl': best_global_expl.sum().detach().cpu().numpy(),
            'ent_params': ent_params.detach().cpu().numpy()
        })

        # Log to wandb if enabled
        if trainer_cfg.use_wandb:
            wandb.log({
                'welfare': sum(best_result['energy']),
                'energy/player_1': best_result['energy'][0],
                'energy/player_2': best_result['energy'][1],
                'energy/player_3': best_result['energy'][2],
                'tot_expl': best_global_expl.sum().item(),
                'ent_params/I1': ent_params[0].item(),
                'ent_params/I2': ent_params[1].item(),
                'ent_params/I3': ent_params[2].item(),
                'ent_params/I4': ent_params[3].item(),
                'ent_params/I5': ent_params[4].item(),
            })

    df = pd.DataFrame(df)
    # df = post_process(df)

    # Generate dataset save name with relevant parameters and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = (
        f"qpd_opt_"
        f"chi{trainer_cfg.mps_bond_dim}_"
        f"lr{trainer_cfg.lr:.0e}_"
        f"steps{trainer_cfg.n_optimizer_steps}_"
        f"alpha{solver_cfg.alpha}_"
        f"expl{solver_cfg.expl_threshold:.0e}_"
        f"{timestamp}.pkl"
    )

    os.makedirs(data_cfg.save_dir, exist_ok=True)
    df.to_pickle(os.path.join(data_cfg.save_dir, save_name))
    print(f"Dataset saved to: {os.path.join(data_cfg.save_dir, save_name)}")

    # Finish wandb run if enabled
    if trainer_cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    data_cfg = DataConfig()
    solver_cfg = NEFinderConfig()
    trainer_cfg = TrainerConfig()
    trainer_cfg.mps_bond_dim = 2
    trainer_cfg.lr = 1e-4 * (8/2) ** 2 # some heuristics for scaling the learning rate...
    trainer_cfg.n_optimizer_steps = 30000
    train(data_cfg=data_cfg, solver_cfg=solver_cfg, trainer_cfg=trainer_cfg)


