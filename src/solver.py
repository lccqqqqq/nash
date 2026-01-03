import os
# Enable MPS fallback for unsupported operations (e.g., linalg_qr)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from tqdm import tqdm
import torch as t
from functools import reduce
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import math
from scipy.optimize import differential_evolution
import einops
from jaxtyping import Float
import pandas as pd
import wandb
import argparse
import pickle
import uuid
import datetime

# Import from refactored src/ modules
try:
    # When imported as a module (e.g., from src.solver import ...)
    from src.mps_utils import (
        apply_unitary,
        to_canonical_form,
        to_comp_basis,
        from_comp_basis,
        get_rand_mps,
        get_rand_state_as_mps,
        get_product_state,
        get_ghz_state,
        apply_random_unitaries,
        test_canonical_form
    )
    from src.game import get_default_3players, get_default_2players, get_default_H, get_default_cyclic_players, H_QPD, get_perturbed_H_QPD
    from src.entanglement import compute_entanglement_params as compute_ent_params_from_state
except ImportError:
    # When run directly as a script (e.g., python src/solver.py)
    from mps_utils import (
        apply_unitary,
        to_canonical_form,
        to_comp_basis,
        from_comp_basis,
        get_rand_mps,
        get_rand_state_as_mps,
        get_product_state,
        get_ghz_state,
        apply_random_unitaries,
        test_canonical_form
    )
    from game import get_default_3players, get_default_2players, get_default_H, get_default_cyclic_players, H_QPD, get_perturbed_H_QPD
    from entanglement import compute_entanglement_params as compute_ent_params_from_state


# Pre-computed Pauli matrices and tensor products for unitary perturbations
# Complex generators (for backward compatibility with complex Hamiltonians)
PAULIS = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),      # σ_x
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),   # σ_y
    np.array([[1, 0], [0, -1]], dtype=np.complex128)      # σ_z
]

PAULI_TENSORS_2Q = np.array([
    np.kron(PAULIS[i], PAULIS[j])
    for i in range(3) for j in range(3)
])  # Shape: (9, 4, 4)

# Real generators for real Hamiltonians - use only those where exp(iH) is real
# These are combinations involving Y: X⊗Y, Y⊗X, Y⊗Z, Z⊗Y
# We work with iH directly (which is real and antisymmetric)
# Key insight: iσ_y = [[0, 1], [-1, 0]] is real!
iY = np.array([[0, 1], [-1, 0]], dtype=np.float64)  # iσ_y is real!
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.float64)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.float64)

REAL_PAULI_GENERATORS = np.array([
    np.kron(sigma_x, iY),   # i(X⊗Y)
    np.kron(iY, sigma_x),   # i(Y⊗X)
    np.kron(iY, sigma_z),   # i(Y⊗Z)
    np.kron(sigma_z, iY),   # i(Z⊗Y)
], dtype=np.float64)  # Shape: (4, 4, 4)


def is_mps_format(state):
    """Check if state is MPS format (list of tensors) or computational basis (single array)."""
    return isinstance(state, list)


# Functions that compute the Nash equilibrium (using a local algorithm) and verify by computing the exploitability with differential evolution. This is doable because we are dealing with a small parameter space, and considering deviation only in the exp(iY) direction.

def apply_u(u, psi, idx):
    l = len(u.shape)//2
    psi = np.tensordot(u, psi, axes=(list(range(l)), idx))
    return np.moveaxis(psi, list(range(l)), idx)

# Depracated... does not need with random initializations
def kick_with_u(Psi):
    """
    Apply random single-qubit unitaries to each site of the MPS.

    The unitaries are real (orthogonal) or complex (unitary) depending on the dtype of Psi.
    This keeps the state on the same orbit.
    """
    L = len(Psi)
    # Detect dtype from the first tensor
    dtype = Psi[0].dtype
    is_complex = np.iscomplexobj(Psi[0])

    for i in range(L):
        if is_complex:
            # Generate random complex matrix for complex dtype
            random_matrix = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(dtype)
        else:
            # Generate random real matrix for real dtype
            random_matrix = np.random.randn(2, 2).astype(dtype)

        # QR decomposition gives unitary (complex) or orthogonal (real) matrix
        U = np.linalg.qr(random_matrix)[0]
        Psi[i] = apply_unitary(U.T.conj(), Psi[i])
    return Psi

def compute_exploitability(psi, H, player_idx, maxiter=300, seed=42, real_strategies=True):
    """
    Compute exploitability of a quantum state for a given player using differential evolution
    to search over all single-qubit unitaries in SU(2).

    Args:
        psi: Quantum state as ndarray with shape (2,2,...,2) for L qubits
        H: List of Hamiltonian tensors for each player
        player_idx: Index of the player to compute exploitability for
        maxiter: Maximum iterations for differential evolution (default: 300)
        seed: Random seed for differential evolution (default: 42)
        real_strategies: Whether to use real strategies (default: True)
    Returns:
        exploitability: Maximum payoff gain from single-qubit unitary deviation
    """
    L = psi.ndim

    def uni_dev_payoff(alpha_vec):
        alpha = alpha_vec[0]
        if real_strategies:
            theta = math.pi / 2
            phi = math.pi / 2
        else:
            theta = alpha_vec[1]
            phi = alpha_vec[2]

        nx = math.sin(theta) * math.cos(phi)
        ny = math.sin(theta) * math.sin(phi)
        nz = math.cos(theta)

        # Correct SU(2) parametrization: U = cos(α)I + i·sin(α)(n·σ)
        # This is equivalent to exp(i·α·n·σ)
        if real_strategies: # real inputs
            unitary = np.eye(2, dtype=np.float64) * math.cos(alpha) + math.sin(alpha) * np.array(
                [[0, 1], [-1, 0]]
            )
        else: # complex inputs
            unitary = np.eye(2, dtype=np.complex128) * math.cos(alpha) + 1j * math.sin(alpha) * (
                nx * PAULIS[0] + ny * PAULIS[1] + nz * PAULIS[2]
            )

        psi_dev = apply_u(unitary, psi, [player_idx])
        dE = np.tensordot(H[player_idx], psi_dev, axes=([L+j for j in range(L)], [j for j in range(L)]))
        dE = np.tensordot(psi_dev.conj(), dE, axes=([j for j in range(L) if j != player_idx], [j for j in range(L) if j != player_idx]))
        return -float(np.trace(dE).real)

    result = differential_evolution(
        uni_dev_payoff,
        bounds=[(0, math.pi)] if real_strategies else [(0, math.pi), (0, math.pi), (0, 2*math.pi)],
        maxiter=maxiter,
        seed=seed,
        atol=1e-6,
        tol=1e-6,
    )

    plain_payoff = uni_dev_payoff(np.array([0])) if real_strategies else uni_dev_payoff(np.array([0, 0, 0]))
    return -result.fun + plain_payoff

def find_nash_eq1(
    Psi: list[np.ndarray] | np.ndarray, # allowing for both MPS and computational basis input
    H: list[np.ndarray],
    max_iter: int = 10000,
    alpha: float = 0.01,
    convergence_threshold: float = 1e-7,
    expl_threshold: float = 5e-4,
    use_tqdm: bool = False,
    expl_check_interval: int = 10,
    return_history: bool = False,
    expl_maxiter: int = 300,
    expl_seed: int = 42,
    real_strategies: bool = True,
):
    # Convert types to ndarray
    if isinstance(Psi, list) and isinstance(Psi[0], t.Tensor):
        Psi = [p.cpu().numpy() for p in Psi]
    if isinstance(H[0], t.Tensor):
        H = [h.cpu().numpy() for h in H]

    if isinstance(Psi, list):
        L = len(Psi)
    else:
        L = int(np.log2(Psi.shape[0]))
    Es = []
    psi_list = [] if return_history else None
    Psi_list = [] if return_history else None
    local_converged = False
    global_converged = False
    expl_list = []
    for n in tqdm(range(max_iter), disable=not use_tqdm):
        if isinstance(Psi, list):
            psi = to_comp_basis(Psi).reshape([2] * L)
        else:
            psi = Psi.reshape([2] * L)
        unitaries = []
        E = []
        for i in range(L):
            dE = np.tensordot(H[i], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE, axes=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))

            E.append(np.trace(dE).real)
            dE = np.eye(2, dtype=dE.dtype) - alpha * dE

            Y, _, Z = np.linalg.svd(dE)
            unitaries.append((Y @ Z).T if np.isrealobj(dE) else (Y @ Z).T.conj())

        Es.append(np.array(E))
        if return_history:
            psi_list.append(psi)
            Psi_list.append(Psi)
        for i in range(L):
            # Here the convention is made sure to be the same as in `apply_u`
            if isinstance(Psi, list):
                Psi[i] = apply_unitary(unitaries[i].T, Psi[i])
            else:
                Psi = reduce(np.kron, unitaries) @ Psi
        
        if n > 2 and not local_converged:
            local_converged = sum([abs(E[i] - Es[-2][i]) for i in range(L)]) < convergence_threshold
            if local_converged:
                print(f"Converged to Nash state at iteration {n}")


        if n % expl_check_interval == 0:
            expl = [compute_exploitability(psi, H, i, maxiter=expl_maxiter, seed=expl_seed, real_strategies=real_strategies) for i in range(L)]
            expl_list.append(expl)
            if sum(expl) < expl_threshold:
                global_converged = True
                break


    result = {
        'nash_state': local_converged,
        'nash_equilibrium': global_converged,
        'energy': np.stack(Es) if return_history else Es[-1],  # Only return final energy if not tracking history
        'state': psi_list if return_history else psi,
        'state_': Psi_list if return_history else Psi,
        'num_iters': n,
        'expl': np.array(expl_list) if return_history else expl_list[-1],
    }

    return result

def find_nash_eq1_with_retry(
    Psi: list[np.ndarray] | np.ndarray,
    H: list[np.ndarray],
    max_iter: int,
    base_alpha: float,
    max_alpha: float,
    max_retries: int,
    expl_check_interval: int,
    expl_maxiter: int,
    real_strategies: bool,
    return_history: bool = False,
):
    """
    Try to find Nash equilibrium with increasing learning rates on failure.

    If NE finding fails, retry with progressively higher learning rates up to max_alpha.
    This addresses premature termination issues in the Nash solver.

    Args:
        Psi: MPS or computational basis state
        H: List of Hamiltonian tensors
        max_iter: Max iterations for Nash solver
        base_alpha: Initial learning rate (often the current working LR)
        max_alpha: Maximum learning rate to try
        max_retries: Number of retry attempts with increasing LRs
        expl_check_interval: Check exploitability every N iterations
        expl_maxiter: Max iterations for exploitability computation
        real_strategies: Whether to use real strategies for exploitability
        return_history: Whether to return full history

    Returns:
        result: dict from find_nash_eq1
        success: bool indicating if Nash equilibrium was found
        final_alpha: The learning rate that successfully found NE (or last tried LR if failed)
    """
    current_alpha = base_alpha

    for retry_count in range(max_retries):
        result = find_nash_eq1(
            Psi, H,
            max_iter=max_iter,
            alpha=current_alpha,
            expl_check_interval=expl_check_interval,
            expl_maxiter=expl_maxiter,
            real_strategies=real_strategies,
            return_history=return_history
        )

        if result['nash_equilibrium']:
            if retry_count > 0:
                print(f"Success on retry {retry_count + 1} with LR {current_alpha:.4f}")
            return result, True, current_alpha

        # Failed - increase LR for next retry
        if retry_count < max_retries - 1:
            current_alpha = base_alpha + ((retry_count + 1) / max_retries) * (max_alpha - base_alpha)
            print(f"Retry {retry_count + 1}/{max_retries} with higher learning rate: {current_alpha:.4f}")

    # All retries failed
    return result, False, current_alpha

def perturb_state(Psi: list[t.Tensor] | list[np.ndarray] | np.ndarray, lr: float = 0.01, site: int = 0, method: str = 'schmidt'):
    """
    Perturb the state using one of two methods:

    - 'schmidt': Left-canonicalize and perturb singular values at specified bond
    - 'unitary': Apply random two-site unitary via Cayley transform

    Supports both MPS (list of tensors) and computational basis (single array) inputs.
    Output format matches input format.

    Returns:
        For 'schmidt': (new_Psi, original_S, batch_perturbed_S)
        For 'unitary': (new_Psi, original_coefs, batch_coefs)
    """
    # Detect input format
    input_is_mps = is_mps_format(Psi)

    if input_is_mps:
        # Handle PyTorch tensors
        if isinstance(Psi[0], t.Tensor):
            Psi_mps = [p.cpu().numpy() for p in Psi]
        else:
            Psi_mps = Psi
    else:
        # Convert computational basis to MPS
        L = int(np.log2(Psi.size))
        Psi_mps = from_comp_basis(Psi, L=L)

    Psi_batch, original_param, batch_perturbed_param = batch_perturb(Psi_mps, batch_size=1, lr=lr, site=site, method=method)
    new_Psi_mps = [Psi_batch[i][0] for i in range(len(Psi_batch))]

    # Convert back to original format if needed
    if not input_is_mps:
        new_Psi = to_comp_basis(new_Psi_mps)
    else:
        new_Psi = new_Psi_mps

    return new_Psi, original_param, batch_perturbed_param

def batch_perturb(Psi: list[t.Tensor] | list[np.ndarray] | np.ndarray, batch_size: int = 100, lr: float = 0.01, site: int = 0, method: str = 'schmidt'):
    """
    Perturb the MPS state using one of two methods:

    - 'schmidt': Left-canonicalize and perturb singular values at specified bond (original method)
    - 'unitary': Apply random two-site unitary via Cayley transform with Pauli coefficients
                 For real inputs: uses 4 generators (X⊗Y, Y⊗X, Y⊗Z, Z⊗Y) that preserve real dtype
                 For complex inputs: uses all 9 two-qubit Pauli generators

    Supports both MPS (list of tensors) and computational basis (single array) inputs.
    Output format matches input format.

    Args:
        Psi: MPS state (list of tensors) or computational basis state (array of 2^L elements)
        batch_size: Number of perturbed states to generate
        lr: Learning rate (perturbation strength)
        site: Site index where perturbation is applied
        method: 'schmidt' or 'unitary'

    Returns:
        For MPS input:
            For 'schmidt': (Psi_batch, original_S, batch_perturbed_S) where Psi_batch is list of batched MPS tensors
            For 'unitary': (Psi_batch, original_coefs, batch_coefs)
        For comp basis input:
            (Psi_batch, original_param, batch_perturbed_param) where Psi_batch is stacked array (batch_size, 2^L)
    """
    # Detect input format
    input_is_mps = is_mps_format(Psi)

    # Convert to MPS numpy format
    if input_is_mps:
        if isinstance(Psi[0], t.Tensor):
            Psi_np = [p.cpu().numpy() for p in Psi]
        else:
            Psi_np = Psi
    else:
        # Convert computational basis to MPS
        L = int(np.log2(Psi.size))
        Psi_np = from_comp_basis(Psi, L=L)

    if method == 'schmidt':
        # Original Schmidt value perturbation method
        d_phys = Psi_np[0].shape[0]

        # First, repeat everything...
        Psi_batch = [einops.repeat(psi, '... -> batch ...', batch=batch_size) for psi in Psi_np]

        # Track singular values from the perturbed site
        original_S = None
        batch_perturbed_S = None

        psi = Psi_batch[0]
        for j in range(len(Psi_batch)):
            # left-canonicalize the state
            psi_grouped = einops.rearrange(psi, 'batch d_phys chi_l chi_r -> batch (d_phys chi_l) chi_r')
            U, S, Vh = np.linalg.svd(psi_grouped, full_matrices=False)

            chi_l = psi.shape[2]  # Get chi_l for rearrange
            Psi_batch[j] = einops.rearrange(U, 'batch (d_phys chi_l) chi_r -> batch d_phys chi_l chi_r', d_phys=d_phys, chi_l=chi_l)

            if j < len(Psi_batch) - 1:
                # Now we only perturb the singular values at one particular bond according to the function input
                if j == site:
                    # Save original singular values before perturbation
                    original_S = S.copy()

                    # Generate random perturbation using numpy
                    batch_perturbed_S = S + np.random.randn(*S.shape).astype(S.dtype) * lr
                    # Normalize
                    batch_perturbed_S /= np.sqrt(np.sum(batch_perturbed_S**2, axis=-1, keepdims=True))

                psi = einops.einsum(
                    batch_perturbed_S if j == site else S, Vh, Psi_batch[j+1],
                    'batch bond_r, batch bond_r chi_l, batch d_phys chi_l chi_r -> batch d_phys bond_r chi_r'
                )

        # Convert back to computational basis if input was comp basis
        if not input_is_mps:
            Psi_batch_comp = []
            for b in range(batch_size):
                Psi_sample = [Psi_batch[s][b] for s in range(len(Psi_batch))]
                Psi_batch_comp.append(to_comp_basis(Psi_sample))
            return np.stack(Psi_batch_comp), original_S, batch_perturbed_S

        return Psi_batch, original_S, batch_perturbed_S

    elif method == 'unitary':
        # Two-site unitary perturbation method using Cayley transform
        L = len(Psi_np)
        site_next = site + 1 # Open boundary conditions
        assert site_next < L, "Site next is out of bounds"

        # Auto-detect whether to use real or complex generators
        use_real_generators = np.isrealobj(Psi_np[0]) and all(np.isrealobj(Psi_np[i]) for i in range(len(Psi_np)))

        if use_real_generators:
            num_generators = 4
            generators = REAL_PAULI_GENERATORS.astype(Psi_np[0].dtype)  # Match input dtype
        else:
            num_generators = 9
            generators = PAULI_TENSORS_2Q

        # Make a copy for the batch
        Psi_batch = [einops.repeat(psi, '... -> batch ...', batch=batch_size) for psi in Psi_np]

        # Group the two adjacent sites
        psi_grouped = einops.einsum(
            Psi_batch[site], Psi_batch[site_next],
            'batch d1 chi_l chi_m, batch d2 chi_m chi_r -> batch d1 d2 chi_l chi_r'
        )
        psi_grouped = einops.rearrange(psi_grouped, 'batch d1 d2 chi_l chi_r -> batch (d1 d2) chi_l chi_r')

        # Generate random coefficients
        coefs_batch = np.random.randn(batch_size, num_generators)
        coefs_batch = coefs_batch / np.linalg.norm(coefs_batch, axis=1, keepdims=True)
        original_coefs = np.zeros((batch_size, num_generators))

        if use_real_generators:
            # Real case: generators are already iH (real, antisymmetric)
            # Construct batch of generators
            iH_batch = np.einsum('bk,kij->bij', coefs_batch, generators)

            # Cayley transform: U = (I + λiH/2) @ inv(I - λiH/2)
            # Since iH is real, U is real
            I4 = np.eye(4, dtype=Psi_np[0].dtype)
            numerator = I4[None, :, :] + lr * iH_batch / 2
            denominator = I4[None, :, :] - lr * iH_batch / 2
            U_batch = numerator @ np.linalg.inv(denominator)
        else:
            # Complex case: generators are Pauli matrices (Hermitian)
            # Construct anti-Hermitian generator iH
            H_batch = np.einsum('bk,kij->bij', coefs_batch, generators)

            # Cayley transform: U = (I + iλH/2) @ inv(I - iλH/2)
            I4 = np.eye(4, dtype=np.complex128)
            numerator = I4[None, :, :] + 1j * lr * H_batch / 2
            denominator = I4[None, :, :] - 1j * lr * H_batch / 2
            U_batch = numerator @ np.linalg.inv(denominator)

        # Apply the unitary
        psi_new_grouped = einops.einsum(
            U_batch, psi_grouped,
            'batch i j, batch j chi_l chi_r -> batch i chi_l chi_r'
        )

        psi_new_grouped = einops.rearrange(psi_new_grouped, 'batch (d1 d2) chi_l chi_r -> batch d1 d2 chi_l chi_r', d1=2, d2=2)
        psi_new_grouped_2d = einops.rearrange(psi_new_grouped, 'batch d1 d2 chi_l chi_r -> batch (d1 chi_l) (d2 chi_r)')

        # QR decomposition
        # m, n = psi_new_grouped_2d.shape[1], psi_new_grouped_2d.shape[2]
        # Q = np.zeros((batch_size, m, min(m, n)), dtype=psi_new_grouped_2d.dtype) # (batch (d1 chi_l) chi_m)
        # R = np.zeros((batch_size, min(m, n), n), dtype=psi_new_grouped_2d.dtype) # (batch chi_m (d2 chi_r))

        # for b in range(batch_size):
        #     Q[b], R[b] = np.linalg.qr(psi_new_grouped_2d[b])

        Q, R = np.linalg.qr(psi_new_grouped_2d)

        # Reshape Q back to MPS tensor at site
        Psi_batch[site] = einops.rearrange(
            Q,
            'batch (d chi_l) chi_m -> batch d chi_l chi_m',
            d=2
        )

        # R goes to the next site
        Psi_batch[site_next] = einops.rearrange(
            R,
            'batch chi_m (d chi_r) -> batch d chi_m chi_r',
            d=2
        )

        # Convert back to computational basis if input was comp basis
        if not input_is_mps:
            Psi_batch_comp = []
            for b in range(batch_size):
                Psi_sample = [Psi_batch[s][b] for s in range(len(Psi_batch))]
                Psi_batch_comp.append(to_comp_basis(Psi_sample))
            return np.stack(Psi_batch_comp), original_coefs, coefs_batch

        return Psi_batch, original_coefs, coefs_batch

    else:
        raise ValueError(f"Unknown perturbation method: {method}. Must be 'schmidt' or 'unitary'.")

def estimate_gradient_ols(dX, dy, lam=0.0):
    """
    Estimate the gradient of a function f(x) by least squares.

    Uses SVD-based least squares solver for robustness to singular/near-singular cases.
    When the system is underdetermined (fewer samples than dimensions), returns the
    minimum-norm solution.

    Args:
        dX: np.ndarray of shape (n_samples, n_dims) - Perturbation directions
        dy: np.ndarray of shape (n_samples,) - Corresponding function changes
        lam: float - Ridge regularization parameter (optional, default: 0.0)
                     If > 0, uses normal equations with regularization instead of lstsq

    Returns:
        np.ndarray of shape (n_dims,) - Gradient estimate
    """
    n_samples, n_dims = dX.shape

    # Check if we have enough samples (warn if underdetermined)
    if n_samples < n_dims:
        print(f"  Warning: Only {n_samples} valid samples for {n_dims}-dimensional gradient. "
              f"Estimate may be unreliable.")

    # Use regularized normal equations if lam > 0, otherwise use lstsq
    if lam > 0:
        # Ridge regression: solve (dX.T @ dX + λI) @ g = dX.T @ dy
        A = dX.T @ dX + lam * np.eye(n_dims)
        b = dX.T @ dy
        try:
            g_hat = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Should not happen with regularization, but fallback to lstsq
            print("  Warning: Regularized solve failed. Falling back to lstsq.")
            g_hat, residuals, rank, s = np.linalg.lstsq(dX, dy, rcond=None)
    else:
        # Direct least squares using SVD (robust to singular matrices)
        g_hat, residuals, rank, s = np.linalg.lstsq(dX, dy, rcond=None)

        # Diagnostic: check if system is rank-deficient
        if rank < min(n_samples, n_dims):
            print(f"  Warning: Rank-deficient system (rank {rank}/{min(n_samples, n_dims)}). "
                  f"Using minimum-norm solution.")

    return g_hat

def update_state(Psi, S_grad_est_proj, lr, site):
    """Apply targeted, controlled perturbation to the state.

    Supports both MPS (list of tensors) and computational basis (single array) inputs.
    Output format matches input format.
    """
    # Detect input format
    input_is_mps = is_mps_format(Psi)

    if input_is_mps:
        Psi_mps = to_canonical_form(Psi, form='B')
        if isinstance(Psi_mps[0], t.Tensor):
            Psi_mps = [p.numpy() for p in Psi_mps]
    else:
        # Convert computational basis to MPS
        L = int(np.log2(Psi.size))
        Psi_mps = from_comp_basis(Psi, L=L)
        Psi_mps = to_canonical_form(Psi_mps, form='B')

    L = len(Psi_mps)
    psi = Psi_mps[0]
    d_phys = psi.shape[0]
    for j in range(L):
        psi_grouped = einops.rearrange(
            psi, 'd_phys chi_l chi_r -> (d_phys chi_l) chi_r'
        )
        U, S, Vh = np.linalg.svd(psi_grouped, full_matrices=False)
        Psi_mps[j] = einops.rearrange(U, '(d_phys chi_l) chi_r -> d_phys chi_l chi_r', d_phys=d_phys)


        if j < L - 1:
            if j == site:
                S = S + lr * S_grad_est_proj
                S = S / np.linalg.norm(S)

            psi = einops.einsum(
                S, Vh, Psi_mps[j+1],
                'bond_r, bond_r chi_l, d_phys chi_l chi_r -> d_phys bond_r chi_r'
            )

    Psi_out = to_canonical_form(Psi_mps, form='B')

    # Convert back to original format if needed
    if not input_is_mps:
        return to_comp_basis(Psi_out)
    return Psi_out


def update_state_unitary(Psi, coef_grad_est, lr, site):
    """
    Apply targeted unitary update to the state using Pauli coefficient gradients.

    Auto-detects whether state is real or complex:
    - Real state: Uses 4 generators (X⊗Y, Y⊗X, Y⊗Z, Z⊗Y), preserves real dtype
    - Complex state: Uses all 9 two-qubit Pauli generators

    Supports both MPS (list of tensors) and computational basis (single array) inputs.
    Output format matches input format.

    Args:
        Psi: MPS state (list of tensors) or computational basis state (array of 2^L elements)
        coef_grad_est: Gradient estimate in Pauli coefficient space
                       Shape (4,) for real states, (9,) for complex states
        lr: Learning rate
        site: Site index where update is applied

    Returns:
        Updated state in same format as input, in canonical form 'B' if MPS
    """
    # Detect input format
    input_is_mps = is_mps_format(Psi)

    if input_is_mps:
        Psi_mps = to_canonical_form(Psi, form='B')
        if isinstance(Psi_mps[0], t.Tensor):
            Psi_mps = [p.numpy() for p in Psi_mps]
    else:
        # Convert computational basis to MPS
        L_qubits = int(np.log2(Psi.size))
        Psi_mps = from_comp_basis(Psi, L=L_qubits)
        Psi_mps = to_canonical_form(Psi_mps, form='B')

    L = len(Psi_mps)
    site_next = (site + 1) % L  # Periodic boundary conditions

    # Auto-detect whether to use real or complex generators
    use_real_generators = np.isrealobj(Psi_mps[0]) and all(np.isrealobj(Psi_mps[i]) for i in range(L))

    if use_real_generators:
        num_generators = 4
        generators = REAL_PAULI_GENERATORS.astype(Psi_mps[0].dtype)
    else:
        num_generators = 9
        generators = PAULI_TENSORS_2Q

    # Validate gradient dimension
    assert coef_grad_est.shape[0] == num_generators, \
        f"Gradient shape {coef_grad_est.shape} doesn't match expected num_generators={num_generators}"

    # Normalize the gradient to get update direction
    coef_update = coef_grad_est / (np.linalg.norm(coef_grad_est) + 1e-10)

    if use_real_generators:
        # Real case: generators are already iH
        iH = np.einsum('k,kij->ij', coef_update, generators)

        # Cayley transform with real arithmetic
        I4 = np.eye(4, dtype=Psi_mps[0].dtype)
        numerator = I4 + lr * iH / 2
        denominator = I4 - lr * iH / 2
        U = numerator @ np.linalg.inv(denominator)
    else:
        # Complex case: generators are Pauli matrices
        H = np.einsum('k,kij->ij', coef_update, generators)

        # Cayley transform with complex arithmetic
        I4 = np.eye(4, dtype=np.complex128)
        numerator = I4 + 1j * lr * H / 2
        denominator = I4 - 1j * lr * H / 2
        U = numerator @ np.linalg.inv(denominator)

    # Group the two adjacent sites
    psi_grouped = einops.einsum(
        Psi_mps[site], Psi_mps[site_next],
        'd1 chi_l chi_m, d2 chi_m chi_r -> d1 d2 chi_l chi_r'
    )
    psi_grouped = einops.rearrange(psi_grouped, 'd1 d2 chi_l chi_r -> (d1 d2) chi_l chi_r')

    # Apply the unitary
    psi_new_grouped = einops.einsum(
        U, psi_grouped,
        'i j, j chi_l chi_r -> i chi_l chi_r'
    )
    psi_new_grouped = einops.rearrange(psi_new_grouped, '(d1 d2) chi_l chi_r -> d1 d2 chi_l chi_r', d1=2, d2=2)

    # Restore to canonical form via QR decomposition
    psi_new_grouped_2d = einops.rearrange(
        psi_new_grouped,
        'd1 d2 chi_l chi_r -> (d1 chi_l) (d2 chi_r)'
    )

    Q, R = np.linalg.qr(psi_new_grouped_2d)
    # Now the dimensions are ((d1 chi_l) chi_m) and (chi_m (d2 chi_r))

    # Reshape Q back to MPS tensor at site
    Psi_mps[site] = einops.rearrange(
        Q,
        '(d chi_l) chi_m -> d chi_l chi_m',
        d=2
    )

    # R goes to the next site
    Psi_mps[site_next] = einops.rearrange(
        R,
        'chi_m (d chi_r) -> d chi_m chi_r',
        d=2
    )

    Psi_out = to_canonical_form(Psi_mps, form='B')

    # Convert back to original format if needed
    if not input_is_mps:
        return to_comp_basis(Psi_out)
    return Psi_out


def compute_D(psi: np.ndarray, i: int, j: int):
    """
    Compute the D matrix determinant for a pair of qubits (4-qubit states only).

    Helper function for 4-qubit SLOCC invariants.
    Uses precomputed formulas for efficiency.

    Args:
        psi: 4-qubit state vector (16 elements)
        i, j: Qubit pair indices (0-3)

    Returns:
        Determinant of the D matrix for qubits (i, j)
    """
    a = psi.flatten()

    # Dispatch table for all 6 possible pairs
    # Maps sorted (i,j) pairs to their corresponding D matrix computation
    pair = tuple(sorted([i, j]))

    if pair == (0, 1):  # xy
        D = np.array([
            [-a[1]*a[2] + a[0]*a[3], a[3]*a[4] - a[2]*a[5] - a[1]*a[6] + a[0]*a[7], -a[5]*a[6] + a[4]*a[7]],
            [a[3]*a[8] - a[2]*a[9] - a[1]*a[10] + a[0]*a[11], a[7]*a[8] - a[6]*a[9] - a[5]*a[10] + a[4]*a[11] + a[3]*a[12] - a[2]*a[13] - a[1]*a[14] + a[0]*a[15], a[7]*a[12] - a[6]*a[13] - a[5]*a[14] + a[4]*a[15]],
            [-a[9]*a[10] + a[8]*a[11], a[11]*a[12] - a[10]*a[13] - a[9]*a[14] + a[8]*a[15], -a[13]*a[14] + a[12]*a[15]]
        ])
    elif pair == (0, 2):  # xz
        D = np.array([
            [-a[1]*a[4] + a[0]*a[5], -a[3]*a[4] + a[2]*a[5] - a[1]*a[6] + a[0]*a[7], -a[3]*a[6] + a[2]*a[7]],
            [a[5]*a[8] - a[4]*a[9] - a[1]*a[12] + a[0]*a[13], a[7]*a[8] - a[6]*a[9] + a[5]*a[10] - a[4]*a[11] - a[3]*a[12] + a[2]*a[13] - a[1]*a[14] + a[0]*a[15], a[7]*a[10] - a[6]*a[11] - a[3]*a[14] + a[2]*a[15]],
            [-a[9]*a[12] + a[8]*a[13], -a[11]*a[12] + a[10]*a[13] - a[9]*a[14] + a[8]*a[15], -a[11]*a[14] + a[10]*a[15]]
        ])
    elif pair == (0, 3):  # xt
        D = np.array([
            [-a[2]*a[4] + a[0]*a[6], -a[3]*a[4] - a[2]*a[5] + a[1]*a[6] + a[0]*a[7], -a[3]*a[5] + a[1]*a[7]],
            [a[6]*a[8] - a[4]*a[10] - a[2]*a[12] + a[0]*a[14], a[7]*a[8] + a[6]*a[9] - a[5]*a[10] - a[4]*a[11] - a[3]*a[12] - a[2]*a[13] + a[1]*a[14] + a[0]*a[15], a[7]*a[9] - a[5]*a[11] - a[3]*a[13] + a[1]*a[15]],
            [-a[10]*a[12] + a[8]*a[14], -a[11]*a[12] - a[10]*a[13] + a[9]*a[14] + a[8]*a[15], -a[11]*a[13] + a[9]*a[15]]
        ])
    elif pair == (1, 2):  # yz
        D = np.array([
            [-a[1]*a[8] + a[0]*a[9], -a[3]*a[8] + a[2]*a[9] - a[1]*a[10] + a[0]*a[11], -a[3]*a[10] + a[2]*a[11]],
            [-a[5]*a[8] + a[4]*a[9] - a[1]*a[12] + a[0]*a[13], -a[7]*a[8] + a[6]*a[9] - a[5]*a[10] + a[4]*a[11] - a[3]*a[12] + a[2]*a[13] - a[1]*a[14] + a[0]*a[15], -a[7]*a[10] + a[6]*a[11] - a[3]*a[14] + a[2]*a[15]],
            [-a[5]*a[12] + a[4]*a[13], -a[7]*a[12] + a[6]*a[13] - a[5]*a[14] + a[4]*a[15], -a[7]*a[14] + a[6]*a[15]]
        ])
    elif pair == (1, 3):  # yt
        D = np.array([
            [-a[2]*a[8] + a[0]*a[10], -a[3]*a[8] - a[2]*a[9] + a[1]*a[10] + a[0]*a[11], -a[3]*a[9] + a[1]*a[11]],
            [-a[6]*a[8] + a[4]*a[10] - a[2]*a[12] + a[0]*a[14], -a[7]*a[8] - a[6]*a[9] + a[5]*a[10] + a[4]*a[11] - a[3]*a[12] - a[2]*a[13] + a[1]*a[14] + a[0]*a[15], -a[7]*a[9] + a[5]*a[11] - a[3]*a[13] + a[1]*a[15]],
            [-a[6]*a[12] + a[4]*a[14], -a[7]*a[12] - a[6]*a[13] + a[5]*a[14] + a[4]*a[15], -a[7]*a[13] + a[5]*a[15]]
        ])
    elif pair == (2, 3):  # zt
        D = np.array([
            [-a[4]*a[8] + a[0]*a[12], -a[5]*a[8] - a[4]*a[9] + a[1]*a[12] + a[0]*a[13], -a[5]*a[9] + a[1]*a[13]],
            [-a[6]*a[8] - a[4]*a[10] + a[2]*a[12] + a[0]*a[14], -a[7]*a[8] - a[6]*a[9] - a[5]*a[10] - a[4]*a[11] + a[3]*a[12] + a[2]*a[13] + a[1]*a[14] + a[0]*a[15], -a[7]*a[9] - a[5]*a[11] + a[3]*a[13] + a[1]*a[15]],
            [-a[6]*a[10] + a[2]*a[14], -a[7]*a[10] - a[6]*a[11] + a[3]*a[14] + a[2]*a[15], -a[7]*a[11] + a[3]*a[15]]
        ])
    else:
        raise ValueError(f"Invalid qubit pair: ({i}, {j})")

    return np.linalg.det(D)


def compute_four_qubit_SLOCC_invariants(psi: np.ndarray):
    """
    Compute SLOCC (Stochastic Local Operations and Classical Communication) invariants
    for 4-qubit states.

    Args:
        psi: 4-qubit state vector, shape (16,) or (2,2,2,2)

    Returns:
        tuple: (H, L, M, Dxt) - Four SLOCC invariants
            - H: Hyperdeterminant (degree-4 polynomial invariant)
            - L: Simple determinant of 4×4 reshaping
            - M: Determinant of specific index permutation
            - Dxt: D-invariant for qubits (0,3)

    Reference:
        These invariants characterize 4-qubit entanglement classes under SLOCC.
    """
    psi = psi.flatten()
    assert psi.shape == (16,), f"Expected 16-element state vector, got shape {psi.shape}"

    # Hyperdeterminant H
    H = (psi[0] * psi[15] - psi[1] * psi[14] - psi[2] * psi[13] + psi[3] * psi[12] -
         psi[4] * psi[11] + psi[5] * psi[10] + psi[6] * psi[9] - psi[7] * psi[8])

    # Simple determinant L
    L = np.linalg.det(psi.reshape(4, 4))

    # Permuted determinant M
    M = np.linalg.det([
        [psi[0], psi[8], psi[2], psi[10]],
        [psi[1], psi[9], psi[3], psi[11]],
        [psi[4], psi[12], psi[6], psi[14]],
        [psi[5], psi[13], psi[7], psi[15]],
    ])

    # D-invariant for qubits (0, 3) - x-t pair
    Dxt = compute_D(psi, 0, 3)

    return abs(H), abs(L), abs(M), abs(Dxt)


def compute_ent_params_from_state(state, option='I'):
    """
    Computes entanglement parameters characterizing the quantum state structure.

    Args:
        state: Quantum state of shape (2,2,2) for 3 qubits or (2,2,2,2) for 4 qubits,
               or flattened (8,) or (16,) as numpy array
        option: Return 'I' invariants or 'J' parameters (default: 'I')
                Only applicable for 3-qubit states. 4-qubit states return SLOCC invariants.

    Returns:
        np.ndarray: Entanglement parameters
            - For 3 qubits: shape (5,) with I1-I5 or J1-J5
            - For 4 qubits: shape (4,) with (H, L, M, Dxt) SLOCC invariants

    For 3-qubit states with option='I' (invariants):
        - I1: Tr(ρ_1²) - Single-party purity for player 1
        - I2: Tr(ρ_2²) - Single-party purity for player 2
        - I3: Tr(ρ_3²) - Single-party purity for player 3
        - I4: Tr((ρ_1 ⊗ ρ_2) ρ_12) - Two-party correlation measure
        - I5: |det₃(ψ)|² - Three-party entanglement (generalized concurrence)

    For 3-qubit states with option='J' (derived parameters):
        - J1, J2, J3: Transformed purity measures
        - J4: √I5 - Concurrence
        - J5: Higher-order correlation measure

    For 4-qubit states:
        - H: Hyperdeterminant
        - L: Simple determinant
        - M: Permuted determinant
        - Dxt: D-invariant for qubits (0,3)

    Implementation:
        For 3 qubits: Computes reduced density matrices for all subsystems and uses
        Levi-Civita tensor for determinant computation.

        For 4 qubits: Computes SLOCC invariants that classify entanglement under
        stochastic local operations and classical communication.
    """
    # Detect number of qubits
    if state.ndim == 1:
        n_qubits = int(np.log2(len(state)))
        if n_qubits == 4:
            # 4-qubit case
            return np.array(compute_four_qubit_SLOCC_invariants(state))
        elif n_qubits == 3:
            state = state.reshape(2, 2, 2)
        else:
            raise ValueError(f"Unsupported number of qubits: {n_qubits}. Only 3 or 4 qubits supported.")
    elif state.shape == (2, 2, 2, 2):
        # 4-qubit case
        return np.array(compute_four_qubit_SLOCC_invariants(state))

    # 3-qubit case continues below
    # Compute reduced density matrices
    rho_1 = einops.einsum(state, state.conj(), 'x i j, y i j -> x y')
    rho_2 = einops.einsum(state, state.conj(), 'i x j, i y j -> x y')
    rho_3 = einops.einsum(state, state.conj(), 'i j x, i j y -> x y')
    rho_12 = einops.einsum(state, state.conj(), 'x1 x2 i, y1 y2 i -> x1 y1 x2 y2')
    rho_12 = einops.rearrange(rho_12, 'x1 y1 x2 y2 -> (x1 x2) (y1 y2)')

    # Compute invariants
    I1 = np.trace(np.linalg.matrix_power(rho_1, 2))
    I2 = np.trace(np.linalg.matrix_power(rho_2, 2))
    I3 = np.trace(np.linalg.matrix_power(rho_3, 2))
    I4 = np.trace(np.kron(rho_1, rho_2) @ rho_12)

    # Compute 3-party entanglement using Levi-Civita tensor
    eps = np.array([[0, 1], [-1, 0]], dtype=state.dtype)
    det3 = 1/2 * einops.einsum(
        eps, eps, eps, eps, eps, eps, state, state, state, state,
        'i1 j1, i2 j2, k1 l1, k2 l2, i3 k3, j3 l3, i1 i2 i3, j1 j2 j3, k1 k2 k3, l1 l2 l3 ->'
    )
    I5 = np.abs(det3) ** 2

    if option == 'I':
        return np.stack([I1, I2, I3, I4, I5])
    elif option == 'J':
        J1 = 1/4 * (1 + I1 - I2 - I3 - 2 * np.sqrt(I5))
        J2 = 1/4 * (1 - I1 + I2 - I3 - 2 * np.sqrt(I5))
        J3 = 1/4 * (1 - I1 - I2 + I3 - 2 * np.sqrt(I5))
        J4 = np.sqrt(I5)
        J5 = 1/4 * (3 - 3 * I1 - 3 * I2 - I3 + 4 * I4 - 2 * np.sqrt(I5))
        return np.stack([J1, J2, J3, J4, J5])
    else:
        raise ValueError("Invalid option")


def metrics_to_dataframe(metric_logs, include_state=False, include_ent_params=True):
    """
    Convert metric_logs list to a pandas DataFrame.

    Args:
        metric_logs: List of dictionaries with keys 'energy', 'welfare', 'state', 'ent_params'
        include_state: If True, include the state column (not recommended for large datasets)
        include_ent_params: If True, include entanglement parameter columns (default: True)

    Returns:
        pd.DataFrame with columns for each player's energy, welfare, entanglement params, and optionally state
    """
    if len(metric_logs) == 0:
        return pd.DataFrame()

    # Extract data
    data = {
        'welfare': [log['welfare'] for log in metric_logs],
    }

    # Add per-player energy columns
    num_players = len(metric_logs[0]['energy'])
    for i in range(num_players):
        data[f'energy_player_{i}'] = [log['energy'][i] for log in metric_logs]

    # Add entanglement parameters if available and requested
    if include_ent_params and 'ent_params' in metric_logs[0] and metric_logs[0]['ent_params'] is not None:
        num_ent_params = len(metric_logs[0]['ent_params'])

        if num_ent_params == 5:
            # 3-qubit case: I1, I2, I3, I4, I5
            param_names = ['I1', 'I2', 'I3', 'I4', 'I5']
        elif num_ent_params == 4:
            # 4-qubit case: H, L, M, Dxt
            param_names = ['H', 'L', 'M', 'Dxt']
        else:
            # Generic fallback
            param_names = [f'param_{i}' for i in range(num_ent_params)]

        for i, name in enumerate(param_names):
            data[name] = [log['ent_params'][i] for log in metric_logs]

    # Optionally include state (as a column of arrays)
    if include_state:
        data['state'] = [log['state'] for log in metric_logs]

    df = pd.DataFrame(data)
    df.index.name = 'iteration'

    return df


def save_results(save_dir, Psi, H, metric_logs, **params):
    """Save optimization results with UUID and metadata.

    Args:
        save_dir: Directory to save results
        Psi: Final MPS state
        H: Hamiltonian (list of MPO tensors)
        metric_logs: List of metric dicts from optimization
        **params: All optimization parameters (chi, eps, max_num_steps, etc.)

    Returns:
        filepath: Path to saved file
        run_uuid: UUID string
    """
    from datetime import datetime

    run_uuid = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build pre-computed DataFrame
    df = metrics_to_dataframe(metric_logs, include_state=False, include_ent_params=True)

    # Build flat metadata dict (all params + results summary)
    num_players = len(Psi)
    metadata = {
        # Identifiers
        'uuid': run_uuid,
        'timestamp': timestamp,

        # All optimization parameters (flatten **params)
        **params,

        # Results summary
        'final_welfare': float(df['welfare'].iloc[-1]) if len(df) > 0 else None,
        'best_welfare': float(df['welfare'].max()) if len(df) > 0 else None,
        'num_iterations': len(metric_logs),
    }

    # Package everything
    results = {
        'metadata': metadata,
        'metric_logs': metric_logs,
        'dataframe': df,
        'Hamiltonian': H,
    }

    # Save single file with descriptive prefix
    os.makedirs(save_dir, exist_ok=True)
    filename = f"qpd{num_players}_{timestamp}_{run_uuid}.pkl"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {filepath}")
    print(f"UUID: {run_uuid}")

    return filepath, run_uuid


def opt_fid_state(
    Psi: list[np.ndarray] | np.ndarray, # Initial fiducial state (MPS or computational basis)
    H: list[np.ndarray], # Hamiltonian
    max_num_steps: int = 100, # Number of updates on the fiducial state before the program terminates
    eps: float = 0.005, # Learning rate associated with the update of the fiducial state
    num_perturbations: int = 10, # Number of perturbations to perform at each step to estimate the gradient
    subroutine_max_iter: int = 1000, # Max iter as in the equilibrium-finding subroutine
    subroutine_lr: float = 0.03, # Learning rate as in the equilibrium-finding subroutine
    max_subroutine_lr: float = 1, # Maximum learning rate for the subroutine
    expl_check_interval: int = 10, # Check exploitability every N iterations in Nash solver
    expl_maxiter: int = 300, # Max iterations for exploitability computation (differential evolution)
    real_strategies: bool = True, # Whether to use real strategies (exp(iY) only) for exploitability
    perturbation_method: str = 'schmidt', # Perturbation method: 'schmidt' or 'unitary'
    use_wandb: bool = False, # Whether to use wandb logging
    wandb_project: str = "nash-equilibrium", # W&B project name
    wandb_config: dict = None, # Additional wandb config
    wandb_log_interval: int = 10, # Log to wandb every N steps (1 = every step, 20 = every 20 steps)
    should_save_results: bool = True, # Whether to save results to file
    save_dir: str = "data", # Directory to save results
    seed: int = None, # Random seed used for initialization (for tracking/reproducibility)
):
    """
    Optimize the fiducial state for Nash equilibrium in quantum games.

    Supports both MPS (list of tensors) and computational basis (single array) inputs.
    Output format matches input format.
    """
    # Detect input format
    input_is_mps = is_mps_format(Psi)

    if input_is_mps:
        assert all(isinstance(Psi[i], np.ndarray) for i in range(len(Psi))), "Psi must be a list of numpy arrays"
    else:
        assert isinstance(Psi, np.ndarray), "Psi must be a numpy array"
        # Convert computational basis to MPS for internal optimization
        L = int(np.log2(Psi.size))
        Psi = from_comp_basis(Psi, L=L)

    assert all(isinstance(H[i], np.ndarray) for i in range(len(H))), "H must be a list of numpy arrays"

    # Initialize wandb if requested
    wandb_initialized_here = False
    if use_wandb:
        # Check if wandb is already initialized (e.g., by a sweep)
        if wandb.run is None:
            config = {
                'max_num_steps': max_num_steps,
                'eps': eps,
                'num_perturbations': num_perturbations,
                'subroutine_max_iter': subroutine_max_iter,
                'subroutine_lr': subroutine_lr,
                'max_subroutine_lr': max_subroutine_lr,
                'expl_check_interval': expl_check_interval,
                'expl_maxiter': expl_maxiter,
                'real_strategies': real_strategies,
                'perturbation_method': perturbation_method,
                'chi': Psi[0].shape[1],  # Bond dimension
                'L': len(Psi),  # Number of players
            }
            # Merge with additional config if provided
            if wandb_config:
                config.update(wandb_config)

            wandb.init(project=wandb_project, config=config)
            wandb_initialized_here = True
        else:
            # wandb already initialized (likely by sweep), just update config
            if wandb_config:
                wandb.config.update(wandb_config, allow_val_change=True)

    # Configuration for retry logic and adaptive LR
    max_failures_before_abort = 20  # Maximum number of retries (more gradual LR increases)
    current_working_lr = subroutine_lr  # Track the current effective LR (adapts over time)

    # Initialize: find the Nash equilibrium of the fiducial state
    Psi = to_canonical_form(Psi, form='B')
    baseline_result, baseline_success, final_alpha = find_nash_eq1_with_retry(
        Psi, H,
        max_iter=subroutine_max_iter,
        base_alpha=current_working_lr,
        max_alpha=max_subroutine_lr,
        max_retries=max_failures_before_abort,
        expl_check_interval=expl_check_interval,
        expl_maxiter=expl_maxiter,
        real_strategies=real_strategies,
        return_history=False
    )
    if not baseline_success:
        print(f"Warning: Initial baseline NE not found after {max_failures_before_abort} retries, proceeding with non-NE state")
    else:
        # Update working LR if a higher one was needed
        if final_alpha > current_working_lr:
            print(f"Updated working LR: {subroutine_lr:.4f} → {final_alpha:.4f}")
            current_working_lr = final_alpha
    Psi = to_canonical_form(baseline_result['state_'], form='B')

    metric_logs = []
    for i in tqdm(range(max_num_steps), desc="Optimizing fiducial state"):
        # perturb at specific site
        site = i % (len(Psi) - 1)
        Psi_batch, original_param, batch_perturbed_param = batch_perturb(
            Psi, batch_size=num_perturbations, lr=eps, site=site, method=perturbation_method
        )

        energy_diffs = []
        valid_param_diffs = []
        all_param_diffs = np.array(batch_perturbed_param) - np.array(original_param)

        for j in range(num_perturbations):
            Psi_ = [psi[j] for psi in Psi_batch]

            result_, success, final_alpha = find_nash_eq1_with_retry(
                Psi_, H,
                max_iter=subroutine_max_iter,
                base_alpha=current_working_lr,  # Use current working LR
                max_alpha=max_subroutine_lr,
                max_retries=max_failures_before_abort,
                expl_check_interval=expl_check_interval,
                expl_maxiter=expl_maxiter,
                real_strategies=real_strategies,
                return_history=False
            )

            if success:
                # Update working LR if a higher one was needed
                if final_alpha > current_working_lr:
                    print(f"Updated working LR: {current_working_lr:.4f} → {final_alpha:.4f}")
                    current_working_lr = final_alpha

                # Now result_['energy'] is final energy array (3,) for 3 players
                energy_diff = sum(result_['energy']) - sum(baseline_result['energy'])
                energy_diffs.append(energy_diff)
                valid_param_diffs.append(all_param_diffs[j])  # Only include successful perturbations
            else:
                print(f"Perturbation {j+1}/{num_perturbations}: Failed after {max_failures_before_abort} retries, skipping...")

        # Check if entire batch failed
        if len(energy_diffs) == 0:
            print(f"No Nash equilibrium found for any of the {num_perturbations} perturbations after retries. "
                  f"Capturing problematic run and aborting optimization.")

            # Run with full history to capture the problematic case
            problematic_result = find_nash_eq1(
                Psi, H,
                max_iter=subroutine_max_iter,
                alpha=subroutine_lr,
                expl_check_interval=expl_check_interval,
                expl_maxiter=expl_maxiter,
                real_strategies=real_strategies,
                return_history=True
            )

            # Save the problematic run to a special directory
            failed_runs_dir = os.path.join(save_dir, "failed_runs")
            os.makedirs(failed_runs_dir, exist_ok=True)

            from datetime import datetime
            failed_run_uuid = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failed_run_{timestamp}_{failed_run_uuid}.pkl"
            filepath = os.path.join(failed_runs_dir, filename)

            failed_run_data = {
                'problematic_result': problematic_result,
                'current_state': Psi,
                'Hamiltonian': H,
                'baseline_result': baseline_result,
                'metric_logs_so_far': metric_logs,
                'iteration': i,
                'num_perturbations': num_perturbations,
                'metadata': {
                    'uuid': failed_run_uuid,
                    'timestamp': timestamp,
                    'chi': Psi[0].shape[1],
                    'num_players': len(Psi),
                    'max_num_steps': max_num_steps,
                    'eps': eps,
                    'num_perturbations': num_perturbations,
                    'subroutine_max_iter': subroutine_max_iter,
                    'subroutine_lr': subroutine_lr,
                    'max_subroutine_lr': max_subroutine_lr,
                    'perturbation_method': perturbation_method,
                    'seed': seed,
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(failed_run_data, f)

            print(f"Problematic run saved to: {filepath}")
            print(f"Failed run UUID: {failed_run_uuid}")

            # Break out of the optimization loop
            break

        energy_diffs = np.array(energy_diffs)  # Shape: (num_successful,)
        valid_param_diffs = np.stack(valid_param_diffs)  # Shape: (num_successful, param_dim)

        grad_est = estimate_gradient_ols(valid_param_diffs, energy_diffs)  # Shape: (param_dim,)

        # Update the state using appropriate method
        if perturbation_method == 'schmidt':
            # Project gradient onto tangent space (orthogonal to current singular values)
            grad_est_proj = grad_est - np.dot(grad_est, original_param[0]) * original_param[0] / np.linalg.norm(original_param[0])**2
            Psi = update_state(Psi, grad_est_proj, lr=eps, site=site)
        elif perturbation_method == 'unitary':
            # For unitary method, grad_est is in coefficient space - no projection needed
            Psi = update_state_unitary(Psi, grad_est, lr=eps, site=site)
        else:
            raise ValueError(f"Unknown perturbation method: {perturbation_method}")

        baseline_result = find_nash_eq1(Psi, H, max_iter=subroutine_max_iter, alpha=subroutine_lr, return_history=False)
        Psi = to_canonical_form(baseline_result['state_'], form='B')

        # metric logs
        # Compute entanglement parameters (only for 3 or 4 qubits)
        num_players = len(Psi)
        if num_players <= 4:
            psi_comp = to_comp_basis(Psi).reshape([2] * num_players)
            ent_params = compute_ent_params_from_state(psi_comp, option='I')
        else:
            ent_params = None

        metrics = {
            'energy': baseline_result['energy'],
            'welfare': np.sum(baseline_result['energy']).item(),
            'state': Psi,
            'ent_params': ent_params,
        }
        metric_logs.append(metrics)

        # Log to wandb (at specified interval or on last step)
        should_log = (i % wandb_log_interval == 0) or (i == max_num_steps - 1)
        if use_wandb and should_log:
            wandb_metrics = {
                'welfare': np.real(metrics['welfare']),
            }

            # Add entanglement parameters with appropriate labels (only for 3 or 4 qubits)
            if ent_params is not None:
                if len(ent_params) == 5:
                    # 3-qubit case: I1, I2, I3, I4, I5
                    param_names = ['I1', 'I2', 'I3', 'I4', 'I5']
                elif len(ent_params) == 4:
                    # 4-qubit case: H, L, M, Dxt
                    param_names = ['H', 'L', 'M', 'Dxt']
                else:
                    # Generic fallback
                    param_names = [f'param_{i}' for i in range(len(ent_params))]

                for i_param, name in enumerate(param_names):
                    value = ent_params[i_param]
                    wandb_metrics[f'ent_params/{name}'] = np.real(value.item() if hasattr(value, 'item') else float(value))

            # Log individual player energies
            for player_idx, energy in enumerate(metrics['energy']):
                wandb_metrics[f'energy/player_{player_idx}'] = energy

            wandb.log(wandb_metrics, step=i)

    # Finish wandb run (only if we initialized it here, not in a sweep)
    if use_wandb and wandb_initialized_here:
        wandb.finish()

    # Save results to file
    if should_save_results:
        filepath, run_uuid = save_results(
            save_dir=save_dir,
            Psi=Psi,
            H=H,
            metric_logs=metric_logs,
            chi=Psi[0].shape[1],
            num_players=len(Psi),
            max_num_steps=max_num_steps,
            eps=eps,
            num_perturbations=num_perturbations,
            subroutine_max_iter=subroutine_max_iter,
            subroutine_lr=subroutine_lr,
            max_subroutine_lr=max_subroutine_lr,
            expl_check_interval=expl_check_interval,
            expl_maxiter=expl_maxiter,
            real_strategies=real_strategies,
            perturbation_method=perturbation_method,
            seed=seed,
        )

    # Convert back to original format if needed
    if not input_is_mps:
        Psi = to_comp_basis(Psi)

    return Psi, metric_logs

def parse_args():
    """Parse command line arguments for optimization."""

    # ========== DEFAULT CONFIGURATION ==========
    DEFAULTS = {
        # State initialization
        'non_commutative_norm': 0.0,
        'chi': 4,
        'num_players': 3,
        'seed': None,
        'dtype': 'real',

        # Optimization parameters
        'max_num_steps': 1000,
        'eps': 0.01,
        'num_perturbations': 20,
        'perturbation_method': 'unitary',

        # Nash equilibrium subroutine
        'subroutine_max_iter': 1000,
        'subroutine_lr': 0.009,
        'max_subroutine_lr': 1.0,
        'expl_check_interval': 50,
        'expl_maxiter': 100,
        'real_strategies': True,

        # Logging and saving
        'use_wandb': True,
        'wandb_project': 'nash-equilibrium',
        'wandb_experiment': 'default',
        'save_results': True,
        'save_dir': 'data',
        'include_state': False,
    }
    # ===========================================

    parser = argparse.ArgumentParser(
        description='Optimize fiducial state for Nash equilibrium in quantum games',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # State initialization parameters
    parser.add_argument('--non-commutative-norm', type=float, default=DEFAULTS['non_commutative_norm'],
                        help='Non-commutative norm for the payoff')
    parser.add_argument('--chi', type=int, default=DEFAULTS['chi'],
                        help='MPS bond dimension')
    parser.add_argument('--num-players', type=int, default=DEFAULTS['num_players'],
                        help='Number of players (L)')
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'],
                        help='Random seed for reproducibility')

    # Optimization parameters
    parser.add_argument('--max-num-steps', type=int, default=DEFAULTS['max_num_steps'],
                        help='Number of optimization steps')
    parser.add_argument('--eps', '--lr', type=float, default=DEFAULTS['eps'],
                        help='Learning rate for fiducial state updates')
    parser.add_argument('--num-perturbations', type=int, default=DEFAULTS['num_perturbations'],
                        help='Number of perturbations per step for gradient estimation')
    parser.add_argument('--perturbation-method', type=str, default=DEFAULTS['perturbation_method'],
                        choices=['schmidt', 'unitary'],
                        help='Perturbation method: schmidt (singular values) or unitary (Pauli coefficients)')

    # Nash equilibrium subroutine parameters
    parser.add_argument('--subroutine-max-iter', type=int, default=DEFAULTS['subroutine_max_iter'],
                        help='Max iterations for Nash equilibrium finder')
    parser.add_argument('--subroutine-lr', '--alpha', type=float, default=DEFAULTS['subroutine_lr'],
                        help='Learning rate for Nash equilibrium finder')
    parser.add_argument('--max-subroutine-lr', type=float, default=DEFAULTS['max_subroutine_lr'],
                        help='Maximum learning rate for Nash equilibrium finder retries')
    parser.add_argument('--expl-check-interval', type=int, default=DEFAULTS['expl_check_interval'],
                        help='Check exploitability every N iterations in Nash solver')
    parser.add_argument('--expl-maxiter', type=int, default=DEFAULTS['expl_maxiter'],
                        help='Max iterations for exploitability computation (differential evolution)')
    parser.add_argument('--real-strategies', action='store_true', default=DEFAULTS['real_strategies'],
                        help='Use real strategies (exp(iY) only) for exploitability computation')
    parser.add_argument('--no-real-strategies', dest='real_strategies', action='store_false',
                        help='Use full complex strategies (all SU(2)) for exploitability computation')

    # Logging and saving
    parser.add_argument('--use-wandb', action='store_true', default=DEFAULTS['use_wandb'],
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default=DEFAULTS['wandb_project'],
                        help='W&B project name')
    parser.add_argument('--wandb-experiment', type=str, default=DEFAULTS['wandb_experiment'],
                        help='W&B experiment name/tag')
    parser.add_argument('--save-results', action='store_true', default=DEFAULTS['save_results'],
                        help='Save results to CSV file')
    parser.add_argument('--no-save-results', dest='save_results', action='store_false',
                        help='Disable saving results')
    parser.add_argument('--save-dir', type=str, default=DEFAULTS['save_dir'],
                        help='Directory to save results')
    parser.add_argument('--include-state', action='store_true', default=DEFAULTS['include_state'],
                        help='Include state in the results')
    parser.add_argument('--dtype', type=str, default=DEFAULTS['dtype'],
                        help='Data type for the state and Hamiltonian')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.dtype == 'real':
        dtype = np.float32
    elif args.dtype == 'complex':
        dtype = np.complex64
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    # Initialize state and Hamiltonian
    print(f"Initializing random MPS with L={args.num_players}, chi={args.chi}")
    # Note: Uses global random state set above (no need to pass seed again)
    Psi = get_rand_state_as_mps(L=args.num_players, max_bond_dim=args.chi, dtype=dtype)

    if args.non_commutative_norm > 0:
        print(f"CRITICAL: In this run, implement non-commutative payoff")
        print(f"  Non-commutative norm: {args.non_commutative_norm}")
        print(f"  Seed: {args.seed}")

    Hs = get_perturbed_H_QPD(eps=args.non_commutative_norm, dtype=dtype)
    print(f"  2-Body Interaction Hamiltonians Used: {Hs}")
    H = get_default_cyclic_players(L=args.num_players, Hs=Hs, dtype=dtype)

    # Prepare wandb config
    wandb_config = {
        'experiment': args.wandb_experiment,
        'chi': args.chi,
        'seed': args.seed,
        'dtype': dtype,
    }

    print(f"Starting optimization:")
    print(f"  Steps: {args.max_num_steps}")
    print(f"  Learning rate: {args.eps}")
    print(f"  Perturbations: {args.num_perturbations}")
    print(f"  Perturbation method: {args.perturbation_method}")
    print(f"  Subroutine max iter: {args.subroutine_max_iter}")
    print(f"  Subroutine LR: {args.subroutine_lr}")
    print(f"  Max subroutine LR: {args.max_subroutine_lr}")
    print(f"  Expl check interval: {args.expl_check_interval}")
    print(f"  Expl maxiter: {args.expl_maxiter}")
    print(f"  Real strategies: {args.real_strategies}")
    print(f"  W&B logging: {args.use_wandb}")
    print(f"  Save results: {args.save_results}")

    # Run optimization
    Psi, metric_logs = opt_fid_state(
        Psi, H,
        max_num_steps=args.max_num_steps,
        eps=args.eps,
        num_perturbations=args.num_perturbations,
        subroutine_max_iter=args.subroutine_max_iter,
        subroutine_lr=args.subroutine_lr,
        max_subroutine_lr=args.max_subroutine_lr,
        expl_check_interval=args.expl_check_interval,
        expl_maxiter=args.expl_maxiter,
        real_strategies=args.real_strategies,
        perturbation_method=args.perturbation_method,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_config=wandb_config,
        should_save_results=args.save_results,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Display summary
    df = metrics_to_dataframe(metric_logs, include_state=args.include_state)
    print("\n" + "="*50)
    print("Optimization Summary")
    print("="*50)
    print(f"\nFinal welfare: {df['welfare'].iloc[-1]:.4f}")
    print(f"Best welfare: {df['welfare'].max():.4f}")
    print(f"\nFinal entanglement parameters:")

    # Detect parameter type from column names
    if 'I1' in df.columns:
        # 3-qubit case
        param_names = ['I1', 'I2', 'I3', 'I4', 'I5']
    elif 'H' in df.columns:
        # 4-qubit case
        param_names = ['H', 'L', 'M', 'Dxt']
    else:
        # Fallback: find all ent_param columns
        param_names = [col for col in df.columns if col.startswith('param_')]

    for name in param_names:
        if name in df.columns:
            print(f"  {name}: {df[name].iloc[-1]:.6f}")

    print("\nFirst 5 iterations:")
    print(df.head())
    print("\nLast 5 iterations:")
    print(df.tail())

if __name__ == "__main__":
    main()
