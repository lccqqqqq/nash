"""
Equilibrium solvers for the Nash equilibrium of multiplayer games in the normal form (with continuous strategy spaces). Should work for generic number of players.
"""

import torch as t
from mps_utils import to_comp_basis
import einops
from jaxtyping import Float


def compute_energy(psi: t.Tensor, H: t.Tensor):
    """
    Computes the energy of the state psi under the Hamiltonian H.
    """
    if isinstance(H, list):
        H = t.stack(H)
    
    L = len(psi.shape)
    coord_str = "".join([f"a{i} " for i in range(L)])
    coord_str_conj = "".join([f"b{i} " for i in range(L)])
    contraction_specification = "".join([coord_str, ', batch ', coord_str, ' ', coord_str_conj, ', ', coord_str_conj, ' -> batch'])
    E = einops.einsum(psi, H, psi.conj(), contraction_specification)
    return E

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
    
def procrustes_solver(
    Psi: list[t.Tensor], # List of MPS tensors, should alreasy in canonical form
    H: list[t.Tensor], # List of Hamiltonians, in the MPO form
    max_iter: int = 1000,
    alpha: float = 0.01,
    grad_norm_threshold: float = 1e-6,
    expl_threshold: float = 1e-2, # Threshold for *global* exploitability to accept a solution as a true Nash equilibrium
    return_history: bool = False, # Whether to return the history of the solution, including all the states, energies and exploitabilities
    use_tqdm: bool = True, # Whether to use tqdm to show the progress
    device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu')),
):
    """
    Each player simultaneously applies a close-to-id unitary roation that myopically increase their payoff according to the local gradient.
    """

    L = len(Psi)
    _iter = 0
    converged = False

    if return_history:
        E_history = []
        Psi_history = []


    while _iter <= max_iter and not converged:
        psi = to_comp_basis(Psi).reshape([2] * L)
        unitaries = []
        
        E = []
        for i in range(L):
            dE = t.tensordot(H[i], psi, dims=([L + j for j in range(L)], [j for j in range(L)]))
            dE = t.tensordot(psi.conj(), dE, dims=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))
            E.append(t.trace(dE).real.item())

            # TODO: the normalization here seems to be wrong, need to fix it
            # Compute unitary from SVD
            dE = t.eye(2, dtype=psi.dtype, device=psi.device) - alpha * dE / t.sqrt(t.sum(t.abs(dE)**2))
            U, S, Vh = t.linalg.svd(dE)
            unitaries.append((U @ Vh).T.conj())

        if return_history:
            E_history.append(E)
            Psi_history.append(psi)

        # Apply unitaries
        for i in range(L):
            Psi[i] = apply_unitary(unitaries[i], Psi[i])

        # Using the HS distance from the unitary to the identity to control the strength of the update
        U_norms = t.stack([t.sqrt(4-2*t.real(t.trace(U))) for U in unitaries]).sum()
        
        # Also check the exploitability criteria
        # to be implemented

        if U_norms < grad_norm_threshold:
            converged = True
            break

        _iter = _iter + 1

    result = {
        'converged': converged,
        'energy': E,
        'state': Psi,
        'num_iters': _iter,
    }
    if return_history:
        result['energy_history'] = E_history

    return result


def grad_solver(
    Psi: list[t.Tensor], # List of MPS tensors, should alreasy in canonical form
    H: list[t.Tensor], # List of Hamiltonians, in the MPO form
    max_iter: int = 1000,
    alpha: float = 0.01,
    grad_norm_threshold: float = 1e-6,
    expl_threshold: float = 1e-2, # Threshold for *global* exploitability to accept a solution as a true Nash equilibrium
    return_history: bool = False, # Whether to return the history of the solution, including all the states, energies and exploitabilities
    use_tqdm: bool = True, # Whether to use tqdm to show the progress
    device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu')),
):
    """
    Each player simultaneously compute the local gradient via the expression 
    <psi|i[H_i, A_i]|psi>
    """

    L = len(Psi)
    _iter = 0
    converged = False

    if return_history:
        E_history = []

    X = t.tensor([[0,1],[1,0]], dtype=t.complex64, device=device)
    Y = t.tensor([[0,-1j],[1j,0]], dtype=t.complex64, device=device)
    Z = t.tensor([[1,0],[0,-1]], dtype=t.complex64, device=device)

    while _iter <= max_iter and not converged:
        psi = to_comp_basis(Psi).reshape([2] * L).to(dtype=t.complex64)
        unitaries = []
        E = []
        dE_norms = []
        for i in range(L):
            dE_matrix = t.tensordot(H[i], psi, dims=([L + j for j in range(L)], [j for j in range(L)]))
            dE_matrix = t.tensordot(psi.conj(), dE_matrix, dims=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))
            dE_X = -2 * t.trace(dE_matrix @ X).imag  # Coefficient for X
            dE_Y = -2 * t.trace(dE_matrix @ Y).imag  # Coefficient for Y
            dE_Z = -2 * t.trace(dE_matrix @ Z).imag  # Coefficient for Z
            dE = t.stack([dE_X, dE_Y, dE_Z])
            E.append(t.trace(dE_matrix).real.item())

            # Compute norm (dE is now real)
            dE_norm = t.sqrt(t.sum(dE**2))
            
            # Compute rotation angle
            coef = t.clamp(dE_norm, max=alpha)
            
            # Build unitary: U = exp(i*coef*n·σ) where n = dE/||dE||
            if dE_norm > 1e-10:
                n = dE / dE_norm  # Normalized direction
                unitaries.append(
                    t.eye(2, dtype=t.complex64, device=device) * t.cos(coef) +
                    1j * (n[0] * X + n[1] * Y + n[2] * Z) * t.sin(coef)
                )
            else:
                # No update if gradient is zero
                unitaries.append(t.eye(2, dtype=t.complex64, device=device))
            
            dE_norms.append(dE_norm)
        
        # Apply unitaries
        for i in range(L):
            Psi[i] = apply_unitary(unitaries[i], Psi[i]) # This operation should not interfere with the canonicality of the MPS because we are applying single-qubit unitaries only.
        
        _iter = _iter + 1
        local_max_epl = t.stack(dE_norms).sum().item()

        if return_history:
            E_history.append(E)

        if local_max_epl < grad_norm_threshold:
            converged = True
            break
    
    if _iter > max_iter:
        print("Max iterations reached")

    result = {
        'converged': _iter < max_iter,
        'energy': E,
        'state': Psi,
        'num_iters': _iter,
        'local_max_epl': local_max_epl,
    }

    if return_history:
        result['E_history'] = E_history

    return result