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

from opt_mps_fiducial_state import apply_unitary
from mps_utils import to_canonical_form, to_comp_basis, get_rand_mps, get_product_state, get_ghz_state, apply_random_unitaries, test_canonical_form
from IPython.display import HTML, display
from game import get_default_3players, get_default_2players, get_default_H


# Functions that compute the Nash equilibrium (using a local algorithm) and verify by computing the exploitability with differential evolution. This is doable because we are dealing with a small parameter space, and considering deviation only in the exp(iY) direction.

def apply_u(u, psi, idx):
    l = len(u.shape)//2
    psi = np.tensordot(u, psi, axes=(list(range(l)), idx))
    return np.moveaxis(psi, list(range(l)), idx)

# Depracated... does not need with random initializations
def kick_with_u(Psi):
    L = len(Psi)
    for i in range(L):
        U = np.linalg.qr(np.random.randn(2, 2))[0]
        Psi[i] = apply_unitary(U.T.conj(), Psi[i])
    return Psi

def compute_exploitability(psi, H, player_idx):
    L = psi.ndim

    def uni_dev_payoff(alpha_vec):
        alpha = alpha_vec[0]
        unitary = np.eye(2) * math.cos(alpha) + np.array([[0, 1], [-1, 0]]) * math.sin(alpha)
        psi_dev = apply_u(unitary, psi, [player_idx])
        dE = np.tensordot(H[player_idx], psi_dev, axes=([L+j for j in range(L)], [j for j in range(L)]))
        dE = np.tensordot(psi_dev.conj(), dE, axes=([j for j in range(L) if j != player_idx], [j for j in range(L) if j != player_idx]))
        return -float(np.trace(dE).real)

    result = differential_evolution(
        uni_dev_payoff,
        bounds=[(0, math.pi)],
        maxiter=100,
        seed=42,
        atol=1e-6,
        tol=1e-6,
    )
    return -result.fun + uni_dev_payoff(np.array([0]))

def find_nash_eq1(
    Psi: list[np.ndarray],
    H: list[np.ndarray],
    max_iter: int = 10000,
    alpha: float = 0.01,
    convergence_threshold: float = 1e-6,
    expl_threshold: float = 1e-3,
    use_tqdm: bool = True,
    expl_check_interval: int = 3,
    return_history: bool = False,
):
    # Convert types to ndarray
    if isinstance(Psi[0], t.Tensor):
        Psi = [p.cpu().numpy() for p in Psi]
    if isinstance(H[0], t.Tensor):
        H = [h.cpu().numpy() for h in H]

    L = len(Psi)
    Es = []
    psi_list = [] if return_history else None
    Psi_list = [] if return_history else None
    local_converged = False
    global_converged = False
    expl_list = []
    for n in tqdm(range(max_iter), disable=not use_tqdm):
        psi = to_comp_basis(Psi).reshape([2] * L)
        unitaries = []
        E = []
        for i in range(L):
            dE = np.tensordot(H[i], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE, axes=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))

            E.append(np.trace(dE).real)
            dE = np.eye(2) - alpha * dE

            Y, _, Z = np.linalg.svd(dE)
            unitaries.append((Y @ Z).T.conj())

        Es.append(np.array(E))
        if return_history:
            psi_list.append(psi)
            Psi_list.append(Psi)
        for i in range(L):
            # Here the convention is made sure to be the same as in `apply_u`
            Psi[i] = apply_unitary(unitaries[i].T, Psi[i])
        
        if n > 2 and not local_converged:
            local_converged = sum([abs(E[i] - Es[-2][i]) for i in range(L)]) < convergence_threshold
            if local_converged:
                print(f"Converged to Nash state at iteration {n}")


        if n % expl_check_interval == 0:
            expl = [compute_exploitability(psi, H, i) for i in range(L)]
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
        'expl': np.array(expl_list),
    }

    return result



def batch_perturb(Psi: list[t.Tensor] | list[np.ndarray], batch_size: int = 100, lr: float = 0.01, site: int = 0):
    """
    Assuming that the input is in the right canonical form, perturb the state by 
    left-canonicalizing and then fiddle with the singular values at each step.
    Uses NumPy to avoid device-specific issues.
    """
    # Convert to numpy if needed
    if isinstance(Psi[0], t.Tensor):
        Psi_np = [p.cpu().numpy() for p in Psi]
    else:
        Psi_np = Psi
    
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
        Psi_batch[j] = einops.rearrange(U, 'batch (d_phys chi_l) chi_r -> batch d_phys chi_l chi_r', 
                                         d_phys=d_phys, chi_l=chi_l)
        
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

    return Psi_batch, original_S, batch_perturbed_S

def estimate_gradient_ols(dX, dy, lam=0.0):
    """
    Estimate the gradient of a function f(x) by least squares.
    dX: np.ndarray
    dy: np.ndarray
    lam: float
    return: np.ndarray
    """
    # Center perturbations around x0
    
    # Compute Gram matrix and RHS
    A = dX.T @ dX               # (d, d)
    b = dX.T @ dy                # (d,)

    # Apply ridge regularization if lam > 0
    if lam > 0:
        A = A + lam * np.eye(A.shape[0])

    # Solve for gradient estimate
    g_hat = np.linalg.solve(A, b)
    return g_hat

def update_state(Psi, S_grad_est_proj, lr, site):
    """Apply targeted, controlled perturbation to the state"""
    
    Psi = to_canonical_form(Psi, form='B')
    if isinstance(Psi[0], t.Tensor):
        Psi = [p.numpy() for p in Psi]
    L = len(Psi)
    psi = Psi[0]
    d_phys = psi.shape[0]
    for j in range(L):
        psi_grouped = einops.rearrange(
            psi, 'd_phys chi_l chi_r -> (d_phys chi_l) chi_r'
        )
        U, S, Vh = np.linalg.svd(psi_grouped, full_matrices=False)
        Psi[j] = einops.rearrange(U, '(d_phys chi_l) chi_r -> d_phys chi_l chi_r', d_phys=d_phys)


        if j < L - 1:
            if j == site:
                S = S + lr * S_grad_est_proj
                S = S / np.linalg.norm(S)

            psi = einops.einsum(
                S, Vh, Psi[j+1],
                'bond_r, bond_r chi_l, d_phys chi_l chi_r -> d_phys bond_r chi_r'
            )
            
    return to_canonical_form(Psi, form='B')



def opt_fid_state(
    Psi: list[np.ndarray], # Initial fiducial state
    H: list[np.ndarray], # Hamiltonian
    max_num_steps: int = 100, # Number of updates on the fiducial state before the program terminates
    eps: float = 0.005, # Learning rate associated with the update of the fiducial state
    num_perturbations: int = 10, # Number of perturbations to perform at each step to estimate the gradient
    subroutine_max_iter: int = 1000, # Max iter as in the equilibrium-finding subroutine
    subroutine_lr: float = 0.03, # Learning rate as in the equilibrium-finding subroutine
):
    assert all(isinstance(Psi[i], np.ndarray) for i in range(len(Psi))), "Psi must be a list of numpy arrays"
    assert all(isinstance(H[i], np.ndarray) for i in range(len(H))), "H must be a list of numpy arrays"
    
    # Initialize: find the Nash equilibrium of the fiducial state
    Psi = to_canonical_form(Psi, form='B')
    baseline_result = find_nash_eq1(Psi, H, max_iter=subroutine_max_iter, alpha=subroutine_lr, return_history=False)
    Psi = to_canonical_form(baseline_result['state_'], form='B')

    metric_logs = []
    for i in tqdm(range(max_num_steps), desc="Optimizing fiducial state"):
        # perturb at specific site
        site = i % (len(Psi) - 1)
        Psi_batch, original_S, batch_perturbed_S = batch_perturb(Psi, batch_size=num_perturbations, lr=eps, site=site)

        energy_diffs = []
        valid_Ss_diffs = []
        all_Ss_diffs = np.array(batch_perturbed_S) - np.array(original_S)

        for j in range(num_perturbations):
            Psi_ = [psi[j] for psi in Psi_batch]
            result_ = find_nash_eq1(Psi_, H, max_iter=subroutine_max_iter, alpha=subroutine_lr, return_history=False)
            if result_['nash_equilibrium']:
                # Now result_['energy'] is final energy array (3,) for 3 players
                energy_diff = sum(result_['energy']) - sum(baseline_result['energy'])
                energy_diffs.append(energy_diff)
                valid_Ss_diffs.append(all_Ss_diffs[j])  # Only include successful perturbations

        if len(energy_diffs) == 0:
            print(f"No Nash equilibrium found for any of the {num_perturbations} perturbations. Skipping update.")
            continue

        energy_diffs = np.array(energy_diffs)  # Shape: (num_successful,)
        valid_Ss_diffs = np.stack(valid_Ss_diffs)  # Shape: (num_successful, bond_dim)

        grad_est = estimate_gradient_ols(valid_Ss_diffs, energy_diffs)  # Shape: (bond_dim,)

        # Project gradient onto tangent space (orthogonal to current singular values)
        grad_est_proj = grad_est - np.dot(grad_est, original_S[0]) * original_S[0] / np.linalg.norm(original_S[0])**2

        # Update the state
        Psi = update_state(Psi, grad_est_proj, lr=eps, site=site)
        baseline_result = find_nash_eq1(Psi, H, max_iter=subroutine_max_iter, alpha=subroutine_lr, return_history=False)
        Psi = to_canonical_form(baseline_result['state_'], form='B')

        # metric logs
        metric_logs.append({
            'energy': baseline_result['energy'],
            'state': Psi,
        })
        
    return Psi, metric_logs


if __name__ == "__main__":
    Psi = get_rand_mps(L=3, chi=4, d_phys=2)
    H = get_default_H(num_players=3)
    Psi, metric_logs = opt_fid_state(Psi, H, max_num_steps=100, eps=0.005, num_perturbations=10, subroutine_max_iter=1000, subroutine_lr=0.03)
    print(metric_logs)


