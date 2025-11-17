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
    use_tqdm: bool = False,
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


def compute_ent_params_from_state(state, option='I'):
    """
    Computes entanglement parameters characterizing the quantum state structure.

    Args:
        state: Quantum state of shape (2,2,2) or flattened (8,) as numpy array
        option: Return 'I' invariants or 'J' parameters (default: 'I')

    Returns:
        np.ndarray: Entanglement parameters, shape (5,)

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
    if include_ent_params and 'ent_params' in metric_logs[0]:
        for i in range(5):  # I1, I2, I3, I4, I5
            data[f'I{i+1}'] = [log['ent_params'][i] for log in metric_logs]

    # Optionally include state (as a column of arrays)
    if include_state:
        data['state'] = [log['state'] for log in metric_logs]

    df = pd.DataFrame(data)
    df.index.name = 'iteration'

    return df



def opt_fid_state(
    Psi: list[np.ndarray], # Initial fiducial state
    H: list[np.ndarray], # Hamiltonian
    max_num_steps: int = 100, # Number of updates on the fiducial state before the program terminates
    eps: float = 0.005, # Learning rate associated with the update of the fiducial state
    num_perturbations: int = 10, # Number of perturbations to perform at each step to estimate the gradient
    subroutine_max_iter: int = 1000, # Max iter as in the equilibrium-finding subroutine
    subroutine_lr: float = 0.03, # Learning rate as in the equilibrium-finding subroutine
    use_wandb: bool = False, # Whether to use wandb logging
    wandb_project: str = "nash-equilibrium", # W&B project name
    wandb_config: dict = None, # Additional wandb config
    save_results: bool = True, # Whether to save results to file
    save_dir: str = "data", # Directory to save results
):
    assert all(isinstance(Psi[i], np.ndarray) for i in range(len(Psi))), "Psi must be a list of numpy arrays"
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
        # Compute entanglement parameters
        psi_comp = to_comp_basis(Psi).reshape([2] * len(Psi))
        ent_params = compute_ent_params_from_state(psi_comp, option='I')

        metrics = {
            'energy': baseline_result['energy'],
            'welfare': np.sum(baseline_result['energy']).item(),
            'state': Psi,
            'ent_params': ent_params,
        }
        metric_logs.append(metrics)

        # Log to wandb
        if use_wandb:
            wandb_metrics = {
                'welfare': metrics['welfare'],
                'ent_params/I1': ent_params[0].item() if hasattr(ent_params[0], 'item') else float(ent_params[0]),
                'ent_params/I2': ent_params[1].item() if hasattr(ent_params[1], 'item') else float(ent_params[1]),
                'ent_params/I3': ent_params[2].item() if hasattr(ent_params[2], 'item') else float(ent_params[2]),
                'ent_params/I4': ent_params[3].item() if hasattr(ent_params[3], 'item') else float(ent_params[3]),
                'ent_params/I5': ent_params[4].item() if hasattr(ent_params[4], 'item') else float(ent_params[4]),
            }
            # Log individual player energies
            for player_idx, energy in enumerate(metrics['energy']):
                wandb_metrics[f'energy/player_{player_idx}'] = energy

            wandb.log(wandb_metrics, step=i)

    # Finish wandb run (only if we initialized it here, not in a sweep)
    if use_wandb and wandb_initialized_here:
        wandb.finish()

    # Save results to file
    if save_results:
        os.makedirs(save_dir, exist_ok=True)

        # Convert to DataFrame
        df = metrics_to_dataframe(metric_logs, include_state=False, include_ent_params=True)

        # Generate filename with timestamp and parameters
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chi = Psi[0].shape[1]
        filename = (
            f"opt_fid_state_"
            f"chi{chi}_"
            f"lr{eps:.0e}_"
            f"steps{max_num_steps}_"
            f"alpha{subroutine_lr:.0e}_"
            f"{timestamp}.csv"
        )
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath)
        print(f"Results saved to: {filepath}")

    return Psi, metric_logs

def parse_args():
    """Parse command line arguments for optimization."""

    # ========== DEFAULT CONFIGURATION ==========
    DEFAULTS = {
        # State initialization
        'chi': 4,
        'num_players': 3,
        'seed': None,

        # Optimization parameters
        'max_num_steps': 1000,
        'eps': 0.01,
        'num_perturbations': 5,

        # Nash equilibrium subroutine
        'subroutine_max_iter': 1000,
        'subroutine_lr': 0.03,

        # Logging and saving
        'use_wandb': True,
        'wandb_project': 'nash-equilibrium',
        'wandb_experiment': 'default',
        'save_results': True,
        'save_dir': 'data',
    }
    # ===========================================

    parser = argparse.ArgumentParser(
        description='Optimize fiducial state for Nash equilibrium in quantum games',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # State initialization parameters
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

    # Nash equilibrium subroutine parameters
    parser.add_argument('--subroutine-max-iter', type=int, default=DEFAULTS['subroutine_max_iter'],
                        help='Max iterations for Nash equilibrium finder')
    parser.add_argument('--subroutine-lr', '--alpha', type=float, default=DEFAULTS['subroutine_lr'],
                        help='Learning rate for Nash equilibrium finder')

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize state and Hamiltonian
    print(f"Initializing random MPS with L={args.num_players}, chi={args.chi}")
    Psi = get_rand_mps(L=args.num_players, chi=args.chi, d_phys=2, seed=args.seed)
    H = get_default_H(num_players=args.num_players)

    # Prepare wandb config
    wandb_config = {
        'experiment': args.wandb_experiment,
        'chi': args.chi,
        'seed': args.seed,
    }

    print(f"Starting optimization:")
    print(f"  Steps: {args.max_num_steps}")
    print(f"  Learning rate: {args.eps}")
    print(f"  Perturbations: {args.num_perturbations}")
    print(f"  Subroutine max iter: {args.subroutine_max_iter}")
    print(f"  Subroutine LR: {args.subroutine_lr}")
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
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_config=wandb_config,
        save_results=args.save_results,
        save_dir=args.save_dir
    )

    # Display summary
    df = metrics_to_dataframe(metric_logs, include_state=False)
    print("\n" + "="*50)
    print("Optimization Summary")
    print("="*50)
    print(f"\nFinal welfare: {df['welfare'].iloc[-1]:.4f}")
    print(f"Best welfare: {df['welfare'].max():.4f}")
    print(f"\nFinal entanglement parameters:")
    for i in range(5):
        print(f"  I{i+1}: {df[f'I{i+1}'].iloc[-1]:.6f}")
    print("\nFirst 5 iterations:")
    print(df.head())
    print("\nLast 5 iterations:")
    print(df.tail())


