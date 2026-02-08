#%% Imports
import numpy as np
from tqdm import tqdm
from src.solver import find_nash_eq1
from src.game import get_default_cyclic_players, get_perturbed_H_QPD
from src.mps_utils import to_comp_basis, to_canonical_form, get_product_state
from src.solver import compute_exploitability, apply_u, apply_unitary, kick_with_u, update_state_unitary, compute_bipartite_entanglement_entropies
from functools import reduce
import torch as t

#%% Define the game
# Define the game parameters
L = 2
chi = 8
payoff_dtype = np.complex128
state_dtype = np.complex128
# Get Hamiltonian

# Get initial state

print(f"Game setup: L={L} players, chi={chi}, dtype={state_dtype}")
print(f"Hamiltonian shape: {[h.shape for h in H]}")
#%% Eq solver with sequential updates

def find_nash_eq1_seq(
    Psi: list[np.ndarray] | np.ndarray, # allowing for both MPS and computational basis input
    H: list[np.ndarray],
    max_iter: int = 10000,
    alpha: float = 0.01,
    convergence_threshold: float = 1e-7,
    expl_threshold: float = 5e-4,
    use_tqdm: bool = False,
    expl_check_interval: int = 50,
    return_history: bool = False,
    expl_maxiter: int = 60,
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
        E = []
        psi_before_first_update = None
        for i in range(L):
            # Recompute psi after each update (sequential updates)
            if isinstance(Psi, list):
                psi = to_comp_basis(Psi).reshape([2] * L)
            else:
                psi = Psi.reshape([2] * L)

            if i == 0:
                psi_before_first_update = psi.copy()
            elif i == 1 and n == 0:  # Debug on first iteration
                psi_diff = np.linalg.norm(psi - psi_before_first_update)
                if psi_diff > 1e-10:
                    print(f"Sequential update working: psi changed by {psi_diff:.6e} after player 0")
                else:
                    print("WARNING: psi unchanged after player 0 update!")

            dE = np.tensordot(H[i], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE, axes=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))

            E.append(np.trace(dE).real)
            dE = np.eye(2, dtype=dE.dtype) - alpha * dE

            Y, _, Z = np.linalg.svd(dE)
            unitary = (Y @ Z).T if np.isrealobj(dE) else (Y @ Z).T.conj()

            # Immediately apply the unitary (sequential update)
            if isinstance(Psi, list):
                Psi_before = Psi[i].copy()
                Psi[i] = apply_unitary(unitary.T, Psi[i])
                if n == 0 and i == 0:  # Debug first update
                    print(f"MPS tensor changed by: {np.linalg.norm(Psi[i] - Psi_before):.6e}")
            else:
                Psi = apply_u(unitary.T, psi, [i]).reshape(-1)

        Es.append(np.array(E))
        if n == 0:  # Debug first iteration
            print(f"Energies saved on iteration 0: {E}")
        if return_history:
            psi_list.append(psi)
            Psi_list.append(Psi)
        
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




#%% New Solver

import math
from scipy.optimize import differential_evolution
from src.solver import apply_u, apply_unitary

PAULIS = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),  # σ_x
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # σ_y
    np.array([[1, 0], [0, -1]], dtype=np.complex128),  # σ_z
]

def compute_best_response_de(
    psi: np.ndarray,
    H: list[np.ndarray],
    player_idx: int,
    real_strategies: bool = False,
    maxiter: int = 300,
    seed: int = 42,
    step_size: float = 1.0,
    verbose: bool = False
):
    """
    Compute best response for a single player using differential evolution.

    Args:
        psi: Current state as ndarray with shape (2,2,...,2) for L qubits
        H: List of Hamiltonian tensors for each player
        player_idx: Index of the player computing best response
        real_strategies: Whether to use real strategies (σ_y rotations only)
        maxiter: Maximum iterations for differential evolution
        seed: Random seed
        step_size: Damping factor in (0, 1] - scales rotation angle towards BR (default 1.0)
        verbose: Print optimization details

    Returns:
        best_unitary: Damped unitary (step_size * optimal rotation)
        best_payoff: Payoff achieved with FULL best response unitary
        baseline_payoff: Payoff with identity (no change)
    """
    L = psi.ndim
    psi_dtype = psi.dtype
    
    # Compute baseline payoff (identity unitary)
    def compute_payoff(psi_test):
        dE = np.tensordot(H[player_idx], psi_test, 
                         axes=([L+j for j in range(L)], [j for j in range(L)]))
        dE = np.tensordot(psi_test.conj(), dE, 
                         axes=([j for j in range(L) if j != player_idx], 
                               [j for j in range(L) if j != player_idx]))
        return float(np.trace(dE).real)
    
    baseline_payoff = compute_payoff(psi)
    
    # Define objective function for differential evolution
    def negative_payoff(params):
        """Negative payoff (for minimization)"""
        alpha = params[0]
        if real_strategies:
            theta = math.pi / 2
            phi = math.pi / 2
        else:
            theta = params[1]
            phi = params[2]
        
        # Construct SU(2) unitary: U = cos(α)I + i·sin(α)(n·σ)
        nx = math.sin(theta) * math.cos(phi)
        ny = math.sin(theta) * math.sin(phi)
        nz = math.cos(theta)
        
        if real_strategies:
            unitary = (np.eye(2, dtype=np.float64) * math.cos(alpha) + 
                      math.sin(alpha) * np.array([[0, 1], [-1, 0]], dtype=np.float64))
        else:
            unitary = (np.eye(2, dtype=np.complex128) * math.cos(alpha) + 
                      1j * math.sin(alpha) * (nx * PAULIS[0] + ny * PAULIS[1] + nz * PAULIS[2]))
        
        # Apply unitary to player's qubit (match compute_exploitability convention)
        psi_deviated = apply_u(unitary, psi, [player_idx])

        # Compute payoff
        payoff = compute_payoff(psi_deviated)
        
        return -payoff  # Minimize negative payoff = maximize payoff
    
    # Run differential evolution
    bounds = [(0, math.pi)] if real_strategies else [(0, math.pi), (0, math.pi), (0, 2*math.pi)]
    
    result = differential_evolution(
        negative_payoff,
        bounds=bounds,
        maxiter=maxiter,
        seed=seed,
        atol=1e-6,
        tol=1e-6,
        disp=verbose
    )
    
    best_payoff_full = -result.fun

    # Reconstruct best unitary with damping
    alpha_optimal = result.x[0]
    alpha_damped = alpha_optimal * step_size  # Apply damping to rotation angle

    if real_strategies:
        theta = math.pi / 2
        phi = math.pi / 2
    else:
        theta = result.x[1]
        phi = result.x[2]

    nx = math.sin(theta) * math.cos(phi)
    ny = math.sin(theta) * math.sin(phi)
    nz = math.cos(theta)

    if real_strategies:
        best_unitary = (np.eye(2, dtype=np.float64) * math.cos(alpha_damped) +
                       math.sin(alpha_damped) * np.array([[0, 1], [-1, 0]], dtype=np.float64))
    else:
        best_unitary = (np.eye(2, dtype=np.complex128) * math.cos(alpha_damped) +
                       1j * math.sin(alpha_damped) * (nx * PAULIS[0] + ny * PAULIS[1] + nz * PAULIS[2]))

    # Compute actual payoff from damped unitary (not the full BR)
    psi_damped = apply_u(best_unitary, psi, [player_idx])
    best_payoff = compute_payoff(psi_damped)

    if verbose:
        improvement = best_payoff - baseline_payoff
        print(f"  Player {player_idx}: baseline={baseline_payoff:.6f}, "
              f"best(damped)={best_payoff:.6f}, best(full)={best_payoff_full:.6f}, "
              f"improvement={improvement:.6f}, step_size={step_size:.2f}")

    return best_unitary, best_payoff, baseline_payoff

def find_nash_eq_iterated_br_fixed(
    Psi: list[np.ndarray] | np.ndarray,
    H: list[np.ndarray],
    max_iter: int = 1000,
    expl_threshold: float = 5e-4,
    convergence_threshold: float = 1e-7,
    real_strategies: bool = False,
    de_maxiter: int = 300,
    de_seed: int = 42,
    step_size: float = 1.0,
    expl_check_interval: int = 50,
    expl_maxiter: int = 300,
    return_history: bool = False,
    use_tqdm: bool = False,
    verbose: bool = False
):
    """
    Find Nash equilibrium using iterated best response with differential evolution.

    FIXED VERSION: Properly handles state saving and exploitability checking.

    Args:
        step_size: Damping factor in (0, 1] for best response steps (default 1.0).
                   Values < 1.0 help prevent limit cycles by taking partial steps.
    """
    # Convert to numpy if needed
    if isinstance(Psi, list) and isinstance(Psi[0], np.ndarray):
        pass  # Already numpy
    else:
        raise ValueError("Psi must be list of numpy arrays (MPS format)")
    
    L = len(Psi)
    
    # Initialize tracking
    Es = []
    psi_list = [] if return_history else None
    Psi_list = [] if return_history else None
    expl_list = []
    improvements_list = [] if return_history else None
    
    local_converged = False
    global_converged = False
    
    # Main iteration loop
    iterator = tqdm(range(max_iter), disable=not use_tqdm)
    for n in iterator:
        # Convert to computational basis for energy calculation
        psi = to_comp_basis(Psi).reshape([2] * L)
        
        # Compute current energies for all players
        E = []
        for i in range(L):
            dE = np.tensordot(H[i], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE, 
                            axes=([j for j in range(L) if j != i], 
                                  [j for j in range(L) if j != i]))
            E.append(np.trace(dE).real)
                   
        # Sequential best response updates
        iteration_improvements = []
        
        if verbose:
            print(f"\nIteration {n}, Energies: {E}")
        
        for player_idx in range(L):
            # Compute best response for this player
            best_unitary, best_payoff, baseline_payoff = compute_best_response_de(
                psi, H, player_idx,
                real_strategies=real_strategies,
                maxiter=de_maxiter,
                seed=de_seed + n * L + player_idx,
                step_size=step_size,
                verbose=verbose
            )
            
            improvement = best_payoff - baseline_payoff
            iteration_improvements.append(improvement)

            # Apply best response unitary to MPS
            # apply_unitary uses standard convention, apply_u has transpose built-in
            # So to match apply_u(best_unitary, ...), we use apply_unitary(best_unitary.T, ...)
            Psi[player_idx] = apply_unitary(best_unitary.T, Psi[player_idx])
            
            # Update psi for next player
            psi = to_comp_basis(Psi).reshape([2] * L)
        
        # FIX: Recompute energies AFTER all updates
        E_after = []
        for i in range(L):
            dE = np.tensordot(H[i], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE, 
                            axes=([j for j in range(L) if j != i], 
                                  [j for j in range(L) if j != i]))
            E_after.append(np.trace(dE).real)
        
        # Save post-update state and energies
        Es.append(np.array(E_after))
        if return_history:
            psi_list.append(psi.copy())
            Psi_list.append([p.copy() for p in Psi])
            improvements_list.append(np.array(iteration_improvements))
        
        # Check local convergence (energy stabilization)
        if n > 2 and not local_converged:
            energy_change = sum([abs(E_after[i] - Es[-2][i]) for i in range(L)])
            local_converged = energy_change < convergence_threshold
            if local_converged and verbose:
                print(f"Local convergence at iteration {n}, energy change: {energy_change:.2e}")
        
        # Check global convergence (exploitability)
        if n % expl_check_interval == 0:
            expl = [compute_exploitability(psi, H, i, maxiter=expl_maxiter, 
                                           seed=de_seed, real_strategies=real_strategies) 
                   for i in range(L)]
            expl_list.append(expl)
            
            if verbose or use_tqdm:
                msg = f"Iter {n}: expl={np.array(expl)}, improvements={iteration_improvements}"
                if use_tqdm:
                    iterator.set_postfix_str(f"expl={np.sum(expl):.2e}")
                if verbose:
                    print(msg)
            
            if sum(expl) < expl_threshold:
                global_converged = True
                if verbose:
                    print(f"Global convergence at iteration {n}, total exploitability: {sum(expl):.2e}")
                break
    
    result = {
        'nash_state': local_converged,
        'nash_equilibrium': global_converged,
        'energy': np.stack(Es) if return_history else Es[-1],
        'state': psi_list if return_history else psi,
        'state_': Psi_list if return_history else Psi,
        'num_iters': n,
        'expl': np.array(expl_list) if return_history else expl_list[-1],
    }
    
    if return_history:
        result['improvements'] = np.array(improvements_list)
    
    return result


#%% Plotting utility
import matplotlib.pyplot as plt
def plot_energy_and_exploitability(result, relative_to_init=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Energy
    L = len(result['energy'][0])
    if relative_to_init:
        axs[0].plot(result['energy']-result['energy'][0], linewidth=1, label=[f"energy {i}" for i in range(L)])
    else:
        axs[0].plot(result['energy'], linewidth=1, label=[f"energy {i}" for i in range(L)])
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Energy")
    if relative_to_init:
        axs[0].set_title("Energy Trajectory (Relative to Initial)")
    else:
        axs[0].set_title("Energy Trajectory (Absolute)")
    axs[0].legend()

    # Right panel: Exploitability
    axs[1].plot(result['expl'])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Exploitability")
    axs[1].set_title("Exploitability Trajectory")
    axs[1].legend()
    axs[1].set_yscale('log')

    plt.tight_layout()

def scramble(state, depth=10, unitary_eps=0.1, seed=42):
    L = len(state)
    np.random.seed(seed)
    for i in range(depth):
        state = update_state_unitary(state, np.random.randn(9), lr=unitary_eps, site=np.mod(i, L-1))
    return state



#%% Eq finding, BR
Psi = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
# Psi = scramble(Psi, depth=10, unitary_eps=1, seed=42)

print(f"Initial state: {to_comp_basis(Psi)}")
print(f"Entanglement entropies: {compute_bipartite_entanglement_entropies(Psi)}")
Hs = get_perturbed_H_QPD(eps=1, dtype=payoff_dtype, seed=43)
H = get_default_cyclic_players(L=L, Hs=Hs, dtype=payoff_dtype)
result_ibr = find_nash_eq_iterated_br_fixed(
    Psi,
    H,
    max_iter=100,
    expl_threshold=1e-16,
    real_strategies=False,
    de_maxiter=200,
    expl_check_interval=200,
    return_history=True,
    use_tqdm=True,
    verbose=False,
    step_size=1
)
#%% Plot IBR result
plot_energy_and_exploitability(result_ibr, relative_to_init=False)


#%%
Psi = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
# Psi = scramble(Psi, depth=1, unitary_eps=0.1, seed=42)
print(f"Initial State: {to_comp_basis(Psi)}")
result_ori_seq = find_nash_eq1_seq(
    Psi,
    H,
    max_iter=2000,
    expl_threshold=1e-16,
    alpha=0.1,
    real_strategies=False,
    return_history=True,
    use_tqdm=True,
    expl_check_interval=29,
    expl_maxiter=300,
)

#%%
Psi = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
# Psi = scramble(Psi, depth=1, unitary_eps=0.1, seed=42)
print(f"Initial State: {to_comp_basis(Psi)}")
result_ori = find_nash_eq1(
    Psi,
    H,
    max_iter=2000,
    expl_threshold=1e-16,
    alpha=0.1,
    real_strategies=False,
    return_history=True,
    use_tqdm=True,
    expl_check_interval=29,
    expl_maxiter=300,
)

#%% Compare results
print("Energy difference (max):", np.max(np.abs(result_ori_seq['energy'] - result_ori['energy'])))
print("Energy difference (mean):", np.mean(np.abs(result_ori_seq['energy'] - result_ori['energy'])))
print("\nFirst 5 iterations energy comparison:")
print("Sequential:", result_ori_seq['energy'][:5])
print("Original:", result_ori['energy'][:5])
print("Difference:", result_ori_seq['energy'][:5] - result_ori['energy'][:5])

#%% Test 1: Verify BR actually improves payoff for a single player
def test_single_br_improvement():
    """Test that a single best response actually improves payoff"""
    print("\n=== TEST 1: Single BR Improvement ===")

    Psi_test = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
    psi_test = to_comp_basis(Psi_test).reshape([2] * L)

    # Compute baseline payoff for player 0
    def compute_payoff_player(psi, H, player_idx):
        dE = np.tensordot(H[player_idx], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
        dE = np.tensordot(psi.conj(), dE,
                         axes=([j for j in range(L) if j != player_idx],
                               [j for j in range(L) if j != player_idx]))
        return float(np.trace(dE).real)

    baseline = compute_payoff_player(psi_test, H, 0)
    print(f"Baseline payoff (player 0): {baseline:.6f}")

    # Compute best response
    best_u, best_payoff, _ = compute_best_response_de(
        psi_test, H, 0,
        real_strategies=False,
        maxiter=300,
        seed=42,
        step_size=1.0,
        verbose=True
    )

    # Apply BR and verify (use same convention as compute_exploitability)
    psi_after = apply_u(best_u, psi_test, [0])
    actual_payoff = compute_payoff_player(psi_after, H, 0)

    print(f"BR predicted payoff: {best_payoff:.6f}")
    print(f"Actual payoff after BR: {actual_payoff:.6f}")
    print(f"Improvement: {actual_payoff - baseline:.6f}")
    print(f"Prediction error: {abs(actual_payoff - best_payoff):.2e}")

    if actual_payoff < baseline - 1e-6:
        print("❌ FAIL: BR decreased payoff!")
    elif abs(actual_payoff - best_payoff) > 1e-4:
        print("❌ FAIL: Prediction mismatch!")
    else:
        print("✅ PASS: BR improves payoff as expected")

    return baseline, best_payoff, actual_payoff

#%% Test 2: Compare BR direction to gradient direction
def test_br_vs_gradient():
    """Compare BR update direction to gradient-based update"""
    print("\n=== TEST 2: BR vs Gradient Direction ===")

    Psi_test = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
    psi_test = to_comp_basis(Psi_test).reshape([2] * L)

    # Compute gradient-based unitary (from find_nash_eq1)
    i = 0
    dE = np.tensordot(H[i], psi_test, axes=([L+j for j in range(L)], [j for j in range(L)]))
    dE = np.tensordot(psi_test.conj(), dE,
                     axes=([j for j in range(L) if j != i],
                           [j for j in range(L) if j != i]))

    baseline_energy = np.trace(dE).real
    dE_grad = np.eye(2, dtype=dE.dtype) - 0.1 * dE
    Y, _, Z = np.linalg.svd(dE_grad)
    unitary_grad = (Y @ Z).T.conj()

    # Apply gradient unitary
    psi_grad = apply_u(unitary_grad.T, psi_test, [i])
    dE_after_grad = np.tensordot(H[i], psi_grad, axes=([L+j for j in range(L)], [j for j in range(L)]))
    dE_after_grad = np.tensordot(psi_grad.conj(), dE_after_grad,
                                 axes=([j for j in range(L) if j != i],
                                       [j for j in range(L) if j != i]))
    energy_grad = np.trace(dE_after_grad).real

    # Compute BR unitary
    best_u, best_payoff, _ = compute_best_response_de(
        psi_test, H, i,
        real_strategies=False,
        maxiter=300,
        seed=42,
        step_size=1.0
    )

    print(f"Baseline energy: {baseline_energy:.6f}")
    print(f"After gradient step: {energy_grad:.6f} (Δ = {energy_grad - baseline_energy:+.6f})")
    print(f"After BR step: {best_payoff:.6f} (Δ = {best_payoff - baseline_energy:+.6f})")
    print(f"BR advantage: {best_payoff - energy_grad:+.6f}")

    if best_payoff > energy_grad + 1e-6:
        print("✅ BR finds better direction than gradient")
    elif best_payoff < energy_grad - 1e-6:
        print("⚠️  Gradient step better than BR!")
    else:
        print("≈ BR and gradient similar")

#%% Test 3: Multi-step consistency
def test_multistep_consistency():
    """Test if multiple BR steps actually progress toward equilibrium"""
    print("\n=== TEST 3: Multi-step Consistency ===")

    Psi_test = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)

    energies_history = []
    for step in range(5):
        psi = to_comp_basis(Psi_test).reshape([2] * L)
        E = []

        for player_idx in range(L):
            dE = np.tensordot(H[player_idx], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE,
                            axes=([j for j in range(L) if j != player_idx],
                                  [j for j in range(L) if j != player_idx]))
            E.append(np.trace(dE).real)

            # Compute and apply BR
            best_u, _, _ = compute_best_response_de(
                psi, H, player_idx,
                real_strategies=False,
                maxiter=200,
                seed=42 + step * L + player_idx,
                step_size=1.0
            )
            Psi_test[player_idx] = apply_unitary(best_u, Psi_test[player_idx])
            psi = to_comp_basis(Psi_test).reshape([2] * L)

        energies_history.append(E)
        print(f"Step {step}: E = {E}")

    # Check if energies are changing
    E_changes = [np.linalg.norm(np.array(energies_history[i+1]) - np.array(energies_history[i]))
                 for i in range(4)]
    print(f"Energy changes: {E_changes}")

    if all(c < 1e-8 for c in E_changes):
        print("⚠️  Energies stuck (not changing)")
    elif E_changes[-1] > E_changes[0]:
        print("⚠️  Not converging (changes increasing)")
    else:
        print("✅ Energies evolving")

#%% Test 4: Verify exploitability computation
def test_exploitability_vs_br():
    """Compare exploitability calculation to BR improvement"""
    print("\n=== TEST 4: Exploitability vs BR ===")

    # Use the final state from IBR
    if 'result_ibr' in globals() and result_ibr['num_iters'] > 0:
        # Convert to computational basis if needed
        # result_ibr['state_'] is a list of snapshots, get the last one
        psi_test = result_ibr['state']
        if not isinstance(psi_test, np.ndarray):
            # state is already in comp basis format (from the non-history return)
            # But we have return_history=True, so use the last snapshot
            psi_test = result_ibr['state'][-1] if isinstance(result_ibr['state'], list) else result_ibr['state']

        for player_idx in range(L):
            expl = compute_exploitability(psi_test, H, player_idx,
                                         maxiter=300, seed=42,
                                         real_strategies=False)

            # Also compute via BR
            best_u, best_payoff, baseline = compute_best_response_de(
                psi_test, H, player_idx,
                real_strategies=False,
                maxiter=300,
                seed=42,
                step_size=1.0
            )
            br_expl = best_payoff - baseline

            print(f"Player {player_idx}: expl={expl:.6e}, BR improvement={br_expl:.6e}, diff={abs(expl-br_expl):.2e}")

            if abs(expl - br_expl) > 1e-3:
                print(f"  ⚠️  Large mismatch for player {player_idx}!")
    else:
        print("Run result_ibr first")

#%% Test 5: Step size comparison
def test_step_sizes():
    """Test how different step sizes affect convergence"""
    print("\n=== TEST 5: Step Size Comparison ===")

    step_sizes = [1.0, 0.5, 0.3, 0.1]
    results = {}

    for step_size in step_sizes:
        print(f"\n--- Testing step_size={step_size} ---")
        Psi_test = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)

        # Run 10 iterations
        energies = []
        for step in range(10):
            psi = to_comp_basis(Psi_test).reshape([2] * L)
            E = []

            for player_idx in range(L):
                dE = np.tensordot(H[player_idx], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
                dE = np.tensordot(psi.conj(), dE,
                                axes=([j for j in range(L) if j != player_idx],
                                      [j for j in range(L) if j != player_idx]))
                E.append(np.trace(dE).real)

                # Compute and apply BR with damping
                best_u, _, _ = compute_best_response_de(
                    psi, H, player_idx,
                    real_strategies=False,
                    maxiter=200,
                    seed=42 + step * L + player_idx,
                    step_size=step_size
                )
                Psi_test[player_idx] = apply_unitary(best_u.T, Psi_test[player_idx])
                psi = to_comp_basis(Psi_test).reshape([2] * L)

            energies.append(E)

        # Compute final exploitability
        psi_final = to_comp_basis(Psi_test).reshape([2] * L)
        expl = [compute_exploitability(psi_final, H, i, maxiter=300, seed=42, real_strategies=False)
                for i in range(L)]

        results[step_size] = {
            'energies': np.array(energies),
            'final_expl': sum(expl),
            'energy_change': np.linalg.norm(np.array(energies[-1]) - np.array(energies[0]))
        }

        print(f"  Initial energies: {energies[0]}")
        print(f"  Final energies: {energies[-1]}")
        print(f"  Energy change: {results[step_size]['energy_change']:.6f}")
        print(f"  Final exploitability: {results[step_size]['final_expl']:.6e}")

    # Compare results
    print("\n=== Summary ===")
    print(f"{'Step Size':<12} {'Energy Change':<15} {'Final Expl':<15} {'Status'}")
    print("-" * 60)
    for step_size in step_sizes:
        r = results[step_size]
        status = "✅ Good" if r['final_expl'] < 1e-3 else "⚠️  High expl"
        print(f"{step_size:<12.2f} {r['energy_change']:<15.6f} {r['final_expl']:<15.6e} {status}")

    return results

#%% Run all tests
test_single_br_improvement()
test_br_vs_gradient()
test_multistep_consistency()
test_exploitability_vs_br()
test_step_sizes()

#%%
plot_energy_and_exploitability(result_ibr, relative_to_init=False)
# plot_energy_and_exploitability(result_ori_seq, relative_to_init=False)
# plot_energy_and_exploitability(result_ori, relative_to_init=False)

#%% Using saved results

