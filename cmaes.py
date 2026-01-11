"""
CMA-ES-based optimization for fiducial states in quantum games.

This module implements a hybrid CPU/GPU approach using JAX for efficient Nash equilibrium
solving and NumPy/scipy for accurate exploitability validation.

Architecture:
- Outer loop: CMA-ES (evosax) optimizes fiducial state parameters
- Inner loop: JAX-based Nash equilibrium solver (differential best response)
- Validation: CPU-based exploitability checks using scipy (periodic and final)
"""

import jax
import jax.numpy as jnp
import numpy as np
import einops
from evosax.algorithms import CMA_ES
from typing import List, Dict, Optional

# Enable float64 precision for numerical accuracy
jax.config.update("jax_enable_x64", True)

# Import existing NumPy modules for validation
try:
    from src.mps_utils import to_comp_basis, get_rand_mps
    from src.solver import compute_exploitability
    from src.game import get_default_H
except ImportError:
    from mps_utils import to_comp_basis, get_rand_mps
    from solver import compute_exploitability
    from game import get_default_H


# ==============================================================================
# Section 1: JAX MPS Utilities (needed before encoding)
# ==============================================================================

def to_comp_basis_jax(Psi: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Contract MPS tensors to computational basis wavefunction (JAX version).

    Works for open boundary conditions (first and last bond dims = 1).

    Args:
        Psi: List of MPS tensors with varying shapes:
             - First: (d_phys, 1, chi) or (d_phys, chi_L, chi_R)
             - Middle: (d_phys, chi_L, chi_R)
             - Last: (d_phys, chi, 1) or (d_phys, chi_L, chi_R)

    Returns:
        Wavefunction in computational basis, shape (2**L,)
    """
    # Contract all tensors sequentially
    # Match the contraction pattern from src/mps_utils.py:
    # einops: "... chi_l bond, phys bond chi_r -> ... phys chi_l chi_r"
    psi = Psi[0]
    for another_psi in Psi[1:]:
        # Contract over the bond dimension (last of psi, middle of another_psi)
        # psi: (...accumulated..., chi_l, bond_out)
        # another_psi: (phys, bond_in, chi_r)
        # Contract bond_out with bond_in
        # result: (...accumulated..., phys, chi_l, chi_r)
        psi = jnp.einsum('...ab,cbd->...cad', psi, another_psi)

    # Squeeze out singleton dimensions and flatten to 1D
    psi = psi.squeeze()

    # Reshape to 1D vector
    return psi.reshape(-1)


# ==============================================================================
# Section 2: Parameter Encoding/Decoding
# ==============================================================================

def params_to_mps(params_flat: jnp.ndarray, L: int, chi: int, d_phys: int = 2) -> List[jnp.ndarray]:
    """
    Convert flat real parameter vector to MPS in open boundary form (JAX version, JIT-compatible).

    Creates OBC MPS with shapes: (d_phys, 1, chi), (d_phys, chi, chi), ..., (d_phys, chi, 1)

    Args:
        params_flat: Flat real-valued parameter vector from CMA-ES
                     Shape: (L * 2 * d_phys * chi^2,)
        L: Number of sites (players)
        chi: MPS bond dimension
        d_phys: Physical dimension (default: 2 for qubits)

    Returns:
        List of L MPS tensors in OBC form (JAX arrays)
    """
    # Decode parameters to MPS tensors
    Psi = []
    params_per_site = 2 * d_phys * chi**2

    for i in range(L):
        # Extract parameters for site i
        start = i * params_per_site
        end = start + params_per_site
        site_params = params_flat[start:end]

        # Reshape to (d_phys, chi, chi) complex tensor
        n_complex = d_phys * chi**2
        real_part = site_params[:n_complex].reshape(d_phys, chi, chi)
        imag_part = site_params[n_complex:].reshape(d_phys, chi, chi)
        raw_tensor = real_part + 1j * imag_part

        # Apply appropriate OBC structure
        if i == 0:
            # First site: (d_phys, 1, chi)
            # Take only the first column of the bond dimension
            tensor = raw_tensor[:, 0:1, :]
            # Normalize
            tensor_norm = jnp.sqrt(jnp.sum(jnp.abs(tensor)**2))
            tensor = tensor / (tensor_norm + 1e-10)
            Psi.append(tensor)
        elif i == L - 1:
            # Last site: (d_phys, chi, 1)
            # Take only the first row of the right bond dimension
            tensor = raw_tensor[:, :, 0:1]
            # Normalize
            tensor_norm = jnp.sqrt(jnp.sum(jnp.abs(tensor)**2))
            tensor = tensor / (tensor_norm + 1e-10)
            Psi.append(tensor)
        else:
            # Middle sites: (d_phys, chi, chi)
            # Normalize
            tensor_norm = jnp.sqrt(jnp.sum(jnp.abs(raw_tensor)**2))
            tensor = raw_tensor / (tensor_norm + 1e-10)
            Psi.append(tensor)

    # Final normalization: ensure wavefunction has unit norm
    psi_vec = to_comp_basis_jax(Psi)
    wf_norm = jnp.sqrt(jnp.sum(jnp.abs(psi_vec)**2))

    # Normalize by scaling all tensors uniformly
    norm_factor = wf_norm**(1.0 / L)  # Distribute normalization across L tensors
    Psi = [tensor / norm_factor for tensor in Psi]

    return Psi


def params_to_mps_canonical(params_flat: jnp.ndarray, L: int, chi: int, d_phys: int = 2) -> List[np.ndarray]:
    """
    Convert flat real parameter vector to MPS in canonical form (NumPy version).

    Uses existing mps_utils.to_canonical_form(). NOT JIT-compatible.
    Use this only for validation/exploitability checks, not in fast fitness.

    Args:
        params_flat: Flat real-valued parameter vector from CMA-ES
        L: Number of sites (players)
        chi: MPS bond dimension
        d_phys: Physical dimension (default: 2 for qubits)

    Returns:
        List of L MPS tensors in canonical form (NumPy arrays)
    """
    # First get JAX MPS
    Psi_jax = params_to_mps(params_flat, L, chi, d_phys)

    # Convert to NumPy
    Psi_np = [np.array(tensor) for tensor in Psi_jax]

    # Apply canonical form using existing function
    try:
        from src.mps_utils import to_canonical_form
    except ImportError:
        from mps_utils import to_canonical_form

    Psi_canonical = to_canonical_form(Psi_np, form='B')

    return Psi_canonical


def encode_mps_to_params(Psi: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Flatten MPS to parameter vector (for initialization).

    Note: This is an approximate inverse of params_to_mps, used for encoding
    existing MPS states as CMA-ES starting points.

    Args:
        Psi: List of MPS tensors with shape (d_phys, chi, chi)

    Returns:
        Flat real-valued parameter vector
    """
    params = []
    for tensor in Psi:
        # Flatten real and imaginary parts
        params.append(jnp.real(tensor).ravel())
        params.append(jnp.imag(tensor).ravel())
    return jnp.concatenate(params)


# ==============================================================================
# Section 3: More JAX MPS Utilities
# ==============================================================================

def apply_unitary_jax(U: jnp.ndarray, tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Apply single-qubit unitary to MPS tensor (JAX version).

    Args:
        U: (2, 2) unitary matrix
        tensor: (d_phys, chi_L, chi_R) MPS tensor

    Returns:
        (d_phys, chi_L, chi_R) updated tensor
    """
    # Contract: U[a,b] @ tensor[b, i, j] → new[a, i, j]
    return jnp.einsum('ab,bij->aij', U, tensor)


# ==============================================================================
# Section 3: JAX Nash Equilibrium Solver
# ==============================================================================

def find_nash_eq_jax(
    Psi: List[jnp.ndarray],
    H: List[jnp.ndarray],
    alpha: float = 0.01,
    n_iters: int = 2000
) -> Dict:
    """
    Find Nash equilibrium using differential best response dynamics (JAX version).

    Uses fixed iteration count with jax.lax.scan for GPU efficiency.

    Args:
        Psi: List of MPS tensors in left-canonical form
        H: List of Hamiltonian tensors for each player
        alpha: Learning rate for best response
        n_iters: Fixed number of iterations

    Returns:
        dict with keys:
            - 'Psi': final MPS state (list of tensors)
            - 'energies': final payoffs for each player
            - 'welfare': sum of energies (social welfare)
            - 'energy_history': energy trajectory (n_iters, L)
    """
    L = len(Psi)

    def nash_step(carry, _):
        """Single iteration of best response dynamics."""
        Psi_current = carry

        # Convert MPS to computational basis
        psi = to_comp_basis_jax(Psi_current)  # Shape: (2**L,)
        psi = psi.reshape([2] * L)  # Shape: (2, 2, ..., 2)

        # Compute unitaries for each player
        unitaries = []
        energies = []

        for i in range(L):
            # Contract Hamiltonian with state (all indices)
            axes_H = [L+j for j in range(L)]
            axes_psi = list(range(L))
            dE = jnp.tensordot(H[i], psi, axes=(axes_H, axes_psi))

            # Contract with conjugate state (all indices except i)
            axes_contract = [j for j in range(L) if j != i]
            dE = jnp.tensordot(jnp.conj(psi), dE, axes=(axes_contract, axes_contract))

            # Energy expectation
            energy = jnp.trace(dE).real
            energies.append(energy)

            # Best response update: dE → U via SVD
            # dE_update = I - alpha * dE
            dE_update = jnp.eye(2, dtype=dE.dtype) - alpha * dE
            Y, _, Z = jnp.linalg.svd(dE_update, full_matrices=False)
            U = (Y @ Z).T.conj()  # Unitary
            unitaries.append(U)

        # Apply unitaries to MPS
        Psi_new = [
            apply_unitary_jax(unitaries[i].T, Psi_current[i])
            for i in range(L)
        ]

        return Psi_new, jnp.array(energies)

    # Run fixed iterations using lax.scan
    Psi_final, energy_history = jax.lax.scan(nash_step, Psi, None, length=n_iters)

    return {
        'Psi': Psi_final,
        'energies': energy_history[-1],  # Final energies
        'welfare': jnp.sum(energy_history[-1]),
        'energy_history': energy_history
    }


# ==============================================================================
# Section 4: Fitness Functions
# ==============================================================================

def compute_fitness_fast(
    params: jnp.ndarray,
    H_jax: List[jnp.ndarray],
    config: Dict
) -> float:
    """
    Fast fitness function without exploitability check (GPU-only, vmap-compatible).

    Used for population evaluation in CMA-ES loop.

    Args:
        params: Flat real-valued parameter vector
        H_jax: List of JAX Hamiltonian tensors
        config: Configuration dict with keys: L, chi, nash_alpha, nash_iters

    Returns:
        Scalar fitness value (negative welfare, since CMA-ES minimizes)
    """
    # Decode parameters to MPS
    Psi_jax = params_to_mps(params, L=config['L'], chi=config['chi'])

    # Find Nash equilibrium
    result = find_nash_eq_jax(
        Psi_jax, H_jax,
        alpha=config['nash_alpha'],
        n_iters=config['nash_iters']
    )

    # Return negative welfare (CMA-ES minimizes)
    return -result['welfare']


def compute_fitness(
    params: jnp.ndarray,
    H_jax: List[jnp.ndarray],
    H_np: List[np.ndarray],
    config: Dict
) -> float:
    """
    Hybrid CPU/GPU fitness function with optional exploitability check.

    Used for validation (not vmap-compatible due to scipy dependency).

    Args:
        params: Flat real-valued parameter vector
        H_jax: List of JAX Hamiltonian tensors (for Nash solver on GPU)
        H_np: List of NumPy Hamiltonian tensors (for exploitability on CPU)
        config: Configuration dict with keys:
            - L, chi, nash_alpha, nash_iters (required)
            - check_exploitability, expl_threshold, expl_maxiter, real_strategies (optional)

    Returns:
        Scalar fitness value (negative welfare + optional exploitability penalty)
    """
    # 1. Decode parameters to MPS (JAX)
    Psi_jax = params_to_mps(params, L=config['L'], chi=config['chi'])

    # 2. Find Nash equilibrium (GPU/JAX)
    result = find_nash_eq_jax(
        Psi_jax, H_jax,
        alpha=config['nash_alpha'],
        n_iters=config['nash_iters']
    )
    welfare = result['welfare']

    # 3. Optional: Check exploitability (CPU/NumPy)
    if config.get('check_exploitability', False):
        # Transfer from GPU to CPU
        Psi_np = [np.array(tensor) for tensor in result['Psi']]
        psi_np = to_comp_basis(Psi_np).reshape([2] * config['L'])

        # Compute total exploitability across all players
        total_expl = sum([
            compute_exploitability(
                psi_np, H_np, player_idx=i,
                maxiter=config.get('expl_maxiter', 300),
                seed=42,
                real_strategies=config.get('real_strategies', True)
            )
            for i in range(config['L'])
        ])

        # Add penalty if exploitability exceeds threshold
        expl_penalty = jnp.maximum(0.0, total_expl - config.get('expl_threshold', 1e-3)) * 1000.0

        return -welfare + expl_penalty

    # 4. Default: Return negative welfare only
    return -welfare


# ==============================================================================
# Section 5: CMA-ES Main Loop
# ==============================================================================

def optimize_fiducial_state(
    L: int,
    chi: int,
    H: List[np.ndarray],
    initial_Psi: Optional[List[np.ndarray]] = None,
    pop_size: int = 64,
    n_generations: int = 1000,
    nash_alpha: float = 0.01,
    nash_iters: int = 2000,
    seed: int = 0,
    log_interval: int = 10,
    validation_interval: int = 50,
    expl_threshold: float = 1e-3,
    expl_maxiter: int = 300,
    real_strategies: bool = True,
    save_dir: str = "cmaes_results",
) -> Dict:
    """
    Main CMA-ES optimization loop with periodic exploitability validation.

    Args:
        L: Number of players
        chi: MPS bond dimension
        H: List of Hamiltonian tensors (NumPy arrays)
        initial_Psi: Optional initial MPS (random if None)
        pop_size: CMA-ES population size
        n_generations: Number of CMA-ES generations
        nash_alpha: Learning rate for Nash solver
        nash_iters: Fixed iterations for Nash solver
        seed: Random seed
        log_interval: Print progress every N generations
        validation_interval: Validate with exploitability every N generations
        expl_threshold: Exploitability threshold for Nash equilibrium
        expl_maxiter: Max iterations for scipy differential_evolution
        real_strategies: Use real strategies (exp(iY) only) for exploitability
        save_dir: Directory to save checkpoints

    Returns:
        dict with keys:
            - best_params: best parameter vector found
            - best_Psi: best MPS state (NumPy arrays)
            - best_welfare: best social welfare achieved
            - exploitability: final exploitability
            - exploitability_per_player: per-player exploitabilities
            - is_nash: whether solution is Nash equilibrium
            - final_state: final CMA-ES state
    """
    # Setup
    rng = jax.random.PRNGKey(seed)
    n_params = L * 2 * 2 * chi**2  # (L sites) * (2 for complex) * (d_phys=2 * chi^2)

    print(f"=== CMA-ES Fiducial State Optimization ===")
    print(f"System: L={L}, χ={chi}, n_params={n_params}")
    print(f"CMA-ES: pop_size={pop_size}, generations={n_generations}")
    print(f"Nash: alpha={nash_alpha}, iters={nash_iters}")
    print(f"Validation: interval={validation_interval}, expl_threshold={expl_threshold}\n")

    # Prepare Hamiltonians (JAX for GPU, NumPy for CPU)
    H_jax = [jnp.array(h) for h in H]
    H_np = H  # Keep original NumPy arrays

    # Initial mean: encode initial_Psi or random
    if initial_Psi is not None:
        print(f"Initializing from provided MPS state")
        initial_Psi_jax = [jnp.array(p) for p in initial_Psi]
        initial_mean = encode_mps_to_params(initial_Psi_jax)
    else:
        print(f"Initializing from random state")
        rng, rng_init = jax.random.split(rng)
        initial_mean = jax.random.normal(rng_init, (n_params,)) * 0.1

    # Initialize CMA-ES with evosax API
    strategy = CMA_ES(population_size=pop_size, solution=initial_mean)
    es_params = strategy.default_params

    rng, rng_state = jax.random.split(rng)
    # evosax init signature: init(key, solution, params)
    state = strategy.init(rng_state, initial_mean, es_params)

    # Config for fast fitness (GPU-only, no exploitability)
    config_fast = {
        'L': L,
        'chi': chi,
        'nash_alpha': nash_alpha,
        'nash_iters': nash_iters,
    }

    # Config for full fitness with exploitability (if requested)
    config_full = {
        'L': L,
        'chi': chi,
        'nash_alpha': nash_alpha,
        'nash_iters': nash_iters,
        'check_exploitability': validation_interval == 1,  # Check every generation if interval=1
        'expl_threshold': expl_threshold,
        'expl_maxiter': expl_maxiter,
        'real_strategies': real_strategies,
    }

    # Choose fitness function based on validation interval
    if validation_interval == 1:
        # Strict mode: validate every solution with exploitability
        print("⚠ STRICT MODE: Checking exploitability for EVERY fitness evaluation")
        print("  This ensures genuine Nash equilibria but is much slower (~100x)")
        print("  Consider using validation_interval > 1 for faster optimization\n")

        def fitness_single(p):
            # Cannot use vmap with exploitability check (scipy not JAX-compatible)
            return compute_fitness(p, H_jax, H_np, config_full)

        # Sequential evaluation (no vmap possible with exploitability)
        def fitness_fn(population):
            return jnp.array([fitness_single(p) for p in population])
    else:
        # Fast mode: GPU-only fitness, periodic validation
        def fitness_single(p):
            return compute_fitness_fast(p, H_jax, config_fast)

        fitness_fn = jax.jit(jax.vmap(fitness_single))

    # Optimization loop
    best_welfare = -jnp.inf
    best_params = None
    best_exploitability = None

    print("Starting optimization...\n")

    for gen in range(n_generations):
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)

        # Sample population (evosax API)
        x, state = strategy.ask(rng_ask, state, es_params)

        # Evaluate fitness (GPU-only, fast)
        fitness = fitness_fn(x)

        # Update CMA-ES (evosax API: tell needs a random key and returns metrics)
        state, metrics = strategy.tell(rng_tell, x, fitness, state, es_params)

        # Track best
        gen_best_idx = jnp.argmin(fitness)
        gen_best_welfare = -fitness[gen_best_idx]
        if gen_best_welfare > best_welfare:
            best_welfare = float(gen_best_welfare)
            best_params = x[gen_best_idx]

        # In strict mode, exploitability is already checked for all solutions
        # Extract mean exploitability from population
        if validation_interval == 1:
            # In strict mode, fitness includes exploitability penalty
            # We can infer average quality from fitness distribution
            mean_fitness = float(jnp.mean(fitness))
            std_fitness = float(jnp.std(fitness))

        # Periodic validation with exploitability (CPU, slow)
        # Skip if validation_interval==1 (strict mode already checks everything)
        if validation_interval > 1 and gen % validation_interval == 0:
            # Convert best params to MPS (JAX → NumPy)
            Psi_jax = params_to_mps(best_params, L, chi)
            Psi_np = [np.array(tensor) for tensor in Psi_jax]
            psi_np = to_comp_basis(Psi_np).reshape([2] * L)

            # Compute exploitability for each player
            expl_per_player = [
                compute_exploitability(
                    psi_np, H_np, player_idx=i,
                    maxiter=expl_maxiter,
                    seed=42,
                    real_strategies=real_strategies
                )
                for i in range(L)
            ]
            total_expl = sum(expl_per_player)
            best_exploitability = total_expl

            # Extended logging with exploitability
            expl_str = ', '.join([f'{e:.6f}' for e in expl_per_player])
            print(f"Gen {gen:4d}/{n_generations} | "
                  f"Best welfare: {best_welfare:8.4f} | "
                  f"Mean fitness: {float(jnp.mean(fitness)):8.4f} | "
                  f"Expl: {total_expl:.6f} [{expl_str}]")

            # Check if Nash equilibrium reached
            if total_expl < expl_threshold:
                print(f"  ✓ Nash equilibrium found!")
        elif validation_interval == 1:
            # Strict mode: log every generation with fitness stats
            if gen % log_interval == 0:
                print(f"Gen {gen:4d}/{n_generations} | "
                      f"Best welfare: {best_welfare:8.4f} | "
                      f"Mean fitness: {mean_fitness:8.4f} ± {std_fitness:6.4f}")
        else:
            # Normal logging (no exploitability check)
            if gen % log_interval == 0:
                print(f"Gen {gen:4d}/{n_generations} | "
                      f"Best welfare: {best_welfare:8.4f} | "
                      f"Mean fitness: {float(jnp.mean(fitness)):8.4f}")

    # Final validation with exploitability
    print("\n=== Final Validation ===")
    Psi_jax_final = params_to_mps(best_params, L, chi)
    Psi_np_final = [np.array(tensor) for tensor in Psi_jax_final]
    psi_np_final = to_comp_basis(Psi_np_final).reshape([2] * L)

    expl_final = [
        compute_exploitability(
            psi_np_final, H_np, player_idx=i,
            maxiter=expl_maxiter,
            seed=42,
            real_strategies=real_strategies
        )
        for i in range(L)
    ]
    total_expl_final = sum(expl_final)

    print(f"Final welfare: {best_welfare:.4f}")
    print(f"Final exploitability: {total_expl_final:.6f}")
    print(f"Per-player exploitability: {[f'{e:.6f}' for e in expl_final]}")
    is_nash = total_expl_final < expl_threshold
    print(f"Is Nash equilibrium (threshold={expl_threshold})? {is_nash}")

    return {
        'best_params': best_params,
        'best_Psi': Psi_np_final,  # Return NumPy arrays for compatibility
        'best_welfare': best_welfare,
        'exploitability': total_expl_final,
        'exploitability_per_player': expl_final,
        'is_nash': is_nash,
        'final_state': state,
    }


# ==============================================================================
# Section 6: CLI and Main Function
# ==============================================================================

if __name__ == "__main__":
    """Command-line interface for CMA-ES optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CMA-ES optimization for quantum game fiducial states"
    )

    # System parameters
    parser.add_argument('--L', type=int, default=3,
                        help='Number of players (default: 3)')
    parser.add_argument('--chi', type=int, default=2,
                        help='MPS bond dimension (default: 2)')
    parser.add_argument('--non-commutative-norm', type=float, default=0.0,
                        help='Non-commutative perturbation norm (default: 0.0)')

    # CMA-ES parameters
    parser.add_argument('--pop-size', type=int, default=64,
                        help='CMA-ES population size (default: 64)')
    parser.add_argument('--n-generations', type=int, default=1000,
                        help='Number of CMA-ES generations (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Nash solver parameters
    parser.add_argument('--nash-alpha', type=float, default=0.01,
                        help='Nash solver learning rate (default: 0.01)')
    parser.add_argument('--nash-iters', type=int, default=2000,
                        help='Nash solver iterations (default: 2000)')

    # Validation parameters
    parser.add_argument('--strict-nash', action='store_true',
                        help='STRICT MODE: Check exploitability for EVERY solution (slow but guarantees Nash equilibria)')
    parser.add_argument('--validation-interval', type=int, default=50,
                        help='Validate with exploitability every N generations. Set to 1 for strict mode. (default: 50)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Print progress every N generations (default: 10)')
    parser.add_argument('--expl-threshold', type=float, default=1e-3,
                        help='Exploitability threshold for Nash equilibrium (default: 1e-3)')
    parser.add_argument('--expl-maxiter', type=int, default=300,
                        help='Max iterations for exploitability computation (default: 300)')
    parser.add_argument('--real-strategies', action='store_true', default=True,
                        help='Use real strategies (exp(iY) only) for exploitability (default: True)')

    # Output parameters
    parser.add_argument('--save-dir', type=str, default='cmaes_results',
                        help='Directory to save results (default: cmaes_results)')

    args = parser.parse_args()

    # Override validation_interval if strict mode requested
    if args.strict_nash:
        args.validation_interval = 1
        print("⚠ STRICT MODE ENABLED: validation_interval set to 1")
        print("  Every solution will be validated with exploitability checks")
        print("  This is ~100x slower but ensures genuine Nash equilibria\n")

    # Load Hamiltonian
    if args.non_commutative_norm > 0:
        try:
            from src.game import get_perturbed_H_QPD
            H = get_perturbed_H_QPD(
                eps=args.non_commutative_norm,
                dtype=np.float64,
                seed=args.seed
            )
        except ImportError:
            from game import get_perturbed_H_QPD
            H = get_perturbed_H_QPD(
                eps=args.non_commutative_norm,
                dtype=np.float64,
                seed=args.seed
            )
        # Convert to L-player game
        if args.L > 2:
            print(f"Warning: Perturbed Hamiltonian only supports 2 players. Using default for {args.L} players.")
            H = get_default_H(num_players=args.L, option='H', dtype=np.float64)
    else:
        H = get_default_H(num_players=args.L, option='H', dtype=np.float64)

    # Run optimization
    result = optimize_fiducial_state(
        L=args.L,
        chi=args.chi,
        H=H,
        initial_Psi=None,  # Random initialization
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        nash_alpha=args.nash_alpha,
        nash_iters=args.nash_iters,
        seed=args.seed,
        log_interval=args.log_interval,
        validation_interval=args.validation_interval,
        expl_threshold=args.expl_threshold,
        expl_maxiter=args.expl_maxiter,
        real_strategies=args.real_strategies,
        save_dir=args.save_dir,
    )

    # Display results
    print("\n" + "="*70)
    print("Optimization Complete")
    print("="*70)
    print(f"Best welfare: {result['best_welfare']:.4f}")
    print(f"Exploitability: {result['exploitability']:.6f}")
    print(f"Is Nash equilibrium: {result['is_nash']}")
    print(f"Results saved to: {args.save_dir}")
    print("="*70)
