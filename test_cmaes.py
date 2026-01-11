"""
Test script for cmaes.py implementation.

Runs incremental tests to validate:
1. Imports and dependencies
2. Parameter encoding/decoding
3. JAX MPS utilities
4. Nash equilibrium solver
5. Small-scale CMA-ES optimization
"""

import numpy as np
import sys

print("="*70)
print("Testing CMA-ES Implementation")
print("="*70)

# ==============================================================================
# Test 1: Imports
# ==============================================================================
print("\n[Test 1] Testing imports...")
try:
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import CMA_ES
    print(f"  ✓ JAX version: {jax.__version__}")
    print(f"  ✓ JAX devices: {jax.devices()}")
    print(f"  ✓ evosax.algorithms.CMA_ES imported successfully")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

try:
    import cmaes
    print(f"  ✓ cmaes.py imported successfully")
except ImportError as e:
    print(f"  ✗ cmaes.py import error: {e}")
    sys.exit(1)

# ==============================================================================
# Test 2: Parameter Encoding
# ==============================================================================
print("\n[Test 2] Testing parameter encoding...")

L = 3
chi = 2
d_phys = 2
n_params = L * 2 * d_phys * chi**2

# Create random parameters
rng = jax.random.PRNGKey(42)
params = jax.random.normal(rng, (n_params,))

# Encode to MPS
Psi = cmaes.params_to_mps(params, L=L, chi=chi)

print(f"  Parameters: {n_params} real values")
print(f"  MPS: {len(Psi)} tensors, each shape {Psi[0].shape}")

# Check canonical form (each tensor should be an isometry)
for i, tensor in enumerate(Psi):
    # Reshape to matrix and check A†A = I
    mat = tensor.reshape(-1, chi)
    prod = mat.T.conj() @ mat
    identity = jnp.eye(chi)
    error = jnp.max(jnp.abs(prod - identity))
    print(f"  Tensor {i}: isometry error = {error:.2e}", end="")
    if error < 1e-10:
        print(" ✓")
    else:
        print(" ✗ FAILED")

# Check normalization
psi = cmaes.to_comp_basis_jax(Psi)
norm = jnp.linalg.norm(psi)
print(f"  Wavefunction norm: {norm:.6f}", end="")
if abs(norm - 1.0) < 1e-10:
    print(" ✓")
else:
    print(" ✗ FAILED")

# ==============================================================================
# Test 3: Compare JAX vs NumPy MPS utilities
# ==============================================================================
print("\n[Test 3] Testing JAX vs NumPy MPS utilities...")

try:
    from src.mps_utils import to_comp_basis as to_comp_basis_np
    from src.mps_utils import get_rand_mps

    # Create random MPS with NumPy
    Psi_np = get_rand_mps(L=L, chi=chi, dtype=np.complex128, seed=42)

    # Convert to JAX
    Psi_jax = [jnp.array(p) for p in Psi_np]

    # Compare outputs
    psi_np = to_comp_basis_np(Psi_np)
    psi_jax = cmaes.to_comp_basis_jax(Psi_jax)

    diff = jnp.max(jnp.abs(psi_np - psi_jax))
    print(f"  to_comp_basis difference: {diff:.2e}", end="")
    if diff < 1e-10:
        print(" ✓")
    else:
        print(" ✗ FAILED")

except ImportError:
    print("  ⚠ Skipping (src.mps_utils not available)")

# ==============================================================================
# Test 4: Nash Equilibrium Solver
# ==============================================================================
print("\n[Test 4] Testing Nash equilibrium solver...")

try:
    from src.game import get_default_H
    from src.solver import find_nash_eq1

    # Setup small game
    L_test = 3
    chi_test = 2
    H_np = get_default_H(num_players=L_test, option='H', dtype=np.float64)
    H_jax = [jnp.array(h) for h in H_np]

    # Create initial state
    Psi_init_np = get_rand_mps(L=L_test, chi=chi_test, dtype=np.complex128, seed=123)
    Psi_init_jax = [jnp.array(p) for p in Psi_init_np]

    print(f"  Running JAX Nash solver (1000 iterations)...")
    result_jax = cmaes.find_nash_eq_jax(
        Psi_init_jax, H_jax,
        alpha=0.01,
        n_iters=1000
    )

    print(f"  Final energies: {[f'{e:.4f}' for e in result_jax['energies']]}")
    print(f"  Social welfare: {result_jax['welfare']:.4f}")

    # Compare with NumPy version (qualitative check)
    print(f"  Running NumPy Nash solver for comparison...")
    result_np = find_nash_eq1(
        Psi_init_np, H_np,
        max_iter=1000,
        alpha=0.01,
        expl_check_interval=1000,  # Skip exploitability check
        use_tqdm=False
    )

    print(f"  NumPy final energies: {[f'{e:.4f}' for e in result_np['energy']]}")
    print(f"  NumPy social welfare: {sum(result_np['energy']):.4f}")

    # Check if energies are similar (they won't be identical due to different iterations)
    energy_diff = jnp.max(jnp.abs(result_jax['energies'] - result_np['energy']))
    print(f"  Energy difference: {energy_diff:.4f} (expect <0.1 for convergence)")

except ImportError as e:
    print(f"  ⚠ Skipping (dependencies not available): {e}")

# ==============================================================================
# Test 5: Fitness Function
# ==============================================================================
print("\n[Test 5] Testing fitness functions...")

try:
    # Use the same setup as Test 4
    config_fast = {
        'L': L_test,
        'chi': chi_test,
        'nash_alpha': 0.01,
        'nash_iters': 500,
    }

    # Test fast fitness (no exploitability)
    params_test = jax.random.normal(jax.random.PRNGKey(456), (L_test * 2 * 2 * chi_test**2,))
    fitness_fast = cmaes.compute_fitness_fast(params_test, H_jax, config_fast)
    print(f"  Fast fitness: {fitness_fast:.4f} ✓")

    # Test vectorization
    print(f"  Testing vmap with population of 4...")
    params_pop = jax.random.normal(jax.random.PRNGKey(789), (4, L_test * 2 * 2 * chi_test**2))
    fitness_fn = jax.jit(jax.vmap(lambda p: cmaes.compute_fitness_fast(p, H_jax, config_fast)))
    fitness_pop = fitness_fn(params_pop)
    print(f"  Population fitness: {[f'{f:.4f}' for f in fitness_pop]} ✓")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ==============================================================================
# Test 6: Small CMA-ES Run
# ==============================================================================
print("\n[Test 6] Running small CMA-ES optimization...")
print("  (This may take 1-2 minutes)")

try:
    result = cmaes.optimize_fiducial_state(
        L=3,
        chi=2,
        H=H_np,
        initial_Psi=None,
        pop_size=16,  # Very small population
        n_generations=20,  # Very short run
        nash_alpha=0.01,
        nash_iters=500,  # Fewer iterations for speed
        validation_interval=10,
        expl_threshold=1e-3,
        expl_maxiter=100,  # Faster exploitability check
        real_strategies=True,
        seed=42
    )

    print("\n  Results:")
    print(f"    Best welfare: {result['best_welfare']:.4f}")
    print(f"    Exploitability: {result['exploitability']:.6f}")
    print(f"    Is Nash: {result['is_nash']}")
    print("  ✓ CMA-ES optimization completed successfully!")

except Exception as e:
    print(f"  ✗ Error during optimization: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*70)
print("Testing complete!")
print("="*70)
print("\nNext steps:")
print("  1. For a more thorough test, run: python cmaes.py")
print("  2. For production runs, increase:")
print("     - n_generations (e.g., 1000)")
print("     - pop_size (e.g., 64)")
print("     - nash_iters (e.g., 2000-5000)")
print("  3. For large systems (L>5, χ>4), ensure GPU is available")
