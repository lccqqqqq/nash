"""
Test Nash equilibrium convergence for 4-qubit systems with chi=16.
Tests multiple random initializations to verify the solver works correctly.
"""

import numpy as np
from solver import find_nash_eq1, compute_exploitability
from mps_utils import get_rand_mps, to_comp_basis, to_canonical_form
from game import get_default_H

np.set_printoptions(precision=6, suppress=True)

def test_single_run(seed, max_iter=2000, alpha=0.01, verbose=True):
    """Test a single Nash equilibrium search from random initialization."""

    # Get 4-player Hamiltonian
    H = get_default_H(num_players=4)

    # Create initial state with chi=16
    chi = 16
    Psi = get_rand_mps(L=4, chi=chi, d_phys=2, seed=seed)
    Psi = to_canonical_form(Psi, form='B')

    if verbose:
        print(f"\nSeed {seed}:")
        print(f"  Initial MPS bond dimension: {chi}")

    # Run Nash equilibrium finder
    result = find_nash_eq1(
        Psi, H,
        max_iter=max_iter,
        alpha=alpha,
        convergence_threshold=1e-6,
        expl_threshold=1e-3,
        use_tqdm=verbose,
        expl_check_interval=10,
        return_history=False
    )

    # Extract final metrics
    nash_converged = result['nash_equilibrium']
    num_iters = result['num_iters']
    final_energies = result['energy']
    final_welfare = np.sum(final_energies)
    final_expl = result['expl'][-1]
    total_expl = np.sum(final_expl)

    if verbose:
        print(f"\n  Results:")
        print(f"    Converged to Nash equilibrium: {nash_converged}")
        print(f"    Iterations: {num_iters}")
        print(f"    Final welfare: {final_welfare:.6f}")
        print(f"    Final energies per player: {final_energies}")
        print(f"    Exploitability per player: {final_expl}")
        print(f"    Total exploitability: {total_expl:.6e}")

        if nash_converged:
            print(f"    ✓ Successfully found Nash equilibrium!")
        else:
            print(f"    ⚠ Did not converge to threshold (1e-3), but may be close")

    return {
        'seed': seed,
        'nash_converged': nash_converged,
        'num_iters': num_iters,
        'final_welfare': final_welfare,
        'final_energies': final_energies,
        'final_expl': final_expl,
        'total_expl': total_expl,
    }


def test_multiple_runs(n_runs=5, max_iter=2000, alpha=0.01):
    """Test multiple random initializations."""

    print("="*70)
    print("TESTING NASH EQUILIBRIUM CONVERGENCE - 4 QUBITS, CHI=16")
    print("="*70)
    print(f"\nRunning {n_runs} tests with different random seeds")
    print(f"Parameters: max_iter={max_iter}, alpha={alpha}")

    results = []
    for i in range(n_runs):
        seed = 42 + i
        result = test_single_run(seed, max_iter=max_iter, alpha=alpha, verbose=True)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL RUNS")
    print("="*70)
    print(f"\n{'Seed':<8} {'Converged':<12} {'Iters':<8} {'Welfare':<12} {'Total Expl':<15}")
    print("-"*70)

    n_converged = 0
    for r in results:
        conv_str = "✓ YES" if r['nash_converged'] else "✗ NO"
        if r['nash_converged']:
            n_converged += 1
        print(f"{r['seed']:<8} {conv_str:<12} {r['num_iters']:<8} "
              f"{r['final_welfare']:<12.6f} {r['total_expl']:<15.6e}")

    print("-"*70)
    print(f"\nConvergence rate: {n_converged}/{n_runs} ({100*n_converged/n_runs:.1f}%)")

    if n_converged > 0:
        converged_results = [r for r in results if r['nash_converged']]
        avg_welfare = np.mean([r['final_welfare'] for r in converged_results])
        avg_iters = np.mean([r['num_iters'] for r in converged_results])
        avg_expl = np.mean([r['total_expl'] for r in converged_results])

        print(f"\nAverages for converged runs:")
        print(f"  Welfare: {avg_welfare:.6f}")
        print(f"  Iterations: {avg_iters:.1f}")
        print(f"  Total exploitability: {avg_expl:.6e}")

    if n_converged == 0:
        print("\n⚠ WARNING: No runs converged to Nash equilibrium!")
        print("  Possible issues:")
        print("  1. Learning rate (alpha) may be too high or too low")
        print("  2. Max iterations may be insufficient")
        print("  3. The game may have multiple local Nash equilibria")
        print("\nTrying one run with adjusted parameters...")

        # Try with different alpha
        print("\nAttempting with alpha=0.005 (lower learning rate):")
        result = test_single_run(seed=100, max_iter=3000, alpha=0.005, verbose=True)
        if result['nash_converged']:
            print("\n✓ Converged with lower learning rate!")
        else:
            print(f"\n✗ Still didn't converge (expl={result['total_expl']:.6e})")

    return results


if __name__ == "__main__":
    # Run tests with moderate number of iterations
    results = test_multiple_runs(n_runs=5, max_iter=2000, alpha=0.01)
