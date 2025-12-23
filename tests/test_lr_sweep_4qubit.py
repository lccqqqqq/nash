"""
Learning rate sweep for 4-qubit Nash equilibrium finder.
Tests different learning rates to find optimal convergence settings.
"""

import numpy as np
import pandas as pd
from solver import find_nash_eq1
from mps_utils import get_rand_mps, to_canonical_form
from game import get_default_H

np.set_printoptions(precision=6, suppress=True)

def test_single_run(seed, alpha, max_iter=2000, verbose=False):
    """Test a single Nash equilibrium search with given learning rate."""

    # Get 4-player Hamiltonian
    H = get_default_H(num_players=4)

    # Create initial state with chi=16
    chi = 16
    Psi = get_rand_mps(L=4, chi=chi, d_phys=2, seed=seed)
    Psi = to_canonical_form(Psi, form='B')

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

    return {
        'seed': seed,
        'alpha': alpha,
        'nash_converged': nash_converged,
        'num_iters': num_iters,
        'final_welfare': final_welfare,
        'total_expl': total_expl,
    }


def lr_sweep(alpha_values, n_seeds=10, max_iter=2000):
    """
    Sweep over learning rates and test convergence statistics.

    Args:
        alpha_values: List of learning rates to test
        n_seeds: Number of random seeds to test per learning rate
        max_iter: Maximum iterations per run
    """

    print("="*80)
    print("LEARNING RATE SWEEP FOR 4-QUBIT NASH EQUILIBRIUM FINDER")
    print("="*80)
    print(f"\nTesting {len(alpha_values)} learning rates × {n_seeds} seeds = {len(alpha_values) * n_seeds} total runs")
    print(f"Parameters: chi=16, max_iter={max_iter}, expl_threshold=1e-3\n")

    all_results = []

    for alpha in alpha_values:
        print(f"\n{'='*80}")
        print(f"Testing alpha = {alpha:.4f}")
        print(f"{'='*80}")

        alpha_results = []
        for i in range(n_seeds):
            seed = 100 + i  # Use consistent seeds across alpha values
            result = test_single_run(seed, alpha, max_iter=max_iter, verbose=False)
            alpha_results.append(result)
            all_results.append(result)

            # Print progress
            status = "✓ CONVERGED" if result['nash_converged'] else f"✗ NO (expl={result['total_expl']:.2e})"
            print(f"  Seed {seed}: {status:30s}  iters={result['num_iters']:4d}  welfare={result['final_welfare']:.4f}")

        # Summary for this alpha
        n_converged = sum(r['nash_converged'] for r in alpha_results)
        success_rate = 100 * n_converged / n_seeds

        print(f"\n  Summary for alpha={alpha:.4f}:")
        print(f"    Success rate: {n_converged}/{n_seeds} ({success_rate:.1f}%)")

        if n_converged > 0:
            converged_results = [r for r in alpha_results if r['nash_converged']]
            avg_iters = np.mean([r['num_iters'] for r in converged_results])
            avg_welfare = np.mean([r['final_welfare'] for r in converged_results])
            avg_expl = np.mean([r['total_expl'] for r in converged_results])

            print(f"    Avg iterations (converged): {avg_iters:.1f}")
            print(f"    Avg welfare (converged): {avg_welfare:.4f}")
            print(f"    Avg exploitability (converged): {avg_expl:.2e}")
        else:
            # Show best exploitability achieved
            best_expl = min(r['total_expl'] for r in alpha_results)
            print(f"    Best exploitability achieved: {best_expl:.2e}")

    # Overall summary
    df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    summary_stats = []
    for alpha in alpha_values:
        alpha_df = df[df['alpha'] == alpha]
        n_total = len(alpha_df)
        n_converged = alpha_df['nash_converged'].sum()
        success_rate = 100 * n_converged / n_total

        if n_converged > 0:
            converged_df = alpha_df[alpha_df['nash_converged']]
            avg_iters = converged_df['num_iters'].mean()
            avg_welfare = converged_df['final_welfare'].mean()
            avg_expl = converged_df['total_expl'].mean()
        else:
            avg_iters = np.nan
            avg_welfare = np.nan
            avg_expl = alpha_df['total_expl'].min()  # Best attempt

        summary_stats.append({
            'alpha': alpha,
            'success_rate': success_rate,
            'n_converged': n_converged,
            'n_total': n_total,
            'avg_iters': avg_iters,
            'avg_welfare': avg_welfare,
            'avg_expl': avg_expl,
        })

    summary_df = pd.DataFrame(summary_stats)

    print("\n" + summary_df.to_string(index=False))

    # Find best learning rate
    best_idx = summary_df['success_rate'].idxmax()
    best_alpha = summary_df.loc[best_idx, 'alpha']
    best_success_rate = summary_df.loc[best_idx, 'success_rate']

    print(f"\n{'='*80}")
    print(f"BEST LEARNING RATE: alpha = {best_alpha:.4f}")
    print(f"  Success rate: {best_success_rate:.1f}%")
    if summary_df.loc[best_idx, 'n_converged'] > 0:
        print(f"  Avg iterations: {summary_df.loc[best_idx, 'avg_iters']:.1f}")
        print(f"  Avg welfare: {summary_df.loc[best_idx, 'avg_welfare']:.4f}")
    print(f"{'='*80}")

    return df, summary_df


if __name__ == "__main__":
    # Test a range of learning rates
    # Start with baseline (0.01) and test both higher and lower
    alpha_values = [
        0.001,   # Very low
        0.003,   # Low
        0.005,   # Low-medium
        0.007,   # Medium-low
        0.010,   # Baseline (from previous tests)
        0.015,   # Medium-high
        0.020,   # High
        0.030,   # Very high
    ]

    # Run sweep with 10 seeds per learning rate
    results_df, summary_df = lr_sweep(
        alpha_values=alpha_values,
        n_seeds=10,
        max_iter=2000
    )

    # Save results
    results_df.to_csv('lr_sweep_results_4qubit.csv', index=False)
    summary_df.to_csv('lr_sweep_summary_4qubit.csv', index=False)
    print(f"\nResults saved to lr_sweep_results_4qubit.csv and lr_sweep_summary_4qubit.csv")
