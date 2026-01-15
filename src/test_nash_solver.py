#!/usr/bin/env python3
"""
Standalone Nash equilibrium solver tester with CLI support.

This script runs find_nash_eq1() or find_nash_eq1_with_retry() directly
without the outer optimization loop, useful for:
- Testing solver robustness
- Debugging convergence issues
- Running experiments to assess equilibrium existence
- Comparing different solver configurations

Usage examples:
    # Basic test with default 3-player game
    python src/test_nash_solver.py --alpha 0.01

    # Test with retry logic
    python src/test_nash_solver.py --use-retry --max-retries 20

    # Multiple random initializations
    python src/test_nash_solver.py --num-trials 100 --save-all

    # Test specific Hamiltonian
    python src/test_nash_solver.py --hamiltonian-type symmetric --num-players 4

    # Load state from file and test
    python src/test_nash_solver.py --load-state path/to/state.pkl
"""

import numpy as np
import argparse
import pickle
import os
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Import solver functions
from solver import find_nash_eq1, find_nash_eq1_with_retry, compute_exploitability
from game import (get_default_3players, get_default_cyclic_players,
                  get_perturbed_H_QPD, H_QPD)
from mps_utils import (get_rand_state_as_mps, get_ghz_state, get_product_state,
                       to_comp_basis, to_canonical_form)


def get_hamiltonian(hamiltonian_type, num_players, dtype=np.float64, **kwargs):
    """
    Get Hamiltonian based on type.

    Args:
        hamiltonian_type: 'qpd' (quantum prisoner's dilemma), 'symmetric',
                         'coordination', 'custom'
        num_players: Number of players
        dtype: Data type
        **kwargs: Additional parameters (e.g., non_commutative_norm)

    Returns:
        H: List of Hamiltonian tensors
    """
    if hamiltonian_type == 'qpd':
        # Quantum Prisoner's Dilemma (default or perturbed)
        eps = kwargs.get('non_commutative_norm', 0.0)
        if num_players == 3:
            if eps == 0:
                H = get_default_3players(dtype=dtype)
            else:
                Hs = get_perturbed_H_QPD(eps=eps, dtype=dtype)
                H = get_default_cyclic_players(L=num_players, Hs=Hs, dtype=dtype)
        else:
            Hs = get_perturbed_H_QPD(eps=eps, dtype=dtype)
            H = get_default_cyclic_players(L=num_players, Hs=Hs, dtype=dtype)

    elif hamiltonian_type == 'custom':
        # Load from file
        filepath = kwargs.get('hamiltonian_file')
        if filepath is None:
            raise ValueError("Must provide --hamiltonian-file for custom type")
        with open(filepath, 'rb') as f:
            H = pickle.load(f)

    else:
        raise ValueError(f"Unknown hamiltonian_type: {hamiltonian_type}")

    return H


def get_initial_state(state_type, num_players, chi, dtype=np.float64, **kwargs):
    """
    Get initial state based on type.

    Args:
        state_type: 'random', 'ghz', 'product', 'custom'
        num_players: Number of players
        chi: Bond dimension
        dtype: Data type
        **kwargs: Additional parameters (e.g., seed, state_file)

    Returns:
        Psi: Initial state as MPS
    """
    if state_type == 'random':
        seed = kwargs.get('seed')
        Psi = get_rand_state_as_mps(L=num_players, max_bond_dim=chi,
                                     dtype=dtype, seed=seed)

    elif state_type == 'ghz':
        Psi = get_ghz_state(L=num_players, dtype=dtype)

    elif state_type == 'product':
        # Product of |0⟩ states
        Psi = get_product_state(L=num_players, dtype=dtype)

    else:
        raise ValueError(f"Unknown state_type: {state_type}")

    # Convert to canonical form
    Psi = to_canonical_form(Psi, form='B')
    return Psi


def run_single_test(Psi, H, use_retry, args):
    """
    Run a single Nash equilibrium test.

    Returns:
        result: Dict with results from solver
        success: Bool indicating if Nash equilibrium was found
        final_alpha: Final learning rate used
    """
    if use_retry:
        result, success, final_alpha = find_nash_eq1_with_retry(
            Psi=Psi,
            H=H,
            max_iter=args.max_iter,
            base_alpha=args.alpha,
            max_alpha=args.max_alpha,
            min_alpha=args.min_alpha,
            max_retries=args.max_retries,
            expl_check_interval=args.expl_check_interval,
            expl_threshold=args.expl_threshold,
            expl_maxiter=args.expl_maxiter,
            real_strategies=args.real_strategies,
            return_history=args.return_history,
        )
    else:
        result = find_nash_eq1(
            Psi=Psi,
            H=H,
            max_iter=args.max_iter,
            alpha=args.alpha,
            convergence_threshold=args.convergence_threshold,
            expl_threshold=args.expl_threshold,
            use_tqdm=args.verbose,
            expl_check_interval=args.expl_check_interval,
            return_history=args.return_history,
            expl_maxiter=args.expl_maxiter,
            real_strategies=args.real_strategies,
        )
        success = result['nash_equilibrium']
        final_alpha = args.alpha

    return result, success, final_alpha


def main():
    parser = argparse.ArgumentParser(
        description='Test Nash equilibrium solver directly',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hamiltonian configuration
    parser.add_argument('--hamiltonian-type', type=str, default='qpd',
                        choices=['qpd', 'symmetric', 'coordination', 'custom'],
                        help='Type of game Hamiltonian')
    parser.add_argument('--hamiltonian-file', type=str, default=None,
                        help='Path to custom Hamiltonian file (for custom type)')
    parser.add_argument('--non-commutative-norm', type=float, default=0.0,
                        help='Non-commutative perturbation for QPD')

    # State configuration
    parser.add_argument('--state-type', type=str, default='random',
                        choices=['random', 'ghz', 'product', 'custom'],
                        help='Type of initial state')
    parser.add_argument('--state-file', type=str, default=None,
                        help='Path to custom state file (for custom type)')
    parser.add_argument('--num-players', type=int, default=3,
                        help='Number of players')
    parser.add_argument('--chi', type=int, default=2,
                        help='Bond dimension for MPS')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64', 'complex64', 'complex128'],
                        help='Data type for states and Hamiltonians')

    # Solver configuration
    parser.add_argument('--use-retry', action='store_true',
                        help='Use find_nash_eq1_with_retry instead of find_nash_eq1')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Maximum iterations for Nash solver')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Learning rate (base LR if using retry)')
    parser.add_argument('--max-alpha', type=float, default=1.0,
                        help='Maximum learning rate for retry')
    parser.add_argument('--min-alpha', type=float, default=0.001,
                        help='Minimum learning rate for retry')
    parser.add_argument('--max-retries', type=int, default=20,
                        help='Maximum number of retries')
    parser.add_argument('--convergence-threshold', type=float, default=1e-7,
                        help='Local convergence threshold (energy change)')
    parser.add_argument('--expl-threshold', type=float, default=5e-4,
                        help='Exploitability threshold for Nash equilibrium')
    parser.add_argument('--expl-check-interval', type=int, default=50,
                        help='Check exploitability every N iterations')
    parser.add_argument('--expl-maxiter', type=int, default=300,
                        help='Max iterations for exploitability computation')
    parser.add_argument('--real-strategies', action='store_true', default=True,
                        help='Use real strategies (exp(iY))')
    parser.add_argument('--no-real-strategies', dest='real_strategies',
                        action='store_false',
                        help='Use complex strategies (full SU(2))')

    # Experiment configuration
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Number of trials with different random initializations')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (base seed for multiple trials)')
    parser.add_argument('--return-history', action='store_true',
                        help='Return full iteration history')
    parser.add_argument('--save-all', action='store_true',
                        help='Save results for all trials (not just summary)')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='data/nash_solver_tests',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Show progress bar')

    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'complex64': np.complex64,
        'complex128': np.complex128,
    }
    dtype = dtype_map[args.dtype]

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Print configuration
    print("="*70)
    print("Nash Equilibrium Solver Test")
    print("="*70)
    print(f"Hamiltonian: {args.hamiltonian_type}")
    print(f"Initial state: {args.state_type}")
    print(f"Number of players: {args.num_players}")
    print(f"Bond dimension: {args.chi}")
    print(f"Data type: {args.dtype}")
    print(f"Solver: {'find_nash_eq1_with_retry' if args.use_retry else 'find_nash_eq1'}")
    print(f"Alpha: {args.alpha}" + (f" (range: [{args.min_alpha}, {args.max_alpha}])" if args.use_retry else ""))
    print(f"Max iterations: {args.max_iter}")
    print(f"Expl threshold: {args.expl_threshold}")
    print(f"Number of trials: {args.num_trials}")
    print("="*70)
    print()

    # Get Hamiltonian
    print("Loading Hamiltonian...")
    H = get_hamiltonian(
        args.hamiltonian_type,
        args.num_players,
        dtype=dtype,
        non_commutative_norm=args.non_commutative_norm,
        hamiltonian_file=args.hamiltonian_file
    )
    print(f"Hamiltonian loaded: {len(H)} players, shape {H[0].shape}")

    # Run trials
    results_list = []
    success_count = 0

    iterator = tqdm(range(args.num_trials), desc="Running trials") if args.num_trials > 1 else range(args.num_trials)

    for trial in iterator:
        trial_seed = (args.seed + trial) if args.seed is not None else None

        # Get initial state
        Psi = get_initial_state(
            args.state_type,
            args.num_players,
            args.chi,
            dtype=dtype,
            seed=trial_seed,
            state_file=args.state_file
        )

        # Run solver
        result, success, final_alpha = run_single_test(Psi, H, args.use_retry, args)

        # Collect results
        trial_result = {
            'trial': trial,
            'seed': trial_seed,
            'nash_state': result['nash_state'],
            'nash_equilibrium': result['nash_equilibrium'],
            'success': success,
            'num_iters': result['num_iters'],
            'final_alpha': final_alpha,
            'exploitability': result['expl'] if not args.return_history else result['expl'][-1],
            'energies': result['energy'] if not args.return_history else result['energy'][-1],
        }

        if args.save_all:
            trial_result['full_result'] = result

        results_list.append(trial_result)

        if success:
            success_count += 1

        # Print trial summary if verbose
        if args.verbose and args.num_trials > 1:
            expl = trial_result['exploitability']
            if isinstance(expl, np.ndarray):
                expl = np.sum(expl)
            print(f"  Trial {trial}: {'SUCCESS' if success else 'FAILED'}, "
                  f"iters={result['num_iters']}, expl={expl:.2e}")

    # Create summary
    print()
    print("="*70)
    print("Results Summary")
    print("="*70)
    print(f"Total trials: {args.num_trials}")
    print(f"Successful: {success_count} ({100*success_count/args.num_trials:.1f}%)")
    print(f"Failed: {args.num_trials - success_count}")

    if args.num_trials > 1:
        # Statistics
        expls = [np.sum(r['exploitability']) for r in results_list]
        iters = [r['num_iters'] for r in results_list]

        print(f"\nExploitability:")
        print(f"  Min: {np.min(expls):.2e}")
        print(f"  Mean: {np.mean(expls):.2e}")
        print(f"  Median: {np.median(expls):.2e}")
        print(f"  Max: {np.max(expls):.2e}")
        print(f"  Std: {np.std(expls):.2e}")

        print(f"\nIterations:")
        print(f"  Mean: {np.mean(iters):.1f}")
        print(f"  Median: {np.median(iters):.1f}")
        print(f"  Max: {np.max(iters)}")
    else:
        # Single trial details
        result = results_list[0]
        expl = result['exploitability']
        if isinstance(expl, np.ndarray):
            print(f"\nPer-player exploitability:")
            for i, e in enumerate(expl):
                print(f"  Player {i}: {e:.2e}")
            print(f"  Total: {np.sum(expl):.2e}")
        else:
            print(f"\nTotal exploitability: {expl:.2e}")

        print(f"\nPer-player energies:")
        energies = result['energies']
        for i, e in enumerate(energies):
            print(f"  Player {i}: {e:.6f}")
        print(f"  Total welfare: {np.sum(energies):.6f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary as JSON
    summary = {
        'config': vars(args),
        'timestamp': timestamp,
        'num_trials': args.num_trials,
        'success_count': success_count,
        'success_rate': success_count / args.num_trials,
    }

    if args.num_trials > 1:
        summary['exploitability_stats'] = {
            'min': float(np.min(expls)),
            'mean': float(np.mean(expls)),
            'median': float(np.median(expls)),
            'max': float(np.max(expls)),
            'std': float(np.std(expls)),
        }

    summary_file = os.path.join(args.save_dir, f"summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Save detailed results as pickle
    results_file = os.path.join(args.save_dir, f"results_{timestamp}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({
            'config': vars(args),
            'H': H,
            'results': results_list,
            'summary': summary,
        }, f)
    print(f"Detailed results saved to: {results_file}")

    # Save as DataFrame for easy analysis
    df = pd.DataFrame([
        {
            'trial': r['trial'],
            'success': r['success'],
            'nash_state': r['nash_state'],
            'nash_equilibrium': r['nash_equilibrium'],
            'num_iters': r['num_iters'],
            'final_alpha': r['final_alpha'],
            'total_exploitability': np.sum(r['exploitability']) if isinstance(r['exploitability'], np.ndarray) else r['exploitability'],
            'total_welfare': np.sum(r['energies']),
        }
        for r in results_list
    ])

    csv_file = os.path.join(args.save_dir, f"results_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Results table saved to: {csv_file}")

    print("="*70)

    # Return exit code based on success
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit(main())
