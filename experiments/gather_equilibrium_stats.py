"""
Gather equilibrium statistics by sampling random initial MPS states.

This script finds Nash equilibria from multiple random initializations and saves
the results for statistical analysis. It uses the differential best response
algorithm from solver.py to find equilibria for quantum games.

Usage:
    # Basic usage with defaults (3 players, 50 samples)
    python experiments/gather_equilibrium_stats.py

    # Customize parameters
    python experiments/gather_equilibrium_stats.py --num-players 4 --chi 64 --num-samples 100

    # Set seed for reproducibility
    python experiments/gather_equilibrium_stats.py --seed 42 --verbose

    # Run with custom save directory
    python experiments/gather_equilibrium_stats.py --save-dir data/my_stats --num-samples 20

Results are saved as pandas DataFrames in pickle format with UUID-based filenames.
Multiple runs can be concatenated using cat_pkl.py.
"""

import numpy as np
from solver import find_nash_eq1
from game import get_default_H
from mps_utils import get_rand_mps
import time
import uuid
import os
import pandas as pd
from tqdm import tqdm
import argparse


def gather_equilibrium_statistics(
    H: list[np.ndarray],
    num_players: int,
    chi: int = 32,
    dtype: np.dtype = np.float32,
    num_samples: int = 50,
    max_iter: int = 1000,
    alpha: float = 0.06,
    convergence_threshold: float = 1e-6,
    expl_threshold: float = 1e-3,
    use_tqdm: bool = False,
    expl_check_interval: int = 10,
    return_history: bool = False,
    save_dir: str = "data/equilibrium_stats",
    seed: int = None,
):
    """
    Gather equilibrium statistics from random initial states.

    For each sample, generates a random MPS state and runs the Nash equilibrium
    finder. Results are saved as a pandas DataFrame.

    Args:
        H: List of Hamiltonian tensors (one per player), shape (2,2,2,2,2,2) for 3 players
        num_players: Number of players in the game
        chi: Bond dimension for MPS representation (controls entanglement capacity)
        dtype: Data type for state initialization (np.float32 or np.float64)
        num_samples: Number of random initial states to sample
        max_iter: Maximum iterations for Nash equilibrium solver
        alpha: Learning rate for differential best response dynamics
        convergence_threshold: Local convergence threshold (energy change)
        expl_threshold: Global convergence threshold (exploitability)
        use_tqdm: Show progress bar
        expl_check_interval: How often to check exploitability
        return_history: Whether to return full iteration history (increases file size)
        save_dir: Directory to save results
        seed: Random seed for reproducibility (optional)

    Returns:
        List of result dictionaries from find_nash_eq1()
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    results = []
    iterator = tqdm(range(num_samples), desc="Sampling equilibria") if use_tqdm else range(num_samples)

    for i in iterator:
        # Generate random initial state
        Psi = get_rand_mps(L=num_players, chi=chi, d_phys=2, dtype=dtype)

        # Find Nash equilibrium
        result = find_nash_eq1(
            Psi=Psi,
            H=H,
            max_iter=max_iter,
            alpha=alpha,
            convergence_threshold=convergence_threshold,
            expl_threshold=expl_threshold,
            use_tqdm=False,  # Disable inner progress bar to avoid clutter
            expl_check_interval=expl_check_interval,
            return_history=return_history
        )
        results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save to disk
    os.makedirs(save_dir, exist_ok=True)
    filename = f"eqdata_randstates_{str(uuid.uuid4())[:8]}.pkl"
    filepath = os.path.join(save_dir, filename)
    df.to_pickle(filepath)

    print(f"\nResults saved to: {filepath}")
    print(f"Total samples: {len(df)}")

    # Print summary statistics
    if 'nash_equilibrium' in df.columns:
        n_converged = df['nash_equilibrium'].sum()
        print(f"Nash equilibrium converged: {n_converged}/{len(df)} ({100*n_converged/len(df):.1f}%)")

    if 'nash_state' in df.columns:
        n_local = df['nash_state'].sum()
        print(f"Local convergence: {n_local}/{len(df)} ({100*n_local/len(df):.1f}%)")

    if 'num_iters' in df.columns:
        print(f"Avg iterations: {df['num_iters'].mean():.1f}")
        print(f"Max iterations: {df['num_iters'].max()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Gather Nash equilibrium statistics from random initial states',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Game configuration
    parser.add_argument('--num-players', type=int, default=3,
                        help='Number of players in the game')
    parser.add_argument('--chi', type=int, default=32,
                        help='MPS bond dimension (controls entanglement capacity)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'],
                        help='Data type for state initialization')

    # Sampling configuration
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of random initial states to sample')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (optional)')

    # Nash solver configuration
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum iterations for Nash equilibrium solver')
    parser.add_argument('--alpha', type=float, default=0.06,
                        help='Learning rate for differential best response dynamics')
    parser.add_argument('--convergence-threshold', type=float, default=1e-6,
                        help='Local convergence threshold (energy change)')
    parser.add_argument('--expl-threshold', type=float, default=1e-3,
                        help='Global convergence threshold (exploitability)')
    parser.add_argument('--expl-check-interval', type=int, default=10,
                        help='How often to check exploitability during optimization')
    parser.add_argument('--return-history', action='store_true',
                        help='Store full iteration history (increases file size)')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='data/equilibrium_stats',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Show progress bar')

    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype = np.float32 if args.dtype == 'float32' else np.float64

    # Print configuration
    print("="*70)
    print("Equilibrium Statistics Gathering")
    print("="*70)
    print(f"Configuration:")
    print(f"  Number of players: {args.num_players}")
    print(f"  Bond dimension (chi): {args.chi}")
    print(f"  Data type: {args.dtype}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Random seed: {args.seed if args.seed is not None else 'None (random)'}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Alpha (learning rate): {args.alpha}")
    print(f"  Convergence threshold: {args.convergence_threshold}")
    print(f"  Exploitability threshold: {args.expl_threshold}")
    print(f"  Save directory: {args.save_dir}")
    print("="*70)

    # Get default Hamiltonian for the specified number of players
    try:
        H = get_default_H(num_players=args.num_players, dtype=dtype)
    except Exception as e:
        print(f"\nError: Could not load default Hamiltonian for {args.num_players} players.")
        print(f"Details: {e}")
        print(f"\nSupported player counts may be limited. Check game.py for available defaults.")
        return

    # Run the statistics gathering
    start_time = time.time()
    results = gather_equilibrium_statistics(
        H=H,
        num_players=args.num_players,
        chi=args.chi,
        dtype=dtype,
        num_samples=args.num_samples,
        max_iter=args.max_iter,
        alpha=args.alpha,
        convergence_threshold=args.convergence_threshold,
        expl_threshold=args.expl_threshold,
        use_tqdm=args.verbose,
        expl_check_interval=args.expl_check_interval,
        return_history=args.return_history,
        save_dir=args.save_dir,
        seed=args.seed,
    )
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Time per sample: {elapsed/args.num_samples:.2f} seconds")
    print("\nDone! Use cat_pkl.py to concatenate results from multiple runs.")


if __name__ == "__main__":
    main()
