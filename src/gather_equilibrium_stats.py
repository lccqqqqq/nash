"""
Gather equilibrium statistics by sampling random initial MPS states.

This script finds Nash equilibria from multiple random initializations and saves
the results for statistical analysis. It uses the differential best response
algorithm with retry logic from solver.py to find equilibria for quantum games.

Usage:
    # Basic usage with defaults (3 players, 50 samples, Haar measure)
    python experiments/gather_equilibrium_stats.py

    # Customize parameters
    python experiments/gather_equilibrium_stats.py --num-players 4 --chi 64 --num-samples 100

    # Use random MPS instead of Haar measure
    python experiments/gather_equilibrium_stats.py --rand-measure rand_mps --num-samples 50

    # Set seed for Hamiltonian (states use rank-dependent seeds in MPI)
    python experiments/gather_equilibrium_stats.py --seed 42 --verbose

    # Run with custom save directory and retry parameters
    python experiments/gather_equilibrium_stats.py --save-dir data/my_stats --num-samples 20 \
        --alpha 0.06 --max-alpha 1.0 --max-retries 20

    # Use complex strategies instead of real
    python experiments/gather_equilibrium_stats.py --no-real-strategies

Results are saved as pandas DataFrames in pickle format with UUID-based filenames.
Multiple runs can be concatenated using cat_pkl.py.
"""

import numpy as np
from solver import find_nash_eq1_with_retry
from game import get_default_H, get_default_cyclic_players, H_QPD, get_perturbed_H_QPD
from mps_utils import get_rand_mps, get_rand_state_as_mps
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
    max_alpha: float = 1.0,
    max_retries: int = 20,
    convergence_threshold: float = 1e-6,
    expl_threshold: float = 1e-3,
    use_tqdm: bool = False,
    expl_check_interval: int = 10,
    return_history: bool = False,
    save_dir: str = "data/equilibrium_stats",
    seed: int = None,
    rand_measure: str = 'haar',
    expl_maxiter: int = 300,
    expl_seed: int = 42,
    real_strategies: bool = True,
):
    """
    Gather equilibrium statistics from random initial states.

    For each sample, generates a random MPS state and runs the Nash equilibrium
    finder with retry logic. Results are saved as a pandas DataFrame.

    Args:
        H: List of Hamiltonian tensors (one per player), shape (2,2,2,2,2,2) for 3 players
        num_players: Number of players in the game
        chi: Bond dimension for MPS representation (controls entanglement capacity)
        dtype: Data type for state initialization (np.float32 or np.float64)
        num_samples: Number of random initial states to sample
        max_iter: Maximum iterations for Nash equilibrium solver
        alpha: Base learning rate for differential best response dynamics
        max_alpha: Maximum learning rate for retries (default: 1.0)
        max_retries: Maximum number of retry attempts with increasing LR (default: 20)
        convergence_threshold: Local convergence threshold (energy change)
        expl_threshold: Global convergence threshold (exploitability)
        use_tqdm: Show progress bar
        expl_check_interval: How often to check exploitability
        return_history: Whether to return full iteration history (increases file size)
        save_dir: Directory to save results
        seed: Random seed for state generation (optional, None means use current RNG state)
        rand_measure: Random state measure ('haar' for Haar random states, 'rand_mps' for random MPS)
        expl_maxiter: Max iterations for differential evolution in exploitability computation (default: 300)
        expl_seed: Random seed for exploitability computation (default: 42)
        real_strategies: Use real strategies (exp(iY)) vs complex (full SU(2)) (default: True)

    Returns:
        List of result dictionaries from find_nash_eq1_with_retry()
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    results = []
    iterator = tqdm(range(num_samples), desc="Sampling equilibria") if use_tqdm else range(num_samples)

    for i in iterator:
        # Generate random initial state
        if rand_measure == 'haar':
            Psi = get_rand_state_as_mps(L=num_players, max_bond_dim=chi, dtype=dtype)
        elif rand_measure == 'rand_mps':
            Psi = get_rand_mps(L=num_players, chi=chi, dtype=dtype)
        else:
            raise ValueError(f"Invalid random measure: {rand_measure}, must be 'haar' or 'rand_mps'")

        # Find Nash equilibrium with retry logic
        result, success, final_alpha = find_nash_eq1_with_retry(
            Psi=Psi,
            H=H,
            max_iter=max_iter,
            base_alpha=alpha,
            max_alpha=max_alpha,
            max_retries=max_retries,
            expl_check_interval=expl_check_interval,
            expl_maxiter=expl_maxiter,
            real_strategies=real_strategies,
            return_history=return_history,
        )

        # Add retry metadata to result
        result['retry_success'] = success
        result['final_alpha'] = final_alpha
        result['base_alpha'] = alpha
        result['max_alpha'] = max_alpha
        result['max_retries'] = max_retries

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

    if 'retry_success' in df.columns:
        n_retry_success = df['retry_success'].sum()
        print(f"Retry success rate: {n_retry_success}/{len(df)} ({100*n_retry_success/len(df):.1f}%)")

    if 'final_alpha' in df.columns:
        print(f"Final alpha - Mean: {df['final_alpha'].mean():.4f}, Min: {df['final_alpha'].min():.4f}, Max: {df['final_alpha'].max():.4f}")

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
    parser.add_argument('--non-commutative-norm', type=float, default=0.0,
                        help='Non-commutative norm for the payoff')
    parser.add_argument('--num-players', type=int, default=3,
                        help='Number of players in the game')
    parser.add_argument('--chi', type=int, default=32,
                        help='MPS bond dimension (controls entanglement capacity)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'complex64'],
                        help='Data type for state initialization')

    # Sampling configuration
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of random initial states to sample')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for Hamiltonian generation (states use seed+rank in MPI)')
    parser.add_argument('--rand-measure', type=str, default='haar', choices=['haar', 'rand_mps'],
                        help="Random state measure: 'haar' for Haar random states, 'rand_mps' for random MPS")

    # Nash solver configuration
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum iterations for Nash equilibrium solver')
    parser.add_argument('--alpha', type=float, default=0.06,
                        help='Base learning rate for differential best response dynamics')
    parser.add_argument('--max-alpha', type=float, default=1.0,
                        help='Maximum learning rate for retry attempts')
    parser.add_argument('--max-retries', type=int, default=20,
                        help='Maximum number of retry attempts with increasing LR')
    parser.add_argument('--convergence-threshold', type=float, default=1e-6,
                        help='Local convergence threshold (energy change)')
    parser.add_argument('--expl-threshold', type=float, default=1e-3,
                        help='Global convergence threshold (exploitability)')
    parser.add_argument('--expl-check-interval', type=int, default=10,
                        help='How often to check exploitability during optimization')
    parser.add_argument('--expl-maxiter', type=int, default=300,
                        help='Max iterations for differential evolution in exploitability computation')
    parser.add_argument('--expl-seed', type=int, default=42,
                        help='Random seed for exploitability computation (for reproducibility)')
    parser.add_argument('--real-strategies', action='store_true', default=True,
                        help='Use real strategies (exp(iY)) vs complex (full SU(2))')
    parser.add_argument('--no-real-strategies', dest='real_strategies', action='store_false',
                        help='Use complex strategies (full SU(2)) instead of real')
    parser.add_argument('--return-history', action='store_true',
                        help='Store full iteration history (increases file size)')

    # Output configuration
    parser.add_argument('--save-dir', type=str, default='data/equilibrium_stats',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Show progress bar')

    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype = np.float32 if args.dtype == 'float32' else np.complex64

    # Detect MPI rank if available
    try:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()
    except (ImportError, Exception):
        mpi_rank = 0
        mpi_size = 1

    # Set random seed BEFORE generating Hamiltonian (so all MPI processes get same H)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Print configuration
    print("="*70)
    print("Equilibrium Statistics Gathering")
    print("="*70)
    print(f"Configuration:")
    print(f"  Number of players: {args.num_players}")
    print(f"  Bond dimension (chi): {args.chi}")
    print(f"  Data type: {args.dtype}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Random measure: {args.rand_measure}")
    print(f"  Hamiltonian seed: {args.seed if args.seed is not None else 'None (random)'}")
    if mpi_size > 1:
        print(f"  MPI: {mpi_size} processes detected")
        print(f"  State seed: Will use {args.seed} + rank (different per MPI process)" if args.seed is not None else "  State seed: Random")
    else:
        print(f"  State seed: {args.seed if args.seed is not None else 'Random'}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Alpha (base learning rate): {args.alpha}")
    print(f"  Max alpha (retry): {args.max_alpha}")
    print(f"  Max retries: {args.max_retries}")
    print(f"  Real strategies: {args.real_strategies}")
    print(f"  Convergence threshold: {args.convergence_threshold}")
    print(f"  Exploitability threshold: {args.expl_threshold}")
    print(f"  Exploitability max iterations: {args.expl_maxiter}")
    print(f"  Exploitability seed: {args.expl_seed}")
    print(f"  Save directory: {args.save_dir}")
    print("="*70)

    # Get default Hamiltonian for the specified number of players
    try:
        Hs = get_perturbed_H_QPD(eps=args.non_commutative_norm, dtype=dtype)
        H = get_default_cyclic_players(L=args.num_players, Hs=Hs, dtype=dtype)
        if args.non_commutative_norm > 0:
            print(f"CRITICAL: In this run, implement non-commutative payoff")
            print(f"  Non-commutative norm: {args.non_commutative_norm}")
            print(f"  Hamiltonian: {Hs}")
    except Exception as e:
        print(f"\nError: Could not load default Hamiltonian for {args.num_players} players.")
        print(f"Details: {e}")
        print(f"\nSupported player counts may be limited. Check game.py for available defaults.")
        return

    # Reseed with rank-dependent seed for state generation
    # This ensures each MPI process samples different initial states
    if args.seed is not None:
        state_seed = args.seed + mpi_rank
        np.random.seed(state_seed)
        print(f"\nMPI rank {mpi_rank}/{mpi_size}: Using state seed = {state_seed}")

    # Run the statistics gathering
    # Note: seed was already used above to generate fixed Hamiltonian
    # After Hamiltonian generation, we reseeded with rank-dependent seed
    # Don't pass seed here so that initial states use the rank-dependent RNG state
    start_time = time.time()
    results = gather_equilibrium_statistics(
        H=H,
        num_players=args.num_players,
        chi=args.chi,
        dtype=dtype,
        num_samples=args.num_samples,
        max_iter=args.max_iter,
        alpha=args.alpha,
        max_alpha=args.max_alpha,
        max_retries=args.max_retries,
        convergence_threshold=args.convergence_threshold,
        expl_threshold=args.expl_threshold,
        use_tqdm=args.verbose,
        expl_check_interval=args.expl_check_interval,
        return_history=args.return_history,
        save_dir=args.save_dir,
        seed=None,  # Don't reset seed - let states be randomly generated
        rand_measure=args.rand_measure,
        expl_maxiter=args.expl_maxiter,
        expl_seed=args.expl_seed,
        real_strategies=args.real_strategies,
    )
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Time per sample: {elapsed/args.num_samples:.2f} seconds")
    print("\nDone! Use cat_pkl.py to concatenate results from multiple runs.")


if __name__ == "__main__":
    main()
