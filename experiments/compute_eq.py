from src.game import get_default_cyclic_players
from src.solver import find_nash_eq1
import numpy as np
from src.mps_utils import from_comp_basis
from src.solver import kick_with_u, apply_u
from utils.misc import get_random_two_qubit_unitary
import pickle
import fcntl
from pathlib import Path


def rand_unitary_circuit(psi, depth: int, seed: int | None = None):
    """Apply random unitary circuit to a wavefunction.

    Args:
        psi: Either 1D array of shape (2^L,) or tensor of shape (2, 2, ..., 2)
        depth: Circuit depth
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)  # Set seed once at the beginning

    # Determine number of qubits
    if psi.ndim == 1:
        L = int(np.log2(len(psi)))
        psi = psi.reshape([2]*L)  # Reshape to tensor form
    else:
        L = psi.ndim

    for d in range(depth):
        for i in range(L):
            if i % 2 == d % 2:
                U = get_random_two_qubit_unitary(seed=None)  # Let RNG evolve naturally
                psi = apply_u(U.reshape(2, 2, 2, 2), psi, [i, np.mod(i+1, L)])

    return psi


def append_result_to_file(filename: str, result_dict: dict):
    """Append a result dictionary to a pickle file with file locking.

    Args:
        filename: Path to output file (will create if doesn't exist)
        result_dict: Dictionary containing result data
    """
    # Ensure parent directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Use a lock file to coordinate access
    lock_file = f"{filename}.lock"
    with open(lock_file, 'w') as lock:
        # Acquire exclusive lock
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # Read existing results
            if Path(filename).exists():
                with open(filename, 'rb') as f:
                    results = pickle.load(f)
            else:
                results = []

            # Append new result
            results.append(result_dict)

            # Write back to file
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        finally:
            # Release lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main(
    num_players: int,
    depth: int,
    alpha: float,
    output_file: str = None,
):
    H = get_default_cyclic_players(L=num_players)
    state = np.zeros([2]*num_players)
    state[(0,)*num_players] = 1.0
    state = rand_unitary_circuit(state, depth=depth, seed=None)
    state_mps = from_comp_basis(state.flatten(), L=num_players)
    state_mps = kick_with_u(state_mps)
    result = find_nash_eq1(state_mps, H=H, max_iter=3000, alpha=alpha, expl_threshold=1e-6, expl_check_interval=50, expl_maxiter=50, real_strategies=True, return_history=False, use_tqdm=True)

    # Prepare result dictionary
    result_dict = {
        'num_players': num_players,
        'depth': depth,
        'alpha': alpha,
        'num_iters': result['num_iters'],
        'energy': np.sum(result['energy']),
        'energy_per_player': result['energy'],
    }

    # Print results
    print(f"Energy: {result_dict['energy']:.6f}")
    print(f"Iterations: {result_dict['num_iters']}")

    # Write to file if specified
    if output_file:
        append_result_to_file(output_file, result_dict)
        print(f"Results written to {output_file}")

    return result_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute Nash equilibrium for random unitary scrambled states')
    parser.add_argument('--num-players', type=int, default=6, help='Number of players (qubits) (default: 6)')
    parser.add_argument('--depth', type=int, default=1, help='Circuit depth for random scrambling (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Nash solver learning rate (default: 0.1)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file for results (pickle format). If not specified, results are only printed.')

    args = parser.parse_args()

    main(num_players=args.num_players, depth=args.depth, alpha=args.alpha, output_file=args.output)