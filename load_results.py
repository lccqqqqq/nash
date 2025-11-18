"""
Utility functions for loading and analyzing saved optimization results.

Usage:
    from load_results import load_pickle, load_csv, analyze_run

    # Load full data with states
    metric_logs = load_pickle('data/opt_fid_state_chi4_lr1e-02_steps100_alpha3e-02_20250117_123456.pkl')

    # Load just metrics
    df = load_csv('data/opt_fid_state_chi4_lr1e-02_steps100_alpha3e-02_20250117_123456.csv')

    # Quick analysis
    analyze_run('data/opt_fid_state_chi4_lr1e-02_steps100_alpha3e-02_20250117_123456.pkl')
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_pickle(filepath):
    """
    Load full metric logs including MPS states from pickle file.

    Args:
        filepath: Path to .pkl file

    Returns:
        list of dict: metric_logs with keys:
            - 'energy': np.ndarray - Final energies for each player
            - 'welfare': float - Total welfare
            - 'state': list[np.ndarray] - MPS tensors (Psi)
            - 'ent_params': np.ndarray - Entanglement parameters (I1-I5)
    """
    with open(filepath, 'rb') as f:
        metric_logs = pickle.load(f)
    print(f"Loaded {len(metric_logs)} iterations from {filepath}")
    return metric_logs


def load_csv(filepath):
    """
    Load metrics DataFrame from CSV file (no states).

    Args:
        filepath: Path to .csv file

    Returns:
        pd.DataFrame with columns:
            - welfare, energy_player_0/1/2, I1-I5
    """
    df = pd.read_csv(filepath, index_col='iteration')
    print(f"Loaded {len(df)} iterations from {filepath}")
    return df


def get_state_at_iteration(metric_logs, iteration):
    """
    Extract MPS state at specific iteration.

    Args:
        metric_logs: Loaded from pickle
        iteration: Index of iteration (0-indexed)

    Returns:
        list[np.ndarray]: MPS tensors [Psi_0, Psi_1, Psi_2]
    """
    return metric_logs[iteration]['state']


def analyze_run(pickle_filepath, plot=True):
    """
    Quick analysis of an optimization run.

    Args:
        pickle_filepath: Path to .pkl file
        plot: Whether to show plots (default: True)

    Returns:
        dict: Summary statistics
    """
    # Load data
    metric_logs = load_pickle(pickle_filepath)

    # Extract metrics
    n_iters = len(metric_logs)
    welfares = [log['welfare'] for log in metric_logs]
    energies = np.array([log['energy'] for log in metric_logs])
    ent_params = np.array([log['ent_params'] for log in metric_logs])

    # Compute statistics
    stats = {
        'n_iterations': n_iters,
        'initial_welfare': welfares[0],
        'final_welfare': welfares[-1],
        'best_welfare': max(welfares),
        'best_welfare_iter': np.argmax(welfares),
        'welfare_improvement': welfares[-1] - welfares[0],
        'final_energies': energies[-1],
        'final_ent_params': ent_params[-1],
    }

    # Print summary
    print("\n" + "="*60)
    print("Optimization Run Summary")
    print("="*60)
    print(f"Iterations: {stats['n_iterations']}")
    print(f"\nWelfare:")
    print(f"  Initial: {stats['initial_welfare']:.4f}")
    print(f"  Final:   {stats['final_welfare']:.4f}")
    print(f"  Best:    {stats['best_welfare']:.4f} (iter {stats['best_welfare_iter']})")
    print(f"  Change:  {stats['welfare_improvement']:+.4f}")
    print(f"\nFinal Player Energies:")
    for i, e in enumerate(stats['final_energies']):
        print(f"  Player {i}: {e:.4f}")
    print(f"\nFinal Entanglement Parameters:")
    for i, param in enumerate(stats['final_ent_params'], 1):
        print(f"  I{i}: {param:.6f}")
    print("="*60)

    # Plot if requested
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Welfare over time
        axes[0, 0].plot(welfares, 'b-', linewidth=2)
        axes[0, 0].axhline(stats['best_welfare'], color='r', linestyle='--', alpha=0.5, label='Best')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Welfare')
        axes[0, 0].set_title('Welfare Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Individual energies
        for i in range(energies.shape[1]):
            axes[0, 1].plot(energies[:, i], label=f'Player {i}')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Energy')
        axes[0, 1].set_title('Individual Player Energies')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Entanglement parameters (I1-I3)
        for i in range(3):
            axes[1, 0].plot(ent_params[:, i], label=f'I{i+1}')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Single-Party Purities (I1-I3)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Entanglement parameters (I4-I5)
        axes[1, 1].plot(ent_params[:, 3], label='I4 (2-party)', color='orange')
        axes[1, 1].plot(ent_params[:, 4], label='I5 (3-party)', color='red')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Multi-Party Entanglement (I4-I5)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return stats


def compare_runs(pickle_filepaths, labels=None):
    """
    Compare multiple optimization runs.

    Args:
        pickle_filepaths: List of paths to .pkl files
        labels: Optional list of labels for each run

    Returns:
        pd.DataFrame: Comparison table
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(pickle_filepaths))]

    results = []
    for filepath, label in zip(pickle_filepaths, labels):
        metric_logs = load_pickle(filepath)
        welfares = [log['welfare'] for log in metric_logs]

        results.append({
            'Run': label,
            'Iterations': len(metric_logs),
            'Initial Welfare': welfares[0],
            'Final Welfare': welfares[-1],
            'Best Welfare': max(welfares),
            'Improvement': welfares[-1] - welfares[0],
        })

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("Run Comparison")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

    return df


def extract_best_state(pickle_filepath):
    """
    Extract the MPS state from the iteration with best welfare.

    Args:
        pickle_filepath: Path to .pkl file

    Returns:
        tuple: (best_state, best_iteration, best_welfare)
            - best_state: list[np.ndarray] - MPS tensors
            - best_iteration: int - Iteration index
            - best_welfare: float - Welfare value
    """
    metric_logs = load_pickle(pickle_filepath)
    welfares = [log['welfare'] for log in metric_logs]
    best_iter = np.argmax(welfares)
    best_state = metric_logs[best_iter]['state']
    best_welfare = welfares[best_iter]

    print(f"Best state found at iteration {best_iter} with welfare {best_welfare:.4f}")

    return best_state, best_iter, best_welfare


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_results.py <path_to_pkl_file>")
        print("\nExample:")
        print("  python load_results.py data/opt_fid_state_chi4_lr1e-02_steps100_alpha3e-02_20250117_123456.pkl")
        sys.exit(1)

    # Analyze the specified run
    analyze_run(sys.argv[1], plot=True)
