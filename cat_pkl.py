"""
Concatenate all pickle files from equilibrium statistics runs into a single pickle file.
Supports both gather_equilibrium_stats.py output and legacy qpd4.py output.
Optionally removes the original split files after successful concatenation.

Usage:
    # Use defaults (data/qpd6_results directory)
    python cat_pkl.py

    # Specify custom directory and pattern
    python cat_pkl.py --input-dir data/equilibrium_stats --pattern "eqdata_randstates_*.pkl"

    # Keep original files
    python cat_pkl.py --no-remove
"""

import pandas as pd
import os
import glob
from datetime import datetime
import secrets
import argparse

def concatenate_pickle_files(
    input_dir: str = "data/qpd4_results",
    output_filename: str = None,
    pattern: str = "eqdata_randstates_*.pkl",
    remove_originals: bool = True,
):
    """
    Concatenate all pickle files matching the pattern into one file.

    Args:
        input_dir: Directory containing the pickle files
        output_filename: Name for the concatenated file (auto-generated if None)
        pattern: Glob pattern to match pickle files
        remove_originals: If True, delete original files after concatenation
    """

    # Find all matching pickle files
    file_pattern = os.path.join(input_dir, pattern)
    pkl_files = sorted(glob.glob(file_pattern))

    if len(pkl_files) == 0:
        print(f"No files found matching pattern: {file_pattern}")
        return None

    print(f"Found {len(pkl_files)} pickle files to concatenate")

    # Load all dataframes
    dfs = []
    total_rows = 0

    print("\nLoading pickle files...")
    for i, filepath in enumerate(pkl_files, 1):
        try:
            df = pd.read_pickle(filepath)
            dfs.append(df)
            total_rows += len(df)

            if i % 50 == 0 or i == len(pkl_files):
                print(f"  Loaded {i}/{len(pkl_files)} files ({total_rows} rows so far)")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue

    if len(dfs) == 0:
        print("No dataframes could be loaded!")
        return None

    # Concatenate all dataframes
    print(f"\nConcatenating {len(dfs)} dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Combined dataframe shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")

    # Generate output filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_id = secrets.token_hex(4)
        output_filename = f"eqdata_randstates_combined_{timestamp}_{file_id}.pkl"

    # Save concatenated dataframe
    output_path = os.path.join(input_dir, output_filename)
    combined_df.to_pickle(output_path)

    print(f"\n{'='*70}")
    print(f"Concatenated data saved to: {output_path}")
    print(f"Total samples: {len(combined_df)}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"{'='*70}")

    # Summary statistics
    print(f"\nSummary Statistics:")

    if 'nash_equilibrium' in combined_df.columns:
        n_converged = combined_df['nash_equilibrium'].sum()
        print(f"  Nash equilibrium converged: {n_converged}/{len(combined_df)} ({100*n_converged/len(combined_df):.1f}%)")

    if 'nash_state' in combined_df.columns:
        n_local = combined_df['nash_state'].sum()
        print(f"  Local convergence: {n_local}/{len(combined_df)} ({100*n_local/len(combined_df):.1f}%)")

    if 'num_iters' in combined_df.columns:
        print(f"  Avg iterations: {combined_df['num_iters'].mean():.1f}")
        print(f"  Max iterations: {combined_df['num_iters'].max()}")

    if 'expl' in combined_df.columns:
        # Handle array column - compute mean exploitability across players and samples
        import numpy as np
        mean_expl = combined_df['expl'].apply(lambda x: np.mean(x) if hasattr(x, '__iter__') else x).mean()
        max_expl = combined_df['expl'].apply(lambda x: np.max(x) if hasattr(x, '__iter__') else x).max()
        print(f"  Avg exploitability: {mean_expl:.6f}")
        print(f"  Max exploitability: {max_expl:.6f}")

    if 'energy' in combined_df.columns:
        # Energy is array per player, compute mean welfare (sum of energies)
        import numpy as np
        mean_welfare = combined_df['energy'].apply(lambda x: np.sum(x) if hasattr(x, '__iter__') else x).mean()
        max_welfare = combined_df['energy'].apply(lambda x: np.sum(x) if hasattr(x, '__iter__') else x).max()
        print(f"  Avg welfare (sum of energies): {mean_welfare:.3f}")
        print(f"  Max welfare: {max_welfare:.3f}")

    # Legacy support: check for welfare column from opt_fid_state results
    if 'welfare' in combined_df.columns:
        print(f"  Avg welfare: {combined_df['welfare'].mean():.3f}")
        print(f"  Max welfare: {combined_df['welfare'].max():.3f}")

    # Remove original files if requested
    if remove_originals:
        print(f"\nRemoving {len(pkl_files)} original pickle files...")
        for filepath in pkl_files:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"  Error removing {filepath}: {e}")
        print("Original files removed.")

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Concatenate pickle files from equilibrium statistics runs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-dir', type=str, default='data/qpd6_results',
                        help='Directory containing pickle files to concatenate')
    parser.add_argument('--pattern', type=str, default='eqdata_randstates_*.pkl',
                        help='Glob pattern to match pickle files')
    parser.add_argument('--output-filename', type=str, default=None,
                        help='Output filename (auto-generated if not specified)')
    parser.add_argument('--no-remove', action='store_true',
                        help='Keep original files after concatenation')
    parser.add_argument('--show-head', action='store_true',
                        help='Display first few rows after concatenation')

    args = parser.parse_args()

    # Run concatenation
    df = concatenate_pickle_files(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_filename=args.output_filename,
        remove_originals=not args.no_remove,
    )

    if df is not None and args.show_head:
        print("\nFirst few rows:")
        print(df.head())
