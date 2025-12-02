"""
Concatenate all pickle files from qpd4.py runs into a single pickle file.
Removes the original split files after successful concatenation.
"""

import pandas as pd
import os
import glob
from datetime import datetime
import secrets

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
    if 'nash_equilibrium' in combined_df.columns:
        n_converged = combined_df['nash_equilibrium'].sum()
        print(f"\nSummary:")
        print(f"  Nash equilibrium converged: {n_converged}/{len(combined_df)} ({100*n_converged/len(combined_df):.1f}%)")

    if 'num_iters' in combined_df.columns:
        print(f"  Avg iterations: {combined_df['num_iters'].mean():.1f}")
        print(f"  Max iterations: {combined_df['num_iters'].max()}")

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
    # Run concatenation
    df = concatenate_pickle_files(
        input_dir="data/qpd4_results",
        pattern="eqdata_randstates_*.pkl",
        remove_originals=True,  # Set to False to keep originals
    )

    if df is not None:
        print("\nFirst few rows:")
        print(df.head())
