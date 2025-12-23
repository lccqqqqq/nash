"""
Test that the refactored save/load system works for 5+ players.
"""

import numpy as np
import os
from solver import save_results, metrics_to_dataframe
from load_results import load_result, get_metadata
from mps_utils import get_rand_mps

def test_5_players():
    """Test saving and loading with 5 players (no entanglement params)."""
    print("="*60)
    print("Testing 5-Player Compatibility")
    print("="*60)

    # Create test data for 5 players
    print("\n1. Creating 5-player test data...")
    num_players = 5
    chi = 2
    Psi = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=42, dtype=np.float64)

    # Create mock metric_logs WITHOUT entanglement parameters
    metric_logs = []
    for i in range(3):
        metric_logs.append({
            'energy': np.array([0.1 * i] * num_players),
            'welfare': 0.5 * num_players * i,
            'state': Psi,
            'ent_params': None,  # None for 5+ players
        })

    print(f"  Created {len(metric_logs)} iterations")
    print(f"  MPS: L={num_players}, chi={chi}")
    print(f"  ent_params: None (expected for 5+ players)")

    # Test metrics_to_dataframe with None ent_params
    print("\n2. Testing metrics_to_dataframe()...")
    df = metrics_to_dataframe(metric_logs, include_state=False, include_ent_params=True)
    print(f"  ✓ DataFrame created successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Check that no entanglement columns are present
    ent_columns = [col for col in df.columns if col in ['I1', 'I2', 'I3', 'I4', 'I5', 'H', 'L', 'M', 'Dxt']]
    assert len(ent_columns) == 0, f"Unexpected entanglement columns: {ent_columns}"
    print(f"  ✓ No entanglement columns (correct for 5+ players)")

    # Save using new function
    print("\n3. Testing save_results()...")
    save_dir = 'test_data'
    filepath, run_uuid = save_results(
        save_dir=save_dir,
        Psi=Psi,
        metric_logs=metric_logs,
        chi=chi,
        num_players=num_players,
        max_num_steps=100,
        eps=0.01,
        num_perturbations=20,
        subroutine_max_iter=1000,
        subroutine_lr=0.009,
        perturbation_method='unitary',
    )

    print(f"  ✓ Saved to: {filepath}")
    assert 'qpd5_' in filepath, "Filename should contain 'qpd5_'"

    # Load and verify
    print("\n4. Testing load_result()...")
    results = load_result(filepath)

    assert results['metadata']['num_players'] == 5, "num_players mismatch"
    assert results['metadata']['chi'] == chi, "chi mismatch"
    assert len(results['dataframe']) == 3, "DataFrame length mismatch"

    # Check that no entanglement columns in saved DataFrame
    ent_columns = [col for col in results['dataframe'].columns if col in ['I1', 'I2', 'I3', 'I4', 'I5', 'H', 'L', 'M', 'Dxt']]
    assert len(ent_columns) == 0, f"Unexpected entanglement columns in saved DataFrame: {ent_columns}"

    print(f"  ✓ Load successful")
    print(f"  ✓ Metadata verified (num_players={results['metadata']['num_players']})")
    print(f"  ✓ No entanglement columns in saved DataFrame")

    # Cleanup
    print("\n5. Cleaning up...")
    os.remove(filepath)
    if os.path.exists(save_dir) and not os.listdir(save_dir):
        os.rmdir(save_dir)
    print("  ✓ Test file removed")

    print("\n" + "="*60)
    print("✅ 5-Player test passed!")
    print("="*60)

if __name__ == '__main__':
    try:
        test_5_players()
    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
