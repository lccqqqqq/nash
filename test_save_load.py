"""
Simple test script to verify UUID-based save/load functionality.
"""

import numpy as np
import os
import sys

# Import functions from solver and load_results
from solver import save_results, metrics_to_dataframe
from load_results import load_result, get_metadata, get_dataframe, get_final_state, find_results
from mps_utils import get_rand_mps

def test_round_trip():
    """Test saving and loading with new UUID format."""
    print("="*60)
    print("Testing UUID-based Save/Load Round-Trip")
    print("="*60)

    # Create test data
    print("\n1. Creating test data...")
    num_players = 3
    chi = 2
    Psi = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=42, dtype=np.float64)

    # Create mock metric_logs
    metric_logs = []
    for i in range(5):
        metric_logs.append({
            'energy': np.array([0.1 * i, 0.2 * i, 0.3 * i]),
            'welfare': 0.6 * i,
            'state': Psi,
            'ent_params': np.array([0.5, 0.5, 0.5, 0.25, 0.125]),
        })

    print(f"  Created {len(metric_logs)} iterations of test data")
    print(f"  MPS: L={num_players}, chi={chi}")

    # Save using new function
    print("\n2. Saving with save_results()...")
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

    print(f"  Saved to: {filepath}")
    print(f"  UUID: {run_uuid}")

    # Load using new function
    print("\n3. Loading with load_result()...")
    results = load_result(filepath)

    # Verify structure
    print("\n4. Verifying structure...")
    assert 'metadata' in results, "Missing 'metadata' key"
    assert 'metric_logs' in results, "Missing 'metric_logs' key"
    assert 'dataframe' in results, "Missing 'dataframe' key"
    print("  ✓ All required keys present")

    # Verify metadata
    print("\n5. Verifying metadata...")
    metadata = results['metadata']
    assert metadata['uuid'] == run_uuid, "UUID mismatch"
    assert metadata['chi'] == chi, "chi mismatch"
    assert metadata['num_players'] == num_players, "num_players mismatch"
    assert metadata['perturbation_method'] == 'unitary', "perturbation_method mismatch"
    assert metadata['num_iterations'] == len(metric_logs), "num_iterations mismatch"
    print("  ✓ Metadata verified")
    print(f"    - UUID: {metadata['uuid']}")
    print(f"    - Chi: {metadata['chi']}")
    print(f"    - Num players: {metadata['num_players']}")
    print(f"    - Final welfare: {metadata['final_welfare']:.4f}")

    # Verify metric_logs
    print("\n6. Verifying metric_logs...")
    assert len(results['metric_logs']) == len(metric_logs), "metric_logs length mismatch"
    assert np.allclose(results['metric_logs'][0]['energy'], metric_logs[0]['energy']), "energy mismatch"
    print("  ✓ metric_logs verified")

    # Verify DataFrame
    print("\n7. Verifying pre-computed DataFrame...")
    df = results['dataframe']
    assert len(df) == len(metric_logs), "DataFrame length mismatch"
    assert 'welfare' in df.columns, "Missing 'welfare' column"
    assert 'I1' in df.columns, "Missing entanglement parameters"
    print("  ✓ DataFrame verified")
    print(f"    - Shape: {df.shape}")
    print(f"    - Columns: {list(df.columns)}")

    # Test convenience functions
    print("\n8. Testing convenience functions...")
    test_metadata = get_metadata(filepath)
    assert test_metadata['uuid'] == run_uuid, "get_metadata() failed"
    print("  ✓ get_metadata() works")

    test_df = get_dataframe(filepath)
    assert len(test_df) == len(metric_logs), "get_dataframe() failed"
    print("  ✓ get_dataframe() works")

    test_state = get_final_state(filepath)
    assert len(test_state) == num_players, "get_final_state() failed"
    print("  ✓ get_final_state() works")

    # Test search function
    print("\n9. Testing find_results()...")
    matches = find_results(save_dir, chi=chi, perturbation_method='unitary')
    assert filepath in matches, "find_results() didn't find saved file"
    print(f"  ✓ find_results() works (found {len(matches)} match(es))")

    # Cleanup
    print("\n10. Cleaning up test file...")
    os.remove(filepath)
    if os.path.exists(save_dir) and not os.listdir(save_dir):
        os.rmdir(save_dir)
    print("  ✓ Test file removed")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)

if __name__ == '__main__':
    try:
        test_round_trip()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
