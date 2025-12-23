"""
Test that seed parameter is properly tracked and saved in metadata.
"""

import numpy as np
import os
from solver import save_results
from load_results import load_result, get_metadata
from mps_utils import get_rand_mps

def test_seed_tracking():
    """Test that seeds are properly saved in metadata."""
    print("="*60)
    print("Testing Seed Tracking in Metadata")
    print("="*60)

    num_players = 3
    chi = 2
    save_dir = 'test_data'

    # Test 1: Verify seed is saved in metadata
    print("\n1. Test seed is saved in metadata...")
    test_seed = 42
    Psi = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=test_seed, dtype=np.float64)

    metric_logs = [{
        'energy': np.array([0.1, 0.2, 0.3]),
        'welfare': 0.6,
        'state': Psi,
        'ent_params': np.array([0.5, 0.5, 0.5, 0.25, 0.125]),
    }]

    filepath, run_uuid = save_results(
        save_dir=save_dir,
        Psi=Psi,
        metric_logs=metric_logs,
        chi=chi,
        num_players=num_players,
        seed=test_seed,
    )

    # Load and check
    metadata = get_metadata(filepath)
    assert metadata['seed'] == test_seed, f"Seed mismatch: {metadata['seed']} != {test_seed}"
    print(f"  ✓ Seed {test_seed} correctly saved in metadata")

    # Cleanup
    os.remove(filepath)

    # Test 2: Verify None seed is handled
    print("\n2. Test None seed is handled...")
    Psi2 = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=None, dtype=np.float64)

    metric_logs2 = [{
        'energy': np.array([0.1, 0.2, 0.3]),
        'welfare': 0.6,
        'state': Psi2,
        'ent_params': np.array([0.5, 0.5, 0.5, 0.25, 0.125]),
    }]

    filepath2, run_uuid2 = save_results(
        save_dir=save_dir,
        Psi=Psi2,
        metric_logs=metric_logs2,
        chi=chi,
        num_players=num_players,
        seed=None,
    )

    metadata2 = get_metadata(filepath2)
    assert metadata2['seed'] is None, f"Seed should be None, got: {metadata2['seed']}"
    print(f"  ✓ None seed correctly saved in metadata")

    # Cleanup
    os.remove(filepath2)

    # Test 3: Verify different seeds produce different states
    print("\n3. Test different seeds produce different states...")
    seed1 = 111
    seed2 = 222

    Psi_s1 = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=seed1, dtype=np.float64)
    Psi_s2 = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=seed2, dtype=np.float64)

    # Compare first tensor
    are_different = not np.allclose(Psi_s1[0], Psi_s2[0])
    assert are_different, "Different seeds should produce different states"
    print(f"  ✓ Seeds {seed1} and {seed2} produce different states")

    # Test 4: Verify same seed produces reproducible results
    print("\n4. Test same seed produces reproducible results...")
    Psi_s1_again = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=seed1, dtype=np.float64)

    are_same = np.allclose(Psi_s1[0], Psi_s1_again[0])
    assert are_same, "Same seed should produce same state"
    print(f"  ✓ Seed {seed1} produces reproducible states")

    # Cleanup test directory
    if os.path.exists(save_dir) and not os.listdir(save_dir):
        os.rmdir(save_dir)

    print("\n" + "="*60)
    print("✅ All seed tracking tests passed!")
    print("="*60)

if __name__ == '__main__':
    try:
        test_seed_tracking()
    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
