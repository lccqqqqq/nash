"""
Test the generic cyclic player Hamiltonian implementation.
Verifies that get_default_H works for arbitrary numbers of players.
"""

import numpy as np
from game import get_default_H, get_default_4players, get_default_cyclic_players

np.set_printoptions(precision=4, suppress=True)

def test_player_count(L):
    """Test Hamiltonian generation for L players."""
    print(f"\n{'='*70}")
    print(f"Testing {L} players")
    print(f"{'='*70}")

    # Get Hamiltonian
    H = get_default_H(num_players=L, option='H')

    # Check structure
    print(f"Number of Hamiltonians: {len(H)}")
    print(f"Expected: {L}")
    assert len(H) == L, f"Expected {L} Hamiltonians, got {len(H)}"

    # Check shapes
    expected_shape = tuple([2] * (2 * L))
    for i, h in enumerate(H):
        print(f"  Player {i} Hamiltonian shape: {h.shape}")
        assert h.shape == expected_shape, f"Expected shape {expected_shape}, got {h.shape}"

    # Check all are numpy arrays with correct dtype
    assert all(isinstance(h, np.ndarray) for h in H), "All Hamiltonians should be numpy arrays"
    assert all(h.dtype == np.float32 for h in H), "All Hamiltonians should be float32"

    # Check ring topology property: each player should have symmetric interactions
    # For a ring, H[i] should be similar in structure to H[(i+1)%L] under rotation

    # Verify non-zero (each player actually has payoffs)
    for i, h in enumerate(H):
        assert np.any(h != 0), f"Player {i} Hamiltonian is all zeros"

    print(f"✓ All checks passed for {L} players")

    return H


def test_backward_compatibility():
    """Test that get_default_4players produces same result as before."""
    print(f"\n{'='*70}")
    print(f"Testing backward compatibility for 4 players")
    print(f"{'='*70}")

    # Old API
    H_old = get_default_4players(option='H')

    # New API via dispatcher
    H_new = get_default_H(num_players=4, option='H')

    # New API direct
    H_direct = get_default_cyclic_players(L=4, option='H')

    # All should be identical
    for i in range(4):
        assert np.allclose(H_old[i], H_new[i]), f"Mismatch at player {i} between old and new API"
        assert np.allclose(H_old[i], H_direct[i]), f"Mismatch at player {i} between old and direct API"

    print("✓ get_default_4players() produces same result as get_default_H(num_players=4)")
    print("✓ get_default_cyclic_players(L=4) produces same result")

    return H_old


def test_error_handling():
    """Test that invalid inputs raise appropriate errors."""
    print(f"\n{'='*70}")
    print(f"Testing error handling")
    print(f"{'='*70}")

    # Test L < 2
    try:
        H = get_default_H(num_players=1)
        assert False, "Should have raised ValueError for num_players=1"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for num_players=1: {e}")

    try:
        H = get_default_cyclic_players(L=0)
        assert False, "Should have raised ValueError for L=0"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for L=0: {e}")

    # Test invalid option
    try:
        H = get_default_H(num_players=4, option='invalid')
        assert False, "Should have raised ValueError for invalid option"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid option: {e}")


def test_stacked_format():
    """Test the 'H_all_in_one' option."""
    print(f"\n{'='*70}")
    print(f"Testing stacked format")
    print(f"{'='*70}")

    L = 5

    # Get both formats
    H_list = get_default_H(num_players=L, option='H')
    H_stacked = get_default_H(num_players=L, option='H_all_in_one')

    # Check stacked shape
    expected_shape = (L,) + tuple([2] * (2 * L))
    print(f"Stacked shape: {H_stacked.shape}")
    print(f"Expected: {expected_shape}")
    assert H_stacked.shape == expected_shape, f"Expected shape {expected_shape}, got {H_stacked.shape}"

    # Check equivalence
    for i in range(L):
        assert np.allclose(H_list[i], H_stacked[i]), f"Mismatch at player {i} between list and stacked"

    print(f"✓ Stacked format works correctly")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING GENERIC CYCLIC PLAYER HAMILTONIAN")
    print("="*70)

    # Test various player counts
    for L in [2, 3, 4, 5, 6, 7, 10]:
        test_player_count(L)

    # Test backward compatibility
    test_backward_compatibility()

    # Test error handling
    test_error_handling()

    # Test stacked format
    test_stacked_format()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)


if __name__ == "__main__":
    main()
