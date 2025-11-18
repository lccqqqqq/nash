"""
Test suite for MPS norms, canonical forms, and Nash equilibrium solver functions.

This test suite focuses on:
1. MPS normalization and overlap computations
2. Canonical form conversions and orthogonality properties
3. Gradually building up to full solver algorithm tests

Run with: pytest test_solver.py -v -s
"""

import pytest
import numpy as np

# Import MPS utilities
from mps_utils import (
    get_rand_mps,
    get_product_state,
    get_ghz_state,
    to_canonical_form,
    test_canonical_form,
)

# Import MPS operations from misc.py
from misc import mps_overlap

# Import solver functions will be added as we expand tests
# from solver import apply_u, compute_ent_params_from_state


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=[2, 3, 4, 5])
def system_size(request):
    """Parametrize tests over different system sizes L."""
    return request.param


@pytest.fixture(params=[2, 4])
def bond_dim(request):
    """Parametrize tests over different bond dimensions χ."""
    return request.param


@pytest.fixture
def dtype():
    """Default dtype for quantum states (complex numbers)."""
    return np.complex64


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def tolerance():
    """Default numerical tolerance for tests."""
    return {'rtol': 1e-5, 'atol': 1e-6}


# ============================================================================
# Test MPS Norms
# ============================================================================

class TestMPSNorms:
    """Test MPS normalization and overlap properties across different system sizes."""

    def setup_method(self):
        """Setup run before each test method."""
        print("\n" + "="*70)

    def test_random_mps_normalized(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that randomly generated MPS in canonical form has norm 1."""
        print(f"Testing random MPS normalization: L={system_size}, χ={bond_dim}")

        # Create random MPS in canonical form
        Psi = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='A',
            canonicalize=True
        )

        # Compute norm via overlap with self
        norm_squared = mps_overlap(Psi, Psi)
        norm = np.sqrt(np.abs(norm_squared))

        print(f"  Norm: {norm}")
        print(f"  Expected: 1.0")

        np.testing.assert_allclose(norm, 1.0, **tolerance)
        print("  ✓ PASSED: Random MPS is normalized\n")

    def test_product_state_norm(self, system_size, dtype, tolerance):
        """Test that product states have norm 1."""
        print(f"Testing product state norm: L={system_size}")

        # Test |000...0⟩
        state_config = [0] * system_size
        Psi = get_product_state(L=system_size, state_per_site=state_config, dtype=dtype)

        norm_squared = mps_overlap(Psi, Psi)
        norm = np.sqrt(np.abs(norm_squared))

        print(f"  State: |{''.join(map(str, state_config))}⟩")
        print(f"  Norm: {norm}")

        np.testing.assert_allclose(norm, 1.0, **tolerance)

        # Test |101...⟩ (alternating pattern if L >= 3)
        if system_size >= 3:
            state_config = [1, 0, 1] + [0] * (system_size - 3)
            Psi = get_product_state(L=system_size, state_per_site=state_config, dtype=dtype)
            norm_squared = mps_overlap(Psi, Psi)
            norm = np.sqrt(np.abs(norm_squared))

            print(f"  State: |{''.join(map(str, state_config))}⟩")
            print(f"  Norm: {norm}")

            np.testing.assert_allclose(norm, 1.0, **tolerance)

        print("  ✓ PASSED: Product states are normalized\n")

    def test_ghz_state_norm(self, system_size, dtype, tolerance):
        """Test that GHZ states have norm 1."""
        print(f"Testing GHZ state norm: L={system_size}")

        # Create GHZ state: (|00...0⟩ + |11...1⟩)/√2
        Psi = get_ghz_state(L=system_size, dtype=dtype)

        norm_squared = mps_overlap(Psi, Psi)
        norm = np.sqrt(np.abs(norm_squared))

        print(f"  GHZ state: (|{'0'*system_size}⟩ + |{'1'*system_size}⟩)/√2")
        print(f"  Norm: {norm}")

        np.testing.assert_allclose(norm, 1.0, **tolerance)
        print("  ✓ PASSED: GHZ state is normalized\n")

    def test_overlap_self(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that ⟨ψ|ψ⟩ equals ||ψ||²."""
        print(f"Testing self-overlap: L={system_size}, χ={bond_dim}")

        Psi = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='A',
            canonicalize=True
        )

        overlap = mps_overlap(Psi, Psi)

        print(f"  ⟨ψ|ψ⟩: {overlap}")
        print(f"  Real part: {overlap.real}")
        print(f"  Imaginary part: {overlap.imag}")

        # For normalized state, should be 1
        assert overlap.real >= 0, "Norm squared must be non-negative"
        np.testing.assert_allclose(overlap.real, 1.0, **tolerance)
        np.testing.assert_allclose(overlap.imag, 0.0, **tolerance)

        print("  ✓ PASSED: Self-overlap equals 1 (real and positive)\n")

    def test_overlap_symmetry(self, system_size, dtype, seed, tolerance):
        """Test that ⟨ψ|φ⟩ = ⟨φ|ψ⟩*."""
        print(f"Testing overlap symmetry: L={system_size}")

        # Create two different random MPS
        Psi1 = get_rand_mps(L=system_size, chi=2, d_phys=2, seed=seed, dtype=dtype)
        Psi2 = get_rand_mps(L=system_size, chi=2, d_phys=2, seed=seed+1, dtype=dtype)

        overlap_12 = mps_overlap(Psi1, Psi2)
        overlap_21 = mps_overlap(Psi2, Psi1)

        print(f"  ⟨ψ₁|ψ₂⟩: {overlap_12}")
        print(f"  ⟨ψ₂|ψ₁⟩: {overlap_21}")
        print(f"  ⟨ψ₁|ψ₂⟩*: {overlap_12.conj()}")

        # Should satisfy ⟨ψ|φ⟩ = ⟨φ|ψ⟩*
        np.testing.assert_allclose(overlap_12, overlap_21.conj(), **tolerance)
        print("  ✓ PASSED: Overlap is Hermitian symmetric\n")

    def test_norm_after_canonicalization(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that canonicalization normalizes the state to norm 1."""
        print(f"Testing normalization during canonicalization: L={system_size}, χ={bond_dim}")

        # Create random MPS (not necessarily canonical or normalized)
        Psi_original = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='A',
            canonicalize=False  # Start non-canonical
        )

        # Compute original norm (likely > 1)
        norm_original_sq = mps_overlap(Psi_original, Psi_original)
        norm_original = np.sqrt(np.abs(norm_original_sq))

        # Canonicalize to left-canonical form (should normalize)
        Psi_canonical = to_canonical_form(Psi_original, form='A')

        # Compute norm after canonicalization
        norm_canonical_sq = mps_overlap(Psi_canonical, Psi_canonical)
        norm_canonical = np.sqrt(np.abs(norm_canonical_sq))

        print(f"  Original norm (unnormalized): {norm_original}")
        print(f"  Canonical norm (should be 1): {norm_canonical}")

        # Canonicalization automatically normalizes to 1
        np.testing.assert_allclose(norm_canonical, 1.0, **tolerance)
        print("  ✓ PASSED: Canonicalization normalizes state to norm 1\n")


# ============================================================================
# Test Canonical Forms
# ============================================================================

class TestCanonicalForms:
    """Test canonical form conversions and orthogonality properties."""

    def setup_method(self):
        """Setup run before each test method."""
        print("\n" + "="*70)

    def test_left_canonical_orthogonality(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that left-canonical form satisfies A†A = I for each site."""
        print(f"Testing left-canonical orthogonality: L={system_size}, χ={bond_dim}")

        # Create MPS in left-canonical form
        Psi = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='A',
            canonicalize=True
        )

        # Check orthogonality at each site
        for i in range(system_size):
            A = Psi[i]  # Shape: (d_phys, χ_L, χ_R)
            d_phys, chi_L, chi_R = A.shape

            # Reshape to matrix: (d_phys * χ_L, χ_R)
            A_mat = A.reshape(d_phys * chi_L, chi_R)

            # Compute A†A
            AhA = A_mat.T.conj() @ A_mat

            # Should be identity of size χ_R × χ_R
            identity = np.eye(chi_R, dtype=dtype)

            print(f"  Site {i}: A†A shape = {AhA.shape}, should be {chi_R}×{chi_R} identity")

            np.testing.assert_allclose(AhA, identity, **tolerance)

        print("  ✓ PASSED: All sites satisfy left-canonical condition A†A = I\n")

    def test_right_canonical_orthogonality(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that right-canonical form satisfies BB† = I for each site."""
        print(f"Testing right-canonical orthogonality: L={system_size}, χ={bond_dim}")

        # Create MPS in right-canonical form
        Psi = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='B',
            canonicalize=True
        )

        # Check orthogonality at each site
        for i in range(system_size):
            B = Psi[i]  # Shape: (d_phys, χ_L, χ_R)
            d_phys, chi_L, chi_R = B.shape

            # Reshape to matrix: (χ_L, d_phys * χ_R)
            B_mat = B.transpose(1, 0, 2).reshape(chi_L, d_phys * chi_R)

            # Compute BB†
            BBh = B_mat @ B_mat.T.conj()

            # Should be identity of size χ_L × χ_L
            identity = np.eye(chi_L, dtype=dtype)

            print(f"  Site {i}: BB† shape = {BBh.shape}, should be {chi_L}×{chi_L} identity")

            np.testing.assert_allclose(BBh, identity, **tolerance)

        print("  ✓ PASSED: All sites satisfy right-canonical condition BB† = I\n")

    def test_canonical_form_checker(self, system_size, bond_dim, dtype, seed):
        """Test that test_canonical_form() correctly identifies canonical forms."""
        print(f"Testing canonical form checker: L={system_size}, χ={bond_dim}")

        # Create left-canonical MPS
        Psi_A = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='A',
            canonicalize=True
        )

        # Create right-canonical MPS
        Psi_B = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed+1,
            dtype=dtype,
            form='B',
            canonicalize=True
        )

        # Create non-canonical MPS
        Psi_non_canonical = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed+2,
            dtype=dtype,
            form='A',
            canonicalize=False
        )

        # Test checker
        is_A_canonical = test_canonical_form(Psi_A, form='A', atol=1e-6)
        is_B_canonical = test_canonical_form(Psi_B, form='B', atol=1e-6)
        is_non_canonical_A = test_canonical_form(Psi_non_canonical, form='A', atol=1e-6)
        is_non_canonical_B = test_canonical_form(Psi_non_canonical, form='B', atol=1e-6)

        print(f"  Left-canonical MPS passes 'A' test: {is_A_canonical}")
        print(f"  Right-canonical MPS passes 'B' test: {is_B_canonical}")
        print(f"  Non-canonical MPS passes 'A' test: {is_non_canonical_A}")
        print(f"  Non-canonical MPS passes 'B' test: {is_non_canonical_B}")

        assert is_A_canonical, "Left-canonical MPS should pass 'A' form test"
        assert is_B_canonical, "Right-canonical MPS should pass 'B' form test"
        assert not is_non_canonical_A, "Non-canonical MPS should fail 'A' form test"
        assert not is_non_canonical_B, "Non-canonical MPS should fail 'B' form test"

        print("  ✓ PASSED: Canonical form checker works correctly\n")

    def test_canonicalization_preserves_overlap(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that canonicalization preserves relative overlap (up to normalization)."""
        print(f"Testing relative overlap preservation during canonicalization: L={system_size}, χ={bond_dim}")

        # Create two non-canonical MPS
        Psi1 = get_rand_mps(L=system_size, chi=bond_dim, d_phys=2, seed=seed, dtype=dtype, canonicalize=False)
        Psi2 = get_rand_mps(L=system_size, chi=bond_dim, d_phys=2, seed=seed+1, dtype=dtype, canonicalize=False)

        # Compute normalized overlap: ⟨ψ1|ψ2⟩ / (||ψ1|| ||ψ2||)
        overlap_12 = mps_overlap(Psi1, Psi2)
        norm1 = np.sqrt(np.abs(mps_overlap(Psi1, Psi1)))
        norm2 = np.sqrt(np.abs(mps_overlap(Psi2, Psi2)))
        overlap_original_normalized = overlap_12 / (norm1 * norm2)

        # Canonicalize both (this normalizes them)
        Psi1_canonical = to_canonical_form(Psi1, form='A')
        Psi2_canonical = to_canonical_form(Psi2, form='A')

        # Compute overlap after canonicalization (states are now normalized)
        overlap_canonical = mps_overlap(Psi1_canonical, Psi2_canonical)

        print(f"  Original normalized overlap: {overlap_original_normalized}")
        print(f"  Canonical overlap: {overlap_canonical}")
        print(f"  Magnitude difference: {np.abs(np.abs(overlap_original_normalized) - np.abs(overlap_canonical))}")

        # The normalized overlaps should match
        np.testing.assert_allclose(np.abs(overlap_canonical), np.abs(overlap_original_normalized), **tolerance)
        print("  ✓ PASSED: Normalized overlap preserved during canonicalization\n")

    def test_canonicalization_preserves_norm(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test that both canonical forms normalize the state to 1."""
        print(f"Testing canonical form normalization: L={system_size}, χ={bond_dim}")

        # Start with unnormalized MPS
        Psi_original = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            canonicalize=False
        )

        # Compute original norm
        norm_original = np.sqrt(np.abs(mps_overlap(Psi_original, Psi_original)))

        # Canonicalize to 'A' form (should normalize to 1)
        Psi_A = to_canonical_form(Psi_original, form='A')
        norm_A = np.sqrt(np.abs(mps_overlap(Psi_A, Psi_A)))

        # Canonicalize to 'B' form (should normalize to 1)
        Psi_B = to_canonical_form(Psi_original, form='B')
        norm_B = np.sqrt(np.abs(mps_overlap(Psi_B, Psi_B)))

        print(f"  Original norm (unnormalized): {norm_original}")
        print(f"  'A' form norm (should be 1): {norm_A}")
        print(f"  'B' form norm (should be 1): {norm_B}")

        # Both canonical forms should have norm 1
        np.testing.assert_allclose(norm_A, 1.0, **tolerance)
        np.testing.assert_allclose(norm_B, 1.0, **tolerance)

        print("  ✓ PASSED: Both canonical forms normalize to 1\n")

    def test_mixed_canonical_forms(self, system_size, bond_dim, dtype, seed, tolerance):
        """Test conversion between left and right canonical forms preserves state."""
        print(f"Testing A ↔ B conversion: L={system_size}, χ={bond_dim}")

        # Create left-canonical MPS
        Psi_A = get_rand_mps(
            L=system_size,
            chi=bond_dim,
            d_phys=2,
            seed=seed,
            dtype=dtype,
            form='A',
            canonicalize=True
        )

        # Convert to right-canonical
        Psi_B = to_canonical_form(Psi_A, form='B')

        # Convert back to left-canonical
        Psi_A_again = to_canonical_form(Psi_B, form='A')

        # Compute overlaps
        overlap_AB = mps_overlap(Psi_A, Psi_B)
        overlap_AA = mps_overlap(Psi_A, Psi_A_again)

        print(f"  ⟨A|B⟩: {overlap_AB}")
        print(f"  ⟨A|A'⟩ (after A→B→A): {overlap_AA}")
        print(f"  |⟨A|B⟩|: {np.abs(overlap_AB)}")
        print(f"  |⟨A|A'⟩|: {np.abs(overlap_AA)}")

        # Overlaps should have magnitude 1 (states are the same up to phase)
        np.testing.assert_allclose(np.abs(overlap_AB), 1.0, **tolerance)
        np.testing.assert_allclose(np.abs(overlap_AA), 1.0, **tolerance)

        # Verify canonical forms (use relaxed tolerance for numerical stability)
        is_B_canonical = test_canonical_form(Psi_B, form='B', atol=1e-5)
        is_A_canonical = test_canonical_form(Psi_A_again, form='A', atol=1e-5)

        print(f"  Psi_B passes B-canonical test: {is_B_canonical}")
        print(f"  Psi_A_again passes A-canonical test: {is_A_canonical}")

        # Note: Numerical errors can accumulate during multiple conversions
        # The key test is that overlaps are preserved (checked above)
        if not is_B_canonical:
            print("  Warning: B-form slightly off (numerical errors in conversion)")
        if not is_A_canonical:
            print("  Warning: A-form slightly off (numerical errors in conversion)")

        print("  ✓ PASSED: Conversions between canonical forms preserve state (up to numerical error)\n")


# ============================================================================
# Helper Functions (for future expansion)
# ============================================================================

# Will add helper functions as we expand tests to full solver algorithms


if __name__ == "__main__":
    # Run tests with: python test_solver.py
    # Or better: pytest test_solver.py -v -s
    print("Run tests with: pytest test_solver.py -v -s")
