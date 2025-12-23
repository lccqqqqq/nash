"""
Consistency tests to ensure old and new functionalities behave consistently as expected.
"""

import os
# Enable MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pytest
import torch as t
from mps_utils import to_comp_basis, get_rand_mps, get_product_state, get_ghz_state
from opt_mps_fiducial_state import get_state_from_tensors


@pytest.fixture
def default_device():
    """Get default device for tests"""
    if t.cuda.is_available():
        return t.device('cuda')
    elif t.backends.mps.is_available():
        return t.device('mps')
    else:
        return t.device('cpu')


@pytest.fixture
def default_dtype():
    """Get default dtype for tests"""
    return t.float32


class TestMPSStateConversion:
    """Tests for consistency between different MPS to computational basis conversion functions, namely the functions `to_comp_basis()` and `get_state_from_tensors()`.
    """

    def test_random_mps_after_canonicalization(self, default_device, default_dtype):
        """
        Test that to_comp_basis and get_state_from_tensors produce the same result
        for a random MPS after canonicalization.

        Both functions should convert MPS tensors to the computational basis state vector,
        and should agree (up to normalization and phase) after proper canonicalization.
        """
        # Create random MPS with OBC (Open Boundary Conditions)
        L = 3
        chi = 4
        Psi = get_rand_mps(L=L, chi=chi, d_phys=2, seed=42,
                          default_dtype=default_dtype, device=default_device,
                          form='A', canonicalize=True)

        # Convert using both methods
        psi_comp_basis = to_comp_basis(Psi, bc='PBC')
        psi_from_tensors = get_state_from_tensors(Psi, bc='PBC')

        # Normalize both
        psi_comp_basis = psi_comp_basis / t.linalg.norm(psi_comp_basis)
        psi_from_tensors = psi_from_tensors.flatten() / t.linalg.norm(psi_from_tensors)

        # Check if they're equal up to a global phase
        # Two states are equal up to phase if |⟨ψ₁|ψ₂⟩| = 1
        inner_product = t.abs(t.vdot(psi_comp_basis, psi_from_tensors))

        assert t.isclose(inner_product, t.tensor(1.0, device=default_device), atol=1e-5), \
            f"States differ: |⟨ψ₁|ψ₂⟩| = {inner_product.item()}"

    def test_product_state_consistency(self, default_device, default_dtype):
        """
        Test that both conversion methods agree on product states (which are simpler).
        """
        # Create product state |101⟩
        Psi = get_product_state(L=3, state_per_site=[1, 0, 1],
                               default_dtype=default_dtype, device=default_device)

        # Convert using both methods
        psi_comp_basis = to_comp_basis(Psi, bc='PBC')
        psi_from_tensors = get_state_from_tensors(Psi, bc='PBC')

        # Normalize both
        psi_comp_basis = psi_comp_basis / t.linalg.norm(psi_comp_basis)
        psi_from_tensors = psi_from_tensors.flatten() / t.linalg.norm(psi_from_tensors)

        # Check if they're equal up to a global phase
        inner_product = t.abs(t.vdot(psi_comp_basis, psi_from_tensors))

        assert t.isclose(inner_product, t.tensor(1.0, device=default_device), atol=1e-5), \
            f"Product states differ: |⟨ψ₁|ψ₂⟩| = {inner_product.item()}"

    def test_ghz_state_consistency(self, default_device, default_dtype):
        """
        Test that both conversion methods agree on GHZ states.
        GHZ state: |GHZ⟩ = (|000⟩ + |111⟩)/√2
        """
        # Create GHZ state
        Psi = get_ghz_state(L=3, default_dtype=default_dtype, device=default_device)

        # Convert using both methods
        psi_comp_basis = to_comp_basis(Psi, bc='PBC')
        psi_from_tensors = get_state_from_tensors(Psi, bc='PBC')

        # Normalize both
        psi_comp_basis = psi_comp_basis / t.linalg.norm(psi_comp_basis)
        psi_from_tensors = psi_from_tensors.flatten() / t.linalg.norm(psi_from_tensors)

        # Check if they're equal up to a global phase
        inner_product = t.abs(t.vdot(psi_comp_basis, psi_from_tensors))

        assert t.isclose(inner_product, t.tensor(1.0, device=default_device), atol=1e-5), \
            f"GHZ states differ: |⟨ψ₁|ψ₂⟩| = {inner_product.item()}"

    def test_multiple_random_seeds(self, default_device, default_dtype):
        """
        Test consistency across multiple random MPS states with different seeds.
        """
        seeds = [0, 10, 42, 123, 999]

        for seed in seeds:
            Psi = get_rand_mps(L=3, chi=3, d_phys=2, seed=seed,
                             default_dtype=default_dtype, device=default_device,
                             form='A', canonicalize=True)

            # Convert using both methods
            psi_comp_basis = to_comp_basis(Psi, bc='PBC')
            psi_from_tensors = get_state_from_tensors(Psi, bc='PBC')

            # Normalize both
            psi_comp_basis = psi_comp_basis / t.linalg.norm(psi_comp_basis)
            psi_from_tensors = psi_from_tensors.flatten() / t.linalg.norm(psi_from_tensors)

            # Check if they're equal up to a global phase
            inner_product = t.abs(t.vdot(psi_comp_basis, psi_from_tensors))

            assert t.isclose(inner_product, t.tensor(1.0, device=default_device), atol=1e-5), \
                f"States differ for seed {seed}: |⟨ψ₁|ψ₂⟩| = {inner_product.item()}"

    def test_different_bond_dimensions(self, default_device, default_dtype):
        """
        Test consistency for different bond dimensions.
        """
        bond_dims = [2, 3, 4, 5]

        for chi in bond_dims:
            Psi = get_rand_mps(L=3, chi=chi, d_phys=2, seed=42,
                             default_dtype=default_dtype, device=default_device,
                             form='B', canonicalize=True)

            # Convert using both methods
            psi_comp_basis = to_comp_basis(Psi, bc='PBC')
            psi_from_tensors = get_state_from_tensors(Psi, bc='PBC')

            # Normalize both
            psi_comp_basis = psi_comp_basis / t.linalg.norm(psi_comp_basis)
            psi_from_tensors = psi_from_tensors.flatten() / t.linalg.norm(psi_from_tensors)

            # Check if they're equal up to a global phase
            inner_product = t.abs(t.vdot(psi_comp_basis, psi_from_tensors))

            assert t.isclose(inner_product, t.tensor(1.0, device=default_device), atol=1e-5), \
                f"States differ for χ={chi}: |⟨ψ₁|ψ₂⟩| = {inner_product.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
