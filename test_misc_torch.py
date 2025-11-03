import numpy as np
import torch
import pytest

# Import both versions
import misc as misc_np
import misc_torch as misc_torch


def to_torch(x):
    """Convert numpy array to torch tensor"""
    if isinstance(x, list):
        return [to_torch(item) for item in x]
    return torch.from_numpy(x)


def to_numpy(x):
    """Convert torch tensor to numpy array"""
    if isinstance(x, list):
        return [to_numpy(item) for item in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class TestGroupLegs:
    def test_group_legs_simple(self):
        """Test basic group_legs operation"""
        print("\n" + "="*60)
        print("TEST: group_legs")
        print("="*60)

        np.random.seed(42)
        a_np = np.random.randn(2, 3, 4, 5)
        a_torch = to_torch(a_np)
        axes = [[0, 1], [2], [3]]

        print(f"Input shape: {a_np.shape}")
        print(f"Axes grouping: {axes}")

        result_np, pipe_np = misc_np.group_legs(a_np, axes)
        result_torch, pipe_torch = misc_torch.group_legs(a_torch, axes)

        print(f"NumPy output shape: {result_np.shape}")
        print(f"Torch output shape: {result_torch.shape}")
        max_diff = np.max(np.abs(result_np - to_numpy(result_torch)))
        print(f"Max difference: {max_diff:.2e}")

        np.testing.assert_allclose(result_np, to_numpy(result_torch), rtol=1e-6)
        print("✓ PASSED\n")

    def test_group_ungroup_roundtrip(self):
        """Test that ungroup_legs inverts group_legs"""
        print("\n" + "="*60)
        print("TEST: group_legs + ungroup_legs roundtrip")
        print("="*60)

        np.random.seed(42)
        a_np = np.random.randn(2, 3, 4, 5)
        a_torch = to_torch(a_np)
        axes = [[0, 2], [1, 3]]

        print(f"Input shape: {a_np.shape}")
        print(f"Axes grouping: {axes}")

        grouped_np, pipe_np = misc_np.group_legs(a_np, axes)
        ungrouped_np = misc_np.ungroup_legs(grouped_np, pipe_np)

        grouped_torch, pipe_torch = misc_torch.group_legs(a_torch, axes)
        ungrouped_torch = misc_torch.ungroup_legs(grouped_torch, pipe_torch)

        print(f"Grouped shape: {grouped_np.shape}")
        print(f"Ungrouped shape: {ungrouped_np.shape}")
        max_diff = np.max(np.abs(ungrouped_np - to_numpy(ungrouped_torch)))
        print(f"Max difference (NumPy vs Torch): {max_diff:.2e}")
        roundtrip_error = np.max(np.abs(a_np - ungrouped_np))
        print(f"Roundtrip error: {roundtrip_error:.2e}")

        np.testing.assert_allclose(ungrouped_np, to_numpy(ungrouped_torch), rtol=1e-6)
        np.testing.assert_allclose(a_np, ungrouped_np, rtol=1e-6)
        print("✓ PASSED\n")


class TestMPSOperations:
    def setup_method(self):
        """Create random MPS for testing"""
        np.random.seed(42)
        L = 5
        d = 2
        chi = 3

        # Create random MPS
        self.mps_np = []
        for i in range(L):
            chi_l = 1 if i == 0 else chi
            chi_r = 1 if i == L-1 else chi
            self.mps_np.append(np.random.randn(d, chi_l, chi_r))

        self.mps_torch = to_torch(self.mps_np)

    def test_transpose_mpo(self):
        """Test MPO transpose"""
        print("\n" + "="*60)
        print("TEST: transpose_mpo")
        print("="*60)

        np.random.seed(42)
        L = 4
        mpo_np = [np.random.randn(2, 2, 3, 3) for _ in range(L)]
        mpo_torch = to_torch(mpo_np)

        print(f"MPO length: {L}, tensor shape: (2, 2, 3, 3)")

        result_np = misc_np.transpose_mpo(mpo_np)
        result_torch = misc_torch.transpose_mpo(mpo_torch)

        max_diffs = []
        for i, (r_np, r_torch) in enumerate(result_np, result_torch):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs.append(diff)
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-6)

        print(f"Max difference across all sites: {max(max_diffs):.2e}")
        print("✓ PASSED\n")

    def test_mps_invert(self):
        """Test MPS inversion"""
        print("\n" + "="*60)
        print("TEST: mps_invert")
        print("="*60)

        print(f"MPS length: {len(self.mps_np)}")
        print(f"MPS tensor shapes: {[t.shape for t in self.mps_np]}")

        result_np = misc_np.mps_invert(self.mps_np)
        result_torch = misc_torch.mps_invert(self.mps_torch)

        max_diffs = []
        for i, (r_np, r_torch) in enumerate(zip(result_np, result_torch)):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs.append(diff)
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-6)

        print(f"Max difference across all sites: {max(max_diffs):.2e}")
        print("✓ PASSED\n")

    def test_mps_group_legs(self):
        """Test MPS leg grouping"""
        print("\n" + "="*60)
        print("TEST: mps_group_legs")
        print("="*60)

        result_np, pipes_np = misc_np.mps_group_legs(self.mps_np, axes='all')
        result_torch, pipes_torch = misc_torch.mps_group_legs(self.mps_torch, axes='all')

        max_diffs = []
        for r_np, r_torch in zip(result_np, result_torch):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs.append(diff)
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-6)

        print(f"Max difference across all sites: {max(max_diffs):.2e}")
        print("✓ PASSED\n")

    def test_mps_2form(self):
        """Test MPS canonical form"""
        print("\n" + "="*60)
        print("TEST: mps_2form (A and B canonical forms)")
        print("="*60)

        print("Testing A-canonical form...")
        result_np = misc_np.mps_2form(self.mps_np, form='A')
        result_torch = misc_torch.mps_2form(self.mps_torch, form='A')

        max_diffs_A = []
        for r_np, r_torch in zip(result_np, result_torch):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs_A.append(diff)
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-5)

        print(f"  A-form max difference: {max(max_diffs_A):.2e}")

        print("Testing B-canonical form...")
        result_np = misc_np.mps_2form(self.mps_np, form='B')
        result_torch = misc_torch.mps_2form(self.mps_torch, form='B')

        max_diffs_B = []
        for r_np, r_torch in zip(result_np, result_torch):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs_B.append(diff)
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-5)

        print(f"  B-form max difference: {max(max_diffs_B):.2e}")
        print("✓ PASSED\n")

    def test_mps_overlap(self):
        """Test MPS overlap computation"""
        print("\n" + "="*60)
        print("TEST: mps_overlap")
        print("="*60)

        np.random.seed(42)
        L = 5
        d = 2
        chi = 3

        print(f"Computing overlap of two MPS (L={L}, d={d}, chi={chi})")

        mps1_np = []
        mps2_np = []
        for i in range(L):
            chi_l = 1 if i == 0 else chi
            chi_r = 1 if i == L-1 else chi
            mps1_np.append(np.random.randn(d, chi_l, chi_r))
            mps2_np.append(np.random.randn(d, chi_l, chi_r))

        mps1_torch = to_torch(mps1_np)
        mps2_torch = to_torch(mps2_np)

        result_np = misc_np.mps_overlap(mps1_np, mps2_np)
        result_torch = misc_torch.mps_overlap(mps1_torch, mps2_torch)

        print(f"NumPy overlap: {result_np}")
        print(f"Torch overlap: {to_numpy(result_torch)}")
        print(f"Difference: {np.abs(result_np - to_numpy(result_torch)):.2e}")

        np.testing.assert_allclose(result_np, to_numpy(result_torch), rtol=1e-5)
        print("✓ PASSED\n")


class TestMPOOperations:
    def test_mpo_on_mps(self):
        """Test MPO application to MPS"""
        print("\n" + "="*60)
        print("TEST: mpo_on_mps")
        print("="*60)

        np.random.seed(42)
        L = 4
        d = 2
        chi = 3
        D = 2

        print(f"Applying MPO to MPS (L={L}, d={d}, chi={chi}, D={D})")

        # Create random MPS
        mps_np = []
        for i in range(L):
            chi_l = 1 if i == 0 else chi
            chi_r = 1 if i == L-1 else chi
            mps_np.append(np.random.randn(d, chi_l, chi_r))

        # Create random MPO
        mpo_np = []
        for i in range(L):
            D_l = 1 if i == 0 else D
            D_r = 1 if i == L-1 else D
            mpo_np.append(np.random.randn(D_l, D_r, d, d))

        mps_torch = to_torch(mps_np)
        mpo_torch = to_torch(mpo_np)

        result_np = misc_np.mpo_on_mps(mpo_np, mps_np)
        result_torch = misc_torch.mpo_on_mps(mpo_torch, mps_torch)

        max_diffs = []
        for i, (r_np, r_torch) in enumerate(zip(result_np, result_torch)):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs.append(diff)
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-5)

        print(f"Result MPS shapes: {[r.shape for r in result_np]}")
        print(f"Max difference across all sites: {max(max_diffs):.2e}")
        print("✓ PASSED\n")

    def test_mpo_on_mpo(self):
        """Test MPO multiplication"""
        np.random.seed(42)
        L = 3
        d = 2
        D = 2

        mpo1_np = []
        mpo2_np = []
        for i in range(L):
            D_l = 1 if i == 0 else D
            D_r = 1 if i == L-1 else D
            mpo1_np.append(np.random.randn(d, d, D_l, D_r))
            mpo2_np.append(np.random.randn(d, d, D_l, D_r))

        mpo1_torch = to_torch(mpo1_np)
        mpo2_torch = to_torch(mpo2_np)

        result_np = misc_np.mpo_on_mpo(mpo1_np, mpo2_np)
        result_torch = misc_torch.mpo_on_mpo(mpo1_torch, mpo2_torch)

        for r_np, r_torch in zip(result_np, result_torch):
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-5)

    def test_mpo_to_full(self):
        """Test MPO to full matrix conversion"""
        np.random.seed(42)
        L = 3
        d = 2
        D = 2

        mpo_np = []
        for i in range(L):
            D_l = 1 if i == 0 else D
            D_r = 1 if i == L-1 else D
            mpo_np.append(np.random.randn(D_l, D_r, d, d))

        mpo_torch = to_torch(mpo_np)

        result_np = misc_np.mpo_to_full(mpo_np)
        result_torch = misc_torch.mpo_to_full(mpo_torch)

        np.testing.assert_allclose(result_np, to_numpy(result_torch), rtol=1e-5)


class TestSVD:
    def test_svd_theta(self):
        """Test SVD with truncation"""
        print("\n" + "="*60)
        print("TEST: svd_theta")
        print("="*60)

        np.random.seed(42)
        theta_np = np.random.randn(10, 10)
        theta_torch = to_torch(theta_np)

        truncation_par = {'p_trunc': 1e-10, 'chi_max': 5}

        print(f"Matrix shape: {theta_np.shape}")
        print(f"Truncation parameters: {truncation_par}")

        A_np, SB_np, info_np = misc_np.svd_theta(theta_np, truncation_par)
        A_torch, SB_torch, info_torch = misc_torch.svd_theta(theta_torch, truncation_par)

        # Check dimensions match
        assert A_np.shape == to_numpy(A_torch).shape
        assert SB_np.shape == to_numpy(SB_torch).shape

        print(f"A shape: {A_np.shape}, SB shape: {SB_np.shape}")
        print(f"Kept {len(info_np['s'])} singular values")

        # Reconstruct and compare
        recon_np = A_np @ SB_np
        recon_torch = to_numpy(A_torch @ SB_torch)

        recon_diff = np.max(np.abs(recon_np - recon_torch))
        print(f"Reconstruction difference: {recon_diff:.2e}")

        np.testing.assert_allclose(recon_np, recon_torch, rtol=1e-5)

        # Check singular values match
        sv_diff = np.max(np.abs(info_np['s'] - to_numpy(info_torch['s'])))
        print(f"Singular values difference: {sv_diff:.2e}")
        np.testing.assert_allclose(info_np['s'], to_numpy(info_torch['s']), rtol=1e-5)
        print("✓ PASSED\n")

    def test_svd_theta_return_XYZ(self):
        """Test SVD with XYZ return format"""
        np.random.seed(42)
        theta_np = np.random.randn(10, 10)
        theta_torch = to_torch(theta_np)

        truncation_par = {'p_trunc': 1e-10, 'chi_max': 5}

        X_np, Y_np, Z_np, info_np = misc_np.svd_theta(theta_np, truncation_par, return_XYZ=True)
        X_torch, Y_torch, Z_torch, info_torch = misc_torch.svd_theta(theta_torch, truncation_par, return_XYZ=True)

        np.testing.assert_allclose(Y_np, to_numpy(Y_torch), rtol=1e-5)
        np.testing.assert_allclose(info_np['s'], to_numpy(info_torch['s']), rtol=1e-5)


class TestCompress:
    def test_compress(self):
        """Test MPS compression"""
        print("\n" + "="*60)
        print("TEST: compress")
        print("="*60)

        np.random.seed(42)
        L = 4
        d = 2
        chi = 8

        print(f"Compressing MPS (L={L}, d={d}, chi={chi} → chi_new=4)")

        # Create random MPS with large bond dimension
        mps_np = []
        for i in range(L):
            chi_l = 1 if i == 0 else chi
            chi_r = 1 if i == L-1 else chi
            mps_np.append(np.random.randn(d, chi_l, chi_r))

        mps_torch = to_torch(mps_np)

        # Compress to smaller bond dimension
        chi_new = 4
        result_np, info_np = misc_np.compress(mps_np, chi_new)
        result_torch, info_torch = misc_torch.compress(mps_torch, chi_new)

        print(f"Original shapes: {[m.shape for m in mps_np]}")
        print(f"Compressed shapes: {[r.shape for r in result_np]}")
        print(f"Truncation error (NumPy): {info_np['p_trunc']:.2e}")
        print(f"Truncation error (Torch): {info_torch['p_trunc']:.2e}")

        # Check dimensions match
        for r_np, r_torch in zip(result_np, result_torch):
            assert r_np.shape == to_numpy(r_torch).shape

        # Check truncation info
        assert abs(info_np['p_trunc'] - info_torch['p_trunc']) < 1e-5
        print("✓ PASSED\n")


class TestEntanglement:
    def test_mps_entanglement_spectrum(self):
        """Test entanglement spectrum calculation"""
        print("\n" + "="*60)
        print("TEST: mps_entanglement_spectrum")
        print("="*60)

        np.random.seed(42)
        L = 5
        d = 2
        chi = 4

        print(f"Computing entanglement spectrum for MPS (L={L}, d={d}, chi={chi})")

        mps_np = []
        for i in range(L):
            chi_l = 1 if i == 0 else chi
            chi_r = 1 if i == L-1 else chi
            mps_np.append(np.random.randn(d, chi_l, chi_r))

        mps_torch = to_torch(mps_np)

        result_np = misc_np.mps_entanglement_spectrum(mps_np)
        result_torch = misc_torch.mps_entanglement_spectrum(mps_torch)

        print(f"Number of bonds: {len(result_np)}")
        max_diffs = []
        for i, (r_np, r_torch) in enumerate(zip(result_np, result_torch)):
            diff = np.max(np.abs(r_np - to_numpy(r_torch)))
            max_diffs.append(diff)
            print(f"  Bond {i}: {len(r_np)} singular values, max diff = {diff:.2e}")
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-5)

        print("✓ PASSED\n")


class TestTransferMPO:
    def test_get_transfer_MPO(self):
        """Test transfer MPO construction"""
        np.random.seed(42)
        L = 3
        d = 2
        D = 2

        # Create MPO with 5 legs (first leg is physical, rest are bond dims)
        mpo_np = []
        for i in range(L):
            D_l = 1 if i == 0 else D
            D_r = 1 if i == L-1 else D
            mpo_np.append(np.random.randn(d, D_l, D_r, d, d))

        mpo_torch = to_torch(mpo_np)

        result_np = misc_np.get_transfer_MPO(mpo_np)
        result_torch = misc_torch.get_transfer_MPO(mpo_torch)

        for r_np, r_torch in zip(result_np, result_torch):
            np.testing.assert_allclose(r_np, to_numpy(r_torch), rtol=1e-5)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING MISC.PY vs MISC_TORCH.PY")
    print("="*60)
    print("Verifying that NumPy and PyTorch implementations")
    print("produce identical numerical results.")
    print("="*60 + "\n")

    exit_code = pytest.main([__file__, '-v', '-s'])

    if exit_code == 0:
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("NumPy and PyTorch implementations are numerically equivalent.")
        print("="*60 + "\n")
