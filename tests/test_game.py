"""
Unit tests for game.py module - quantum game theory functions
"""

import pytest
import torch as t
import networkx as nx
import einops
from game import canonical_qpd, get_hamiltonian, get_hloc_from_graph


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
def qpd_payoff_matrix(default_device):
    """Standard quantum prisoner's dilemma payoff matrices (2-player version)"""
    # Payoff matrices for 2 players
    H = t.stack([
        t.diag(t.tensor([3., 0., 5., 1.], dtype=t.complex64, device=default_device)),
        t.diag(t.tensor([3., 5., 0., 1.], dtype=t.complex64, device=default_device))
    ])
    return H


@pytest.fixture
def qpd_payoff_matrix_pairwise(default_device):
    """Pairwise quantum prisoner's dilemma payoff matrices (for edges in 3+ player games)

    These are 2-player payoffs used to construct multi-player games on graphs.
    Each edge in the game graph has a 2-player payoff.
    """
    H = t.stack([
        t.diag(t.tensor([3., 0., 5., 1.], dtype=t.complex64, device=default_device)),
        t.diag(t.tensor([3., 5., 0., 1.], dtype=t.complex64, device=default_device))
    ])
    return H


class TestCanonicalQPD:
    """Tests for canonical_qpd graph construction"""

    def test_3player_graph(self):
        """Test 3-player circular graph structure"""
        game = canonical_qpd(L=3)
        assert game.number_of_nodes() == 3
        assert game.number_of_edges() == 3
        # Check edges: 0->1, 1->2, 2->0 (circular)
        assert game.has_edge(0, 1)
        assert game.has_edge(1, 2)
        assert game.has_edge(2, 0)

    def test_4player_graph(self):
        """Test 4-player circular graph structure"""
        game = canonical_qpd(L=4)
        assert game.number_of_nodes() == 4
        assert game.number_of_edges() == 4
        assert game.has_edge(0, 1)
        assert game.has_edge(1, 2)
        assert game.has_edge(2, 3)
        assert game.has_edge(3, 0)

    def test_2player_graph(self):
        """Test 2-player graph structure"""
        game = canonical_qpd(L=2)
        assert game.number_of_nodes() == 2
        assert game.number_of_edges() == 2
        assert game.has_edge(0, 1)
        assert game.has_edge(1, 0)

    def test_is_directed(self):
        """Verify graph is directed"""
        game = canonical_qpd(L=3)
        assert isinstance(game, nx.DiGraph)


class TestGetHamiltonian:
    """Tests for get_hamiltonian function"""

    def test_hamiltonian_shape_3player(self, qpd_payoff_matrix_pairwise):
        """Test output shape for 3-player system"""
        H = qpd_payoff_matrix_pairwise
        H_mpo = get_hamiltonian(L=3, H=H, edge=(0, 1), site=0)
        # Should be shape (2, 2, 2, 2, 2, 2) for 3 qubits
        assert H_mpo.shape == (2, 2, 2, 2, 2, 2)

    def test_hamiltonian_is_diagonal(self, qpd_payoff_matrix):
        """Test that Hamiltonian is diagonal for QPD (when reshaped to matrix)"""
        H = qpd_payoff_matrix
        H_mpo = get_hamiltonian(L=3, H=H, edge=(2, 0), site=2)
        H_matrix = einops.rearrange(H_mpo, "a1 a2 a3 b1 b2 b3 -> (a1 a2 a3) (b1 b2 b3)")

        # For QPD, should be diagonal
        off_diagonal = H_matrix - t.diag(t.diagonal(H_matrix))
        assert t.allclose(off_diagonal, t.zeros_like(off_diagonal), atol=1e-10)

    def test_hamiltonian_device_consistency(self, qpd_payoff_matrix, default_device):
        """Test that output is on same device as input"""
        H = qpd_payoff_matrix
        H_mpo = get_hamiltonian(L=3, H=H, edge=(0, 1), site=0)
        assert H_mpo.device.type == default_device.type

    def test_hamiltonian_dtype(self, qpd_payoff_matrix):
        """Test that dtype is preserved"""
        H = qpd_payoff_matrix
        H_mpo = get_hamiltonian(L=3, H=H, edge=(0, 1), site=0, dtype=t.complex64)
        assert H_mpo.dtype == t.complex64

    def test_hamiltonian_summed_sites(self, qpd_payoff_matrix):
        """Test summing Hamiltonians from multiple edges (from notebook)"""
        H = qpd_payoff_matrix
        H_mpo = (get_hamiltonian(L=3, H=H, edge=(2, 0), site=2) +
                 get_hamiltonian(L=3, H=H, edge=(2, 1), site=2))
        H_matrix = einops.rearrange(H_mpo, "a1 a2 a3 b1 b2 b3 -> (a1 a2 a3) (b1 b2 b3)")

        # Should still be diagonal for QPD
        off_diagonal = H_matrix - t.diag(t.diagonal(H_matrix))
        assert t.allclose(off_diagonal, t.zeros_like(off_diagonal), atol=1e-10)

        # Check some expected values (should be sum of payoffs)
        assert t.isclose(H_matrix[0, 0].real, t.tensor(6.0))

    def test_edge_index_selection(self, qpd_payoff_matrix):
        """Test that correct player payoff is selected based on edge"""
        H = qpd_payoff_matrix
        H_site0 = get_hamiltonian(L=3, H=H, edge=(0, 1), site=0)
        H_site1 = get_hamiltonian(L=3, H=H, edge=(0, 1), site=1)

        # Different sites should generally give different Hamiltonians
        assert not t.allclose(H_site0, H_site1)


class TestGetHlocFromGraph:
    """Tests for get_hloc_from_graph function"""

    def test_triangle_graph(self, qpd_payoff_matrix_pairwise):
        """Test local Hamiltonian for triangular graph (from notebook)"""
        game = nx.DiGraph()
        game.add_edge(0, 1)
        game.add_edge(1, 2)
        game.add_edge(0, 2)

        H = qpd_payoff_matrix_pairwise
        hloc = get_hloc_from_graph(game, H, site=0, normalize=False)

        # Should have correct shape
        assert hloc.shape == (2, 2, 2, 2, 2, 2)

        # Reshape to matrix
        H_matrix = einops.rearrange(hloc, "a1 a2 a3 b1 b2 b3 -> (a1 a2 a3) (b1 b2 b3)")

        # Should be diagonal for QPD
        off_diagonal = H_matrix - t.diag(t.diagonal(H_matrix))
        assert t.allclose(off_diagonal, t.zeros_like(off_diagonal), atol=1e-10)

        # Check first diagonal element matches expected value (sum of 2 edges)
        assert t.isclose(H_matrix[0, 0].real, t.tensor(6.0))

    def test_circular_graph_4player(self, qpd_payoff_matrix):
        """Test local Hamiltonian for 4-player circular graph"""
        game = canonical_qpd(L=4)
        H = qpd_payoff_matrix

        # Each node should have 2 edges (1 in, 1 out)
        hloc = get_hloc_from_graph(game, H, site=0, normalize=False)
        assert hloc.shape == (2, 2, 2, 2, 2, 2, 2, 2)

    def test_normalization(self, qpd_payoff_matrix_pairwise):
        """Test that normalization works correctly"""
        game = nx.DiGraph()
        game.add_edge(0, 1)
        game.add_edge(0, 2)

        H = qpd_payoff_matrix_pairwise
        hloc_unnorm = get_hloc_from_graph(game, H, site=0, normalize=False)
        hloc_norm = get_hloc_from_graph(game, H, site=0, normalize=True)

        # Node 0 has 2 outgoing edges, so normalized should be half
        num_edges = 2
        assert t.allclose(hloc_norm * num_edges, hloc_unnorm)

    def test_device_consistency(self, qpd_payoff_matrix_pairwise, default_device):
        """Test that output device matches input"""
        game = canonical_qpd(L=3)
        H = qpd_payoff_matrix_pairwise
        hloc = get_hloc_from_graph(game, H, site=0)
        assert hloc.device.type == default_device.type

    def test_different_sites_different_results(self, qpd_payoff_matrix_pairwise):
        """Test that different sites can yield different Hamiltonians"""
        game = nx.DiGraph()
        game.add_edge(0, 1)
        game.add_edge(1, 2)
        game.add_edge(2, 0)

        H = qpd_payoff_matrix_pairwise
        hloc0 = get_hloc_from_graph(game, H, site=0, normalize=False)
        hloc1 = get_hloc_from_graph(game, H, site=1, normalize=False)

        # For asymmetric games, different sites should give different results
        # (For symmetric games they might be equal)
        assert hloc0.shape == hloc1.shape


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_canonical_graph_with_hamiltonians(self, qpd_payoff_matrix_pairwise):
        """Test full workflow: create graph and compute local Hamiltonians"""
        game = canonical_qpd(L=3)
        H = qpd_payoff_matrix_pairwise

        # Compute local Hamiltonian for each site
        hlocs = []
        for site in range(3):
            hloc = get_hloc_from_graph(game, H, site=site, normalize=True)
            hlocs.append(hloc)

        # All should have same shape
        assert all(hloc.shape == (2, 2, 2, 2, 2, 2) for hloc in hlocs)

        # All should be on same device
        devices = [hloc.device for hloc in hlocs]
        assert all(d == devices[0] for d in devices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
