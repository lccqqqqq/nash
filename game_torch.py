"""
Crisp definition of the game. Defined by the payoff matrices, should work for generic number of players.
"""

import torch as t
import einops
import networkx as nx
from jaxtyping import Complex

def canonical_qpd(L: int = 3):
    """
    Get the canonical game graph for prisoner's dilemma.
    """
    game = nx.DiGraph()
    game.add_edges_from([(i, i+1) for i in range(L-1)] + [(L-1, 0)])
    return game

def get_hamiltonian(L: int, H: Complex[t.Tensor, "player cl cr"], edge: tuple[int, int], site: int, dtype: t.dtype = t.complex64) -> Complex[t.Tensor, "player Cl Cr"]:
    """
    Get the Hamiltonian for a given set of sites.
    """
    idx = edge.index(site)
    device = H.device
    H_mpo = t.eye(2**L, dtype=dtype, device=device).reshape([2] * (2*L))
    H_mpo = t.tensordot(H_mpo, H[idx].reshape(2, 2, 2, 2), dims=([L+edge[0], L+edge[1]], [0, 1]))
    H_mpo = t.moveaxis(H_mpo, [-2, -1], [L+edge[0], L+edge[1]])
    return H_mpo

def get_hloc_from_graph(G: nx.DiGraph, H: Complex[t.Tensor, "player cl cr"], site: int, dtype: t.dtype = t.complex64, normalize: bool = True) -> Complex[t.Tensor, "player Cl Cr"]:
    print(G.number_of_nodes())
    device = H.device
    hloc = t.zeros([2] * (2 * G.number_of_nodes()), dtype=dtype, device=device)
    for edge in G.in_edges(site):
        print(edge)
        H_mpo = get_hamiltonian(L=G.number_of_nodes(), H=H, edge=edge, site=site, dtype=dtype)
        hloc = hloc + H_mpo
    for edge in G.out_edges(site):
        print(edge)
        H_mpo = get_hamiltonian(L=G.number_of_nodes(), H=H, edge=edge, site=site, dtype=dtype)
        hloc = hloc + H_mpo

    if normalize:
        hloc = hloc / (len(list(G.in_edges(site))) + len(list(G.out_edges(site))))
    return hloc


# -------------------------------------------------------------------------------------------------
# Particular game definitions for quantum prisoner's dilemma

def get_default_3players(option: str = 'H', default_dtype: t.dtype = t.float32, device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))):
    """
    Returns the default Hamiltonian for the 3-player quantum game (Prisoner's Dilemma variant).

    Args:
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        default_dtype: Data type for tensors (default: float64)
        device: Computation device (CUDA if available, else CPU)

    Returns:
        List[Tensor] or Tensor: Hamiltonian(s) representing payoff matrices
            - If 'H': List of 3 tensors, each shape (2,2,2,2,2,2)
            - If 'H_all_in_one': Single stacked tensor, shape (3,2,2,2,2,2,2)

    Implementation:
        Creates three diagonal payoff matrices (one per player) encoding a quantum
        Prisoner's Dilemma game. Each matrix represents outcomes for 2^3 = 8
        possible measurement results.
    """
    H = [t.diag(t.tensor([6., 3., 3., 0., 10., 6., 6., 2.], dtype=default_dtype, device=device)),
        t.diag(t.tensor([6., 3., 10., 6., 3., 0., 6., 2.], dtype=default_dtype, device=device)),
        t.diag(t.tensor([6., 10., 3., 6., 3., 6., 0., 2.], dtype=default_dtype, device=device))]
    H = [h.reshape(2, 2, 2, 2, 2, 2) for h in H]

    if option == 'H':
        return H
    elif option == 'H_all_in_one':
        H_all_in_one = t.stack(H)
        return H_all_in_one
    else:
        raise ValueError("Invalid option")

def get_default_2players(option: str = 'H', default_dtype: t.dtype = t.float32, device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))):
    """
    Returns the default Hamiltonian for the 2-player quantum game (Prisoner's Dilemma variant).
    """
    H = [t.diag(t.tensor([3., 0., 5., 1.], dtype=default_dtype, device=device)),
        t.diag(t.tensor([3., 5., 0., 1.], dtype=default_dtype, device=device))]
    H = [h.reshape(2, 2, 2, 2) for h in H]

    if option == 'H':
        return H
    elif option == 'H_all_in_one':
        H_all_in_one = t.stack(H)
        return H_all_in_one
    else:
        raise ValueError("Invalid option")