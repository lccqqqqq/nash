"""
Crisp definition of the game. Defined by the payoff matrices, should work for generic number of players.
"""

import numpy as np
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

def get_hamiltonian(L: int, H: Complex[np.ndarray, "player cl cr"], edge: tuple[int, int], site: int, dtype: np.dtype = np.complex64) -> Complex[np.ndarray, "player Cl Cr"]:
    """
    Get the Hamiltonian for a given set of sites.
    """
    idx = edge.index(site)
    H_mpo = np.eye(2**L, dtype=dtype).reshape([2] * (2*L))
    H_mpo = np.tensordot(H_mpo, H[idx].reshape(2, 2, 2, 2), axes=([L+edge[0], L+edge[1]], [0, 1]))
    H_mpo = np.moveaxis(H_mpo, [-2, -1], [L+edge[0], L+edge[1]])
    return H_mpo

def get_hloc_from_graph(G: nx.DiGraph, H: Complex[np.ndarray, "player cl cr"], site: int, dtype: np.dtype = np.complex64, normalize: bool = True) -> Complex[np.ndarray, "player Cl Cr"]:
    print(G.number_of_nodes())
    hloc = np.zeros([2] * (2 * G.number_of_nodes()), dtype=dtype)
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

def get_default_3players(option: str = 'H', dtype: np.dtype = np.float32):
    """
    Returns the default Hamiltonian for the 3-player quantum game (Prisoner's Dilemma variant).

    Args:
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        dtype: Data type for arrays (default: np.float32)

    Returns:
        List[ndarray] or ndarray: Hamiltonian(s) representing payoff matrices
            - If 'H': List of 3 arrays, each shape (2,2,2,2,2,2)
            - If 'H_all_in_one': Single stacked array, shape (3,2,2,2,2,2,2)

    Implementation:
        Creates three diagonal payoff matrices (one per player) encoding a quantum
        Prisoner's Dilemma game. Each matrix represents outcomes for 2^3 = 8
        possible measurement results.
    """
    H = [np.diag(np.array([6., 3., 3., 0., 10., 6., 6., 2.], dtype=dtype)),
        np.diag(np.array([6., 3., 10., 6., 3., 0., 6., 2.], dtype=dtype)),
        np.diag(np.array([6., 10., 3., 6., 3., 6., 0., 2.], dtype=dtype))]
    H = [h.reshape(2, 2, 2, 2, 2, 2) for h in H]

    if option == 'H':
        return H
    elif option == 'H_all_in_one':
        H_all_in_one = np.stack(H)
        return H_all_in_one
    else:
        raise ValueError("Invalid option")

def get_default_2players(option: str = 'H', dtype: np.dtype = np.float32):
    """
    Returns the default Hamiltonian for the 2-player quantum game (Prisoner's Dilemma variant).
    """
    H = [np.diag(np.array([3., 0., 5., 1.], dtype=dtype)),
        np.diag(np.array([3., 5., 0., 1.], dtype=dtype))]
    H = [h.reshape(2, 2, 2, 2) for h in H]

    if option == 'H':
        return H
    elif option == 'H_all_in_one':
        H_all_in_one = np.stack(H)
        return H_all_in_one
    else:
        raise ValueError("Invalid option")

def get_default_4players(option: str = 'H', dtype: np.dtype = np.float32):
    """
    Returns the default Hamiltonian for the 4-player quantum game on a ring graph.

    Args:
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        dtype: Data type for arrays (default: np.float32)

    Returns:
        List[ndarray] or ndarray: Hamiltonian(s) representing payoff matrices
            - If 'H': List of 4 arrays, each shape (2,2,2,2,2,2,2,2)
            - If 'H_all_in_one': Single stacked array, shape (4,2,2,2,2,2,2,2,2)

    Implementation:
        Uses graph-based construction from more_players.ipynb. Players are arranged
        on a ring (0-1-2-3-0), each interacting with 2 neighbors. Pairwise interactions
        use the same payoff structure as 2-player Prisoner's Dilemma.
    """
    # Pairwise payoff from more_players.ipynb (same as 2-player game)
    H_pairwise = np.stack([
        np.diag(np.array([3., 0., 5., 1.], dtype=dtype)),   # Player 1's payoff in 2-player interaction
        np.diag(np.array([3., 5., 0., 1.], dtype=dtype))    # Player 2's payoff in 2-player interaction
    ])

    # Ring graph topology: 0-1-2-3-0
    L = 4
    H = []

    for site in range(L):
        # Each site has 2 neighbors (left and right on ring)
        neighbors = [(site - 1) % L, (site + 1) % L]

        # Start with zero operator
        hloc = np.zeros([2] * (2 * L), dtype=dtype)

        for neighbor in neighbors:
            edge = tuple(sorted([site, neighbor]))
            idx_in_edge = 0 if edge[0] == site else 1

            # Get pairwise Hamiltonian and embed into full L-qubit space
            H_edge = np.eye(2**L, dtype=dtype).reshape([2] * (2 * L))
            H_2site = H_pairwise[idx_in_edge].reshape(2, 2, 2, 2)

            # Contract: place 2-site interaction at positions (edge[0], edge[1])
            # H_edge has indices [phys_0, phys_1, ..., phys_{L-1}, aux_0, aux_1, ..., aux_{L-1}]
            # We contract auxiliary indices at edge positions with H_2site's physical indices
            H_edge = np.tensordot(H_edge, H_2site,
                                 axes=([L + edge[0], L + edge[1]], [0, 1]))
            # Move the new auxiliary indices back to their positions
            H_edge = np.moveaxis(H_edge, [-2, -1], [L + edge[0], L + edge[1]])

            hloc = hloc + H_edge

        # Normalize by degree (each player has 2 neighbors)
        hloc = hloc / 2.0
        H.append(hloc)

    if option == 'H':
        return H
    elif option == 'H_all_in_one':
        H_all_in_one = np.stack(H)
        return H_all_in_one
    else:
        raise ValueError(f"Invalid option: {option}")

def get_default_H(num_players: int = 3, option: str = 'H', dtype: np.dtype = np.float32):
    if num_players == 3:
        return get_default_3players(option=option, dtype=dtype)
    elif num_players == 2:
        return get_default_2players(option=option, dtype=dtype)
    elif num_players == 4:
        return get_default_4players(option=option, dtype=dtype)
    else:
        raise ValueError(f"Invalid number of players: {num_players}. Supported: 2, 3, 4")