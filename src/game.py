"""
Crisp definition of the game. Defined by the payoff matrices, should work for generic number of players.
"""

import numpy as np
import einops
import networkx as nx
from jaxtyping import Complex
from utils.misc import get_rand_normalized_herm_matrix

H_QPD = np.stack([
    np.diag(np.array([3., 0., 5., 1.])),   # Player 1's payoff in 2-player interaction
    np.diag(np.array([3., 5., 0., 1.]))    # Player 2's payoff in 2-player interaction
])


def get_perturbed_H_QPD(eps: float = 0.2, dtype: np.dtype = np.float32):
    return [h + eps * get_rand_normalized_herm_matrix(d=h.shape[-1], dtype=dtype) for h in H_QPD]

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

def get_default_cyclic_players(L: int, Hs: list[np.ndarray] = H_QPD, option: str = 'H', dtype: np.dtype = np.float32):
    """
    Returns the default Hamiltonian for L-player quantum game on a cyclic/ring graph.

    Args:
        L: Number of players (must be >= 2)
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        dtype: Data type for arrays (default: np.float32)

    Returns:
        List[ndarray] or ndarray: Hamiltonian(s) representing payoff matrices
            - If 'H': List of L arrays, each shape (2,)*2L
            - If 'H_all_in_one': Single stacked array, shape (L, (2,)*2L)

    Implementation:
        Uses graph-based construction from more_players.ipynb. Players are arranged
        on a ring (0-1-2-...-L-1-0), each interacting with 2 neighbors. Pairwise
        interactions use the same payoff structure as 2-player Prisoner's Dilemma.

        For L=2, this reduces to the standard 2-player game.
        For L>=3, each player has exactly 2 neighbors on the ring.
    """
    if L < 2:
        raise ValueError(f"Number of players must be >= 2, got {L}")

    # Pairwise payoff from more_players.ipynb (same as 2-player game)
    

    # Ring graph topology: 0-1-2-...-L-1-0
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
            H_2site = Hs[idx_in_edge].reshape(2, 2, 2, 2)

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


def get_default_4players(option: str = 'H', dtype: np.dtype = np.float32):
    """
    Backward compatibility wrapper for 4-player cyclic game.

    Returns the default Hamiltonian for the 4-player quantum game on a ring graph.
    This is equivalent to get_default_cyclic_players(L=4, ...).

    Args:
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        dtype: Data type for arrays (default: np.float32)

    Returns:
        List[ndarray] or ndarray: Hamiltonian(s) representing payoff matrices
            - If 'H': List of 4 arrays, each shape (2,2,2,2,2,2,2,2)
            - If 'H_all_in_one': Single stacked array, shape (4,2,2,2,2,2,2,2,2)
    """
    return get_default_cyclic_players(L=4, option=option, dtype=dtype)

def get_default_H(num_players: int = 3, option: str = 'H', dtype: np.dtype = np.float32):
    """
    Returns the default Hamiltonian for num_players quantum game.

    Args:
        num_players: Number of players (must be >= 2)
        option: Return format - 'H' for list, 'H_all_in_one' for stacked tensor
        dtype: Data type for arrays (default: np.float32)

    Returns:
        List[ndarray] or ndarray: Hamiltonian(s) for the game

    Implementation:
        - For 2 players: Uses standard 2-player Prisoner's Dilemma payoffs
        - For 3 players: Uses special 3-player symmetric payoffs
        - For 4+ players: Uses cyclic/ring graph topology with pairwise interactions
    """
    if num_players == 2:
        return get_default_2players(option=option, dtype=dtype)
    elif num_players == 3:
        return get_default_3players(option=option, dtype=dtype)
    elif num_players >= 4:
        return get_default_cyclic_players(L=num_players, option=option, dtype=dtype)
    else:
        raise ValueError(f"Invalid number of players: {num_players}. Must be >= 2")