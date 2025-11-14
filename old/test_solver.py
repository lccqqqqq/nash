"""
Test the solvers for the Nash equilibrium of multiplayer games in the normal form (with continuous strategy spaces). Should work for generic number of players.

Not sure what's going on...
"""

import torch as t
from solver import procrustes_solver, grad_solver
from mps_utils import get_ghz_state, get_product_state, get_rand_mps, to_comp_basis
from game import get_default_2players, get_default_3players
import einops
import math

default_dtype = t.complex64
device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))


def apply_unitary(U, Psi_tensor: t.Tensor):
    return einops.einsum(U, Psi_tensor, "new_phys phys, phys chi_l chi_r -> new_phys chi_l chi_r")

def apply_random_unitaries(Psi: t.Tensor, symmetric: bool = False):
    alpha_0 = t.pi * t.rand(1)
    n_0 = t.randn(3)
    n_0 = n_0 / t.linalg.norm(n_0)

    X = t.tensor([[0,1],[1,0]], dtype=Psi[0].dtype, device=Psi[0].device)
    Y = t.tensor([[0,-1j],[1j,0]], dtype=Psi[0].dtype, device=Psi[0].device)
    Z = t.tensor([[1,0],[0,-1]], dtype=Psi[0].dtype, device=Psi[0].device)

    for i in range(len(Psi)):
        if symmetric:
            alpha = alpha_0
            n = n_0
        else:
            alpha = t.pi * t.rand(1)
            n = t.randn(3)
            n = n / t.linalg.norm(n)
        U = t.eye(2, dtype=Psi[i].dtype, device=Psi[i].device) * math.cos(alpha) + 1j * (n[0] * X + n[1] * Y + n[2] * Z) * math.sin(alpha)
        Psi[i] = apply_unitary(U, Psi[i])
    return Psi

def get_default_H(num_players: int = 2):
    if num_players == 2:
        return get_default_2players(option='H', default_dtype=default_dtype, device=device)
    elif num_players == 3:
        return get_default_3players(option='H', default_dtype=default_dtype, device=device)
    else:
        raise ValueError("Invalid number of players")

def test_procrustes_solver(num_players: int = 2, symmetric: bool = False):
    Psi = get_ghz_state(L=2, default_dtype=default_dtype, device=device)
    Psi = apply_random_unitaries(Psi, symmetric=symmetric)
    H = get_default_H(num_players)
    result = procrustes_solver(Psi, H, max_iter=1000, alpha=0.01, grad_norm_threshold=1e-6, return_history=True)
    print(result)

def test_grad_solver(num_players: int = 2, symmetric: bool = False):
    Psi = get_ghz_state(L=num_players, default_dtype=default_dtype, device=device)
    Psi = apply_random_unitaries(Psi, symmetric=symmetric)
    print(Psi)
    H = get_default_H(num_players)
    result = grad_solver(Psi, H, max_iter=1000, alpha=0.01, grad_norm_threshold=1e-6, return_history=True)
    print(result)


if __name__ == "__main__":
    H = get_default_H(3)
    print(H)