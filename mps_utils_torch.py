"""
PyTorch version of MPS utilities (DEPRECATED - kept for reference)

This file contains the original PyTorch implementation.
Use mps_utils.py (NumPy version) for production code to avoid MPS complex number bugs.
"""

import torch as t
import einops
from typing import List
import math
import numpy as np
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def to_left_canonical_form(Psi: List[t.Tensor] | List[np.ndarray]):
    """
    Put the MPS in left canonical form from a random list of tensors (no orthogonality center provided yet, mostly used for initialization).
    """

    # start state:
    psi = Psi[0]
    n_sites = len(Psi)
    d_phys = Psi[0].shape[0]
    for j in range(n_sites):
        psi_grouped = einops.rearrange(psi, 'd_phys chi_l chi_r -> (d_phys chi_l) chi_r')
        if isinstance(psi_grouped, np.ndarray):
            left_iso, orth_center = np.linalg.qr(psi_grouped)
        else:
            left_iso, orth_center = t.linalg.qr(psi_grouped)

        Psi[j] = einops.rearrange(left_iso, '(d_phys chi_l) chi_r -> d_phys chi_l chi_r', d_phys=d_phys)
        if j < n_sites - 1:
            psi = einops.einsum(orth_center, Psi[j+1], 'chi_l bond, d_phys bond chi_r -> d_phys chi_l chi_r')
        else:
            # we are at the last site
            right_orth = orth_center.squeeze(-1)
    # the state is now automatically normalized to left-canonical form
    return Psi


def _swap_last_two_dims(psi):
    """Swap last two dimensions - works for both torch tensors and numpy arrays"""
    if isinstance(psi, t.Tensor):
        return psi.transpose(-2, -1)
    else:  # numpy array
        return np.swapaxes(psi, -2, -1)


def to_canonical_form(Psi: List[t.Tensor] | List[np.ndarray], form: str = 'A'):
    if form == 'A':
        return to_left_canonical_form(Psi)
    elif form == 'B':
        Psi_ = [_swap_last_two_dims(psi) for psi in Psi[::-1]]
        Psi_ = to_left_canonical_form(Psi_)
        Psi = [_swap_last_two_dims(psi) for psi in Psi_[::-1]]
        return Psi
    else:
        raise ValueError(f"Invalid form: {form} or not implemented yet")


def test_canonical_form(Psi: List[t.Tensor], form: str = 'A', atol: float = 1e-6):
    # test if the state is in left-canonical form
    if isinstance(Psi[0], np.ndarray):
        Psi = [t.from_numpy(psi) for psi in Psi]

    for j in range(len(Psi)):
        if form == 'A':
            T = einops.einsum(
                Psi[j], Psi[j].conj(),
                'd_phys chi_l chi_r, d_phys chi_l chi_rc -> chi_r chi_rc'
            )
            if not t.allclose(T, t.eye(T.shape[0], dtype=Psi[j].dtype, device=Psi[j].device), atol=atol):
                return False
        elif form == 'B':
            T = einops.einsum(
                Psi[j], Psi[j].conj(),
                'd_phys chi_l chi_r, d_phys chi_lc chi_r -> chi_l chi_lc'
            )
            if not t.allclose(T, t.eye(T.shape[0], dtype=Psi[j].dtype, device=Psi[j].device), atol=atol):
                return False
        else:
            raise ValueError(f"Invalid form: {form}")
    return True


def get_rand_mps(L: int = 3, chi: int = 4, d_phys: int = 2, seed: int | None = None, default_dtype: t.dtype = t.float32, device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu')), form: str = 'A', canonicalize: bool = True):
    if seed is not None:
        t.manual_seed(seed)
    Psi = [t.randn(d_phys, 1, chi, dtype=default_dtype, device=device)] + [t.randn(d_phys, chi, chi, dtype=default_dtype, device=device) for _ in range(L-2)] + [t.randn(d_phys, chi, 1, dtype=default_dtype, device=device)]
    if canonicalize:
        # sweep twice to ensure the state is at its minimal description
        Psi = to_canonical_form(Psi, form='A')
        Psi = to_canonical_form(Psi, form='B')
        print(f"Warning: canonicalized state may have different bond dimensions from specified")
        return to_canonical_form(Psi, form=form)
    else:
        return Psi


def to_comp_basis(Psi: List[t.Tensor] | List[np.ndarray]):
    """
    NOTE: This works for open boundary conditions only...
    """
    psi = Psi[0]
    for another_psi in Psi[1:]:
        psi = einops.einsum(psi, another_psi, "... chi_l bond, phys bond chi_r -> ... phys chi_l chi_r")

    return psi.squeeze().reshape(2**len(Psi))


def get_product_state(L: int = 3, state_per_site: List[int] | None = None,
                      default_dtype: t.dtype = t.float32,
                      device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))):
    """
    Create a product state in MPS form with OBC (Open Boundary Conditions).

    Args:
        L: Number of sites (default: 3)
        state_per_site: List of local states for each site, e.g., [0, 0, 0] for |000⟩
                       If None, defaults to all |0⟩ states
        default_dtype: Data type (default: float32)
        device: Device (default: auto-select)

    Returns:
        List[Tensor]: MPS tensors with bond dimension χ=1 (no entanglement)

    Example:
        |000⟩: get_product_state(3, [0, 0, 0])
        |101⟩: get_product_state(3, [1, 0, 1])
    """
    if state_per_site is None:
        state_per_site = [0] * L

    assert len(state_per_site) == L, f"state_per_site must have length {L}"

    Psi = []
    for i in range(L):
        # Create tensor with bond dimension 1
        if i == 0:
            # First site: shape (2, 1, 1)
            A = t.zeros(2, 1, 1, dtype=default_dtype, device=device)
            A[state_per_site[i], 0, 0] = 1.0
        elif i == L - 1:
            # Last site: shape (2, 1, 1)
            A = t.zeros(2, 1, 1, dtype=default_dtype, device=device)
            A[state_per_site[i], 0, 0] = 1.0
        else:
            # Middle sites: shape (2, 1, 1)
            A = t.zeros(2, 1, 1, dtype=default_dtype, device=device)
            A[state_per_site[i], 0, 0] = 1.0

        Psi.append(A)

    return Psi


def get_ghz_state(L: int = 3,
                  default_dtype: t.dtype = t.float32,
                  device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))):
    """
    Create a GHZ state in MPS form with OBC: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2

    Args:
        L: Number of sites (default: 3)
        default_dtype: Data type (default: float32)
        device: Device (default: auto-select)

    Returns:
        List[Tensor]: MPS tensors with bond dimension χ=2

    Example:
        3-qubit GHZ: |GHZ₃⟩ = (|000⟩ + |111⟩)/√2
        4-qubit GHZ: |GHZ₄⟩ = (|0000⟩ + |1111⟩)/√2

    MPS structure:
        - Site 0: Creates superposition and passes "which branch" info via bond
        - Sites 1 to L-2: Propagate the branch information
        - Site L-1: Close the MPS
    """
    Psi = []
    norm = t.sqrt(t.tensor(2.0, dtype=default_dtype, device=device))

    # First site: shape (2, 1, 2)
    # |0⟩ → bond [1, 0]/√2 (selects |0⟩ branch)
    # |1⟩ → bond [0, 1]/√2 (selects |1⟩ branch)
    A0 = t.zeros(2, 1, 2, dtype=default_dtype, device=device)
    A0[0, 0, 0] = 1.0 / norm  # |0⟩ state
    A0[1, 0, 1] = 1.0 / norm  # |1⟩ state
    Psi.append(A0)

    # Middle sites: shape (2, 2, 2)
    # Propagate the branch: bond_in[0] → |0⟩ → bond_out[0]
    #                       bond_in[1] → |1⟩ → bond_out[1]
    for i in range(1, L - 1):
        A = t.zeros(2, 2, 2, dtype=default_dtype, device=device)
        A[0, 0, 0] = 1.0  # |0⟩ branch
        A[1, 1, 1] = 1.0  # |1⟩ branch
        Psi.append(A)

    # Last site: shape (2, 2, 1)
    # Close the MPS: bond_in[0] → |0⟩, bond_in[1] → |1⟩
    AL = t.zeros(2, 2, 1, dtype=default_dtype, device=device)
    AL[0, 0, 0] = 1.0  # |0⟩ state
    AL[1, 1, 0] = 1.0  # |1⟩ state
    Psi.append(AL)

    return Psi


def apply_random_unitaries(state,
                           epsilon: float = 0.1,
                           sites: List[int] | None = None,
                           seed: int | None = None,
                           default_dtype: t.dtype = t.float32,
                           device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu')),
                           symmetric: bool = False):
    """
    Apply random single-qubit unitaries to a quantum state.

    Args:
        state: Either MPS form (List[Tensor]) or computational basis (Tensor)
        epsilon: Strength of random kick (default: 0.1)
                For small epsilon, U ≈ I + iε·(random Hermitian)
        sites: List of site indices to apply unitaries to. If None, apply to all sites.
        seed: Random seed for reproducibility
        device: Device to use. If None, infer from state.
        symmetric: Whether to use symmetric unitaries.
    Returns:
        Kicked state in the same format as input

    Examples:
        # MPS form
        Psi = get_product_state(3)
        Psi_kicked = apply_random_unitaries(Psi, epsilon=0.05)

        # Computational basis
        psi = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        psi_kicked = apply_random_unitaries(psi, epsilon=0.05)
    """
    if seed is not None:
        t.manual_seed(seed)

    # Detect input format
    is_mps = isinstance(state, list)

    if is_mps:
        # MPS form
        Psi = state
        L = len(Psi)
        if device is None:
            device = Psi[0].device
        dtype = Psi[0].dtype

        # Create U0 with detected dtype
        U0 = get_random_near_id_unitary(epsilon, dtype, device)

        if sites is None:
            sites = list(range(L))

        # Generate random unitaries for each site
        Psi_kicked = [psi.clone() for psi in Psi]

        for site in sites:
            # Generate random unitary close to identity
            U = U0 if symmetric else get_random_near_id_unitary(epsilon, dtype, device)

            # Apply to physical leg: new_psi[phys, chi_l, chi_r] = sum_phys' U[phys, phys'] * psi[phys', chi_l, chi_r]
            Psi_kicked[site] = einops.einsum(
                U, Psi[site],
                "phys_new phys_old, phys_old chi_l chi_r -> phys_new chi_l chi_r"
            )

        return Psi_kicked

    else:
        # Computational basis form
        psi = state
        if device is None:
            device = psi.device
        dtype = psi.dtype

        # Create U0 with detected dtype
        U0 = get_random_near_id_unitary(epsilon, dtype, device)

        # Infer number of qubits
        n = psi.numel()
        L = int(t.log2(t.tensor(float(n))).item())
        assert 2**L == n, f"State dimension {n} is not a power of 2"

        if sites is None:
            sites = list(range(L))

        # Reshape to multi-index tensor
        psi_tensor = psi.reshape([2] * L)

        for site in sites:
            # Generate random unitary
            U = U0 if symmetric else get_random_near_id_unitary(epsilon, dtype, device)

            # Contract with the specific site
            # Move site to first position, apply U, move back
            axes_order = [site] + [i for i in range(L) if i != site]
            psi_tensor = psi_tensor.permute(axes_order)

            # Apply U to first index
            psi_tensor = einops.einsum(
                U, psi_tensor,
                "phys_new phys_old, phys_old ... -> phys_new ..."
            )

            # Restore original order
            inv_order = [axes_order.index(i) for i in range(L)]
            psi_tensor = psi_tensor.permute(inv_order)

        # Flatten back to vector
        return psi_tensor.reshape(-1)


def get_random_near_id_unitary(eps=5e-2, default_dtype: t.dtype = t.float32, device: t.device = t.device('cuda' if t.cuda.is_available() else ('mps' if t.backends.mps.is_available() else 'cpu'))):
    """
    Generates a unitary matrix close to identity.

    Args:
        eps: Perturbation strength (default: 5e-2)

    Returns:
        Tensor: 2×2 unitary matrix approximately equal to identity

    Implementation:
        U = (1 - ε²/2)I + ε·iσ_y where σ_y is Pauli-Y matrix
    """
    return t.eye(2, dtype=default_dtype, device=device) * math.cos(eps) + math.sin(eps) * t.tensor([[0, 1], [-1, 0]], dtype=default_dtype, device=device)
