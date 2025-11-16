import einops
from typing import List
import math
import numpy as np


def to_left_canonical_form(Psi: List[np.ndarray]):
    """
    Put the MPS in left canonical form from a random list of tensors (no orthogonality center provided yet, mostly used for initialization).
    """

    # start state:
    psi = Psi[0]
    n_sites = len(Psi)
    d_phys = Psi[0].shape[0]
    for j in range(n_sites):
        psi_grouped = einops.rearrange(psi, 'd_phys chi_l chi_r -> (d_phys chi_l) chi_r')
        left_iso, orth_center = np.linalg.qr(psi_grouped)

        Psi[j] = einops.rearrange(left_iso, '(d_phys chi_l) chi_r -> d_phys chi_l chi_r', d_phys=d_phys)
        if j < n_sites - 1:
            psi = einops.einsum(orth_center, Psi[j+1], 'chi_l bond, d_phys bond chi_r -> d_phys chi_l chi_r')
        else:
            # we are at the last site
            right_orth = orth_center.squeeze(-1)
    # the state is now automatically normalized to left-canonical form
    return Psi


def to_canonical_form(Psi: List[np.ndarray], form: str = 'A'):
    """Convert MPS to canonical form (A=left, B=right)"""
    if form == 'A':
        return to_left_canonical_form(Psi)
    elif form == 'B':
        Psi_ = [np.swapaxes(psi, -2, -1) for psi in Psi[::-1]]
        Psi_ = to_left_canonical_form(Psi_)
        Psi = [np.swapaxes(psi, -2, -1) for psi in Psi_[::-1]]
        return Psi
    else:
        raise ValueError(f"Invalid form: {form} or not implemented yet")


def test_canonical_form(Psi: List[np.ndarray], form: str = 'A', atol: float = 1e-6):
    """Test if the state is in canonical form"""
    for j in range(len(Psi)):
        if form == 'A':
            T = einops.einsum(
                Psi[j], Psi[j].conj(),
                'd_phys chi_l chi_r, d_phys chi_l chi_rc -> chi_r chi_rc'
            )
            if not np.allclose(T, np.eye(T.shape[0], dtype=Psi[j].dtype), atol=atol):
                return False
        elif form == 'B':
            T = einops.einsum(
                Psi[j], Psi[j].conj(),
                'd_phys chi_l chi_r, d_phys chi_lc chi_r -> chi_l chi_lc'
            )
            if not np.allclose(T, np.eye(T.shape[0], dtype=Psi[j].dtype), atol=atol):
                return False
        else:
            raise ValueError(f"Invalid form: {form}")
    return True


def get_rand_mps(L: int = 3, chi: int = 4, d_phys: int = 2, seed: int | None = None,
                 dtype: np.dtype = np.float32, form: str = 'A', canonicalize: bool = True):
    """
    Generate random MPS with specified bond dimension.

    Args:
        L: Number of sites
        chi: Bond dimension
        d_phys: Physical dimension (default: 2 for qubits)
        seed: Random seed for reproducibility
        dtype: NumPy dtype (e.g., np.float32, np.complex64)
        form: Canonical form ('A' for left, 'B' for right)
        canonicalize: Whether to put in canonical form
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random complex arrays if dtype is complex
    if np.issubdtype(dtype, np.complexfloating):
        Psi = [
            (np.random.randn(d_phys, 1, chi) + 1j * np.random.randn(d_phys, 1, chi)).astype(dtype)
        ] + [
            (np.random.randn(d_phys, chi, chi) + 1j * np.random.randn(d_phys, chi, chi)).astype(dtype)
            for _ in range(L-2)
        ] + [
            (np.random.randn(d_phys, chi, 1) + 1j * np.random.randn(d_phys, chi, 1)).astype(dtype)
        ]
    else:
        Psi = [
            np.random.randn(d_phys, 1, chi).astype(dtype)
        ] + [
            np.random.randn(d_phys, chi, chi).astype(dtype)
            for _ in range(L-2)
        ] + [
            np.random.randn(d_phys, chi, 1).astype(dtype)
        ]

    if canonicalize:
        # sweep twice to ensure the state is at its minimal description
        Psi = to_canonical_form(Psi, form='A')
        Psi = to_canonical_form(Psi, form='B')
        print(f"Warning: canonicalized state may have different bond dimensions from specified")
        return to_canonical_form(Psi, form=form)
    else:
        return Psi


def to_comp_basis(Psi: List[np.ndarray]):
    """
    Convert MPS to computational basis state vector.
    NOTE: This works for open boundary conditions only.
    """
    psi = Psi[0]
    for another_psi in Psi[1:]:
        psi = einops.einsum(psi, another_psi, "... chi_l bond, phys bond chi_r -> ... phys chi_l chi_r")

    return psi.squeeze().reshape(2**len(Psi))


def get_product_state(L: int = 3, state_per_site: List[int] | None = None,
                      dtype: np.dtype = np.float32):
    """
    Create a product state in MPS form with OBC (Open Boundary Conditions).

    Args:
        L: Number of sites (default: 3)
        state_per_site: List of local states for each site, e.g., [0, 0, 0] for |000⟩
                       If None, defaults to all |0⟩ states
        dtype: NumPy data type (default: np.float32)

    Returns:
        List[np.ndarray]: MPS tensors with bond dimension χ=1 (no entanglement)

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
        A = np.zeros((2, 1, 1), dtype=dtype)
        A[state_per_site[i], 0, 0] = 1.0
        Psi.append(A)

    return Psi


def get_ghz_state(L: int = 3, dtype: np.dtype = np.float32):
    """
    Create a GHZ state in MPS form with OBC: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2

    Args:
        L: Number of sites (default: 3)
        dtype: NumPy data type (default: np.float32)

    Returns:
        List[np.ndarray]: MPS tensors with bond dimension χ=2

    Example:
        3-qubit GHZ: |GHZ₃⟩ = (|000⟩ + |111⟩)/√2
        4-qubit GHZ: |GHZ₄⟩ = (|0000⟩ + |1111⟩)/√2

    MPS structure:
        - Site 0: Creates superposition and passes "which branch" info via bond
        - Sites 1 to L-2: Propagate the branch information
        - Site L-1: Close the MPS
    """
    Psi = []
    norm = np.sqrt(2.0)

    # First site: shape (2, 1, 2)
    # |0⟩ → bond [1, 0]/√2 (selects |0⟩ branch)
    # |1⟩ → bond [0, 1]/√2 (selects |1⟩ branch)
    A0 = np.zeros((2, 1, 2), dtype=dtype)
    A0[0, 0, 0] = 1.0 / norm  # |0⟩ state
    A0[1, 0, 1] = 1.0 / norm  # |1⟩ state
    Psi.append(A0)

    # Middle sites: shape (2, 2, 2)
    # Propagate the branch: bond_in[0] → |0⟩ → bond_out[0]
    #                       bond_in[1] → |1⟩ → bond_out[1]
    for _ in range(1, L - 1):
        A = np.zeros((2, 2, 2), dtype=dtype)
        A[0, 0, 0] = 1.0  # |0⟩ branch
        A[1, 1, 1] = 1.0  # |1⟩ branch
        Psi.append(A)

    # Last site: shape (2, 2, 1)
    # Close the MPS: bond_in[0] → |0⟩, bond_in[1] → |1⟩
    AL = np.zeros((2, 2, 1), dtype=dtype)
    AL[0, 0, 0] = 1.0  # |0⟩ state
    AL[1, 1, 0] = 1.0  # |1⟩ state
    Psi.append(AL)

    return Psi


def apply_random_unitaries(state,
                           epsilon: float = 0.1,
                           sites: List[int] | None = None,
                           seed: int | None = None,
                           dtype: np.dtype = np.float32,
                           symmetric: bool = False):
    """
    Apply random single-qubit unitaries to a quantum state.

    Args:
        state: Either MPS form (List[np.ndarray]) or computational basis (np.ndarray)
        epsilon: Strength of random kick (default: 0.1)
                For small epsilon, U ≈ I + iε·(random Hermitian)
        sites: List of site indices to apply unitaries to. If None, apply to all sites.
        seed: Random seed for reproducibility
        dtype: NumPy dtype
        symmetric: Whether to use symmetric unitaries.
    Returns:
        Kicked state in the same format as input

    Examples:
        # MPS form
        Psi = get_product_state(3)
        Psi_kicked = apply_random_unitaries(Psi, epsilon=0.05)

        # Computational basis
        psi = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
        psi_kicked = apply_random_unitaries(psi, epsilon=0.05)
    """
    if seed is not None:
        np.random.seed(seed)

    # Detect input format
    is_mps = isinstance(state, list)

    if is_mps:
        # MPS form
        Psi = state
        L = len(Psi)
        state_dtype = Psi[0].dtype

        # Create U0 with detected dtype
        U0 = get_random_near_id_unitary(epsilon, state_dtype)

        if sites is None:
            sites = list(range(L))

        # Generate random unitaries for each site
        Psi_kicked = [psi.copy() for psi in Psi]

        for site in sites:
            # Generate random unitary close to identity
            U = U0 if symmetric else get_random_near_id_unitary(epsilon, state_dtype)

            # Apply to physical leg: new_psi[phys, chi_l, chi_r] = sum_phys' U[phys, phys'] * psi[phys', chi_l, chi_r]
            Psi_kicked[site] = einops.einsum(
                U, Psi[site],
                "phys_new phys_old, phys_old chi_l chi_r -> phys_new chi_l chi_r"
            )

        return Psi_kicked

    else:
        # Computational basis form
        psi = state
        state_dtype = psi.dtype

        # Create U0 with detected dtype
        U0 = get_random_near_id_unitary(epsilon, state_dtype)

        # Infer number of qubits
        n = psi.size
        L = int(np.log2(n))
        assert 2**L == n, f"State dimension {n} is not a power of 2"

        if sites is None:
            sites = list(range(L))

        # Reshape to multi-index tensor
        psi_tensor = psi.reshape([2] * L)

        for site in sites:
            # Generate random unitary
            U = U0 if symmetric else get_random_near_id_unitary(epsilon, state_dtype)

            # Contract with the specific site
            # Move site to first position, apply U, move back
            axes_order = [site] + [i for i in range(L) if i != site]
            psi_tensor = np.transpose(psi_tensor, axes_order)

            # Apply U to first index
            psi_tensor = einops.einsum(
                U, psi_tensor,
                "phys_new phys_old, phys_old ... -> phys_new ..."
            )

            # Restore original order
            inv_order = [axes_order.index(i) for i in range(L)]
            psi_tensor = np.transpose(psi_tensor, inv_order)

        # Flatten back to vector
        return psi_tensor.reshape(-1)


def get_random_near_id_unitary(eps: float = 5e-2, dtype: np.dtype = np.float32):
    """
    Generates a unitary matrix close to identity.

    Args:
        eps: Perturbation strength (default: 5e-2)
        dtype: NumPy dtype

    Returns:
        np.ndarray: 2×2 unitary matrix approximately equal to identity

    Implementation:
        U = cos(ε)·I + sin(ε)·iσ_y where σ_y is Pauli-Y matrix
    """
    return np.eye(2, dtype=dtype) * math.cos(eps) + math.sin(eps) * np.array([[0, 1], [-1, 0]], dtype=dtype)
