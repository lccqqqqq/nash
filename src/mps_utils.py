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

def get_rand_state_as_mps(L: int = 3, max_bond_dim: int | None = None, seed: int | None = None,
                          cutoff: float = 1e-16, canonicalize: bool = True, dtype: np.dtype = np.float64):
    """
    Generate random state as MPS with specified bond dimension.

    Args:
        L: Number of sites
        max_bond_dim: Maximum bond dimension (chi)
        seed: Random seed for reproducibility (optional)
        cutoff: SVD truncation threshold
        canonicalize: Whether to canonicalize (currently unused)
        dtype: Data type for the state

    Returns:
        List[np.ndarray]: MPS representation of random Haar state
    """
    if seed is not None:
        np.random.seed(seed)

    if dtype == np.complex64 or dtype == np.complex128:
        psi = np.random.randn(2**L) + 1j * np.random.randn(2**L)
    else:
        psi = np.random.randn(2**L)
    psi = psi / np.linalg.norm(psi)
    return from_comp_basis(psi, L=L, max_bond_dim=max_bond_dim, cutoff=cutoff)


def get_rand_mps(L: int = 3, chi: int = 4, d_phys: int = 2, seed: int | None = None,
                 dtype: np.dtype = np.float64, form: str = 'A', canonicalize: bool = True):
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

def from_comp_basis(state_vector, L=None, max_bond_dim=None, cutoff=1e-16):
    """
    Converts a dense quantum state vector into a Matrix Product State (MPS)
    representation using Singular Value Decomposition (SVD).
    
    The resulting MPS is in left-canonical form (gamma-lambda form without the lambdas explicit).
    
    Args:
        state_vector (np.ndarray): The quantum state vector of shape (2**L,).
        L (int, optional): The number of qubits/sites. If None, inferred from vector length.
        max_bond_dim (int, optional): The maximum bond dimension (chi) to keep. 
                                      If None, no truncation is applied.
        cutoff (float, optional): Singular value truncation threshold. 
                                  Singular values smaller than this are discarded.

    Returns:
        list[np.ndarray]: A list of length L, where the i-th element is a tensor 
                          representing the MPS tensor at site i.
                          Shapes are usually (chi_left, physical_dim, chi_right).
                          Physical dimension is assumed to be 2 (qubits).
    """
    
    # 1. Validation and Setup
    N = state_vector.size
    if L is None:
        L = int(np.log2(N))
    
    if 2**L != N:
        raise ValueError(f"State vector size {N} is not a power of 2 (expected 2^{L}).")

    # Reshape the vector into a 2D-like structure to start the SVD chain
    # We treat the remaining part of the system as the 'columns'
    psi = np.array(state_vector).reshape(1, -1)
    
    mps_tensors = []
    
    # Iterate through sites 0 to L-2
    for i in range(L - 1):
        # Current shape of psi is (r_{i-1}, 2 * 2^{L-1-i})
        # We want to isolate the physical index of the current site 'i'
        # Reshape to: (r_{i-1} * 2,  remaining_dim)
        r_prev = psi.shape[0]
        remaining_sites = L - 1 - i
        dim_right = 2**remaining_sites
        
        # Reshape for SVD: Combine (alpha_{i-1}, sigma_i) into rows
        psi_reshaped = psi.reshape(r_prev * 2, dim_right)
        
        # Perform SVD
        # U: (rows, k), S: (k,), Vh: (k, cols)
        U, S, Vh = np.linalg.svd(psi_reshaped, full_matrices=False)
        
        # --- Truncation Logic ---
        # Determine number of singular values to keep
        keep_indices = S > cutoff
        if max_bond_dim is not None:
            # Keep at most max_bond_dim, but also respect the cutoff
            limit = min(max_bond_dim, len(S))
            # We want the top 'limit' indices that also satisfy the cutoff
            # Since S is sorted descending, we just slice.
            keep_count = min(np.sum(keep_indices), limit)
        else:
            keep_count = np.sum(keep_indices)
            
        # Ensure we keep at least one singular value to avoid empty tensors
        keep_count = max(keep_count, 1)
        
        # Slice matrices
        U = U[:, :keep_count]
        S = S[:keep_count]
        Vh = Vh[:keep_count, :]
        
        # Reshape U back to a rank-3 MPS tensor: (r_{i-1}, physical=2, r_i)
        # U was (r_{i-1} * 2, keep_count)
        mps_tensor = U.reshape(r_prev, 2, keep_count)
        mps_tensors.append(mps_tensor)
        
        # Prepare the remainder state for the next step
        # Contract S and Vh to form the matrix for the next iteration
        psi = np.diag(S) @ Vh
        
    # Handle the last site
    # psi is now (r_{L-1}, 2). The last tensor is just this remaining matrix
    # Reshaped to standard MPS form (r_{L-1}, 2, 1)
    last_tensor = psi.reshape(psi.shape[0], 2, 1)
    mps_tensors.append(last_tensor)
    
    mps_tensors = [einops.rearrange(tensor, "chi_l a0 chi_r -> a0 chi_l chi_r") for tensor in mps_tensors]
    return mps_tensors


def get_product_state(L: int = 3, state_per_site: List[int] | None = None,
                      dtype: np.dtype = np.float64):
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


def get_ghz_state(L: int = 3, dtype: np.dtype = np.float64):
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


def get_low_entanglement_2qubit_state(
    lambda1: float,
    randomize_basis: bool = True,
    seed: int | None = None,
    dtype: np.dtype = np.float64
) -> list[np.ndarray]:
    """
    Create 2-qubit state with specified Schmidt eigenvalue.

    The state is created in Schmidt form: |ψ⟩ = λ₁|00⟩ + λ₂|11⟩
    where λ₂ = √(1 - λ₁²), and then optionally rotated by random local unitaries.

    Args:
        lambda1: Larger Schmidt eigenvalue (λ₁ ∈ [1/√2, 1])
                λ₁ = 1: product state (no entanglement)
                λ₁ = 1/√2 ≈ 0.707: maximally entangled (Bell state)
        randomize_basis: If True, apply random local SU(2) unitaries to each qubit
        seed: Random seed for reproducibility
        dtype: NumPy data type (default: np.float64)

    Returns:
        List[np.ndarray]: MPS representation of the 2-qubit state

    Example:
        # Create product state (no entanglement)
        Psi_product = get_low_entanglement_2qubit_state(lambda1=1.0, randomize_basis=False)

        # Create low-entanglement state with random basis
        Psi_low_ent = get_low_entanglement_2qubit_state(lambda1=0.95, seed=42)
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate lambda1
    min_lambda = 1.0 / np.sqrt(2.0)
    if not (min_lambda - 1e-10 <= lambda1 <= 1.0 + 1e-10):
        raise ValueError(f"lambda1 must be in [{min_lambda:.6f}, 1.0], got {lambda1}")

    # Compute lambda2 from normalization
    lambda2 = np.sqrt(1.0 - lambda1**2)

    # Create state in computational basis: |ψ⟩ = λ₁|00⟩ + λ₂|11⟩
    psi = np.zeros(4, dtype=dtype)
    psi[0] = lambda1  # |00⟩
    psi[3] = lambda2  # |11⟩

    # Optionally apply random local unitaries
    if randomize_basis:
        # Generate random SU(2) unitaries for each qubit
        def random_su2(dtype):
            """Generate random SU(2) unitary using Haar measure."""
            # Use parameterization: U = exp(i * (α σ_x + β σ_y + γ σ_z))
            # Sample uniformly on sphere for axis, uniform angle
            if np.issubdtype(dtype, np.complexfloating):
                # Complex case: full SU(2)
                alpha, beta, gamma = np.random.randn(3)
                norm = np.sqrt(alpha**2 + beta**2 + gamma**2)
                if norm < 1e-10:
                    return np.eye(2, dtype=dtype)
                alpha, beta, gamma = alpha/norm, beta/norm, gamma/norm

                theta = np.random.uniform(0, 2*np.pi)
                cos_t = np.cos(theta/2)
                sin_t = np.sin(theta/2)

                U = np.array([
                    [cos_t - 1j*gamma*sin_t, (-1j*alpha - beta)*sin_t],
                    [(-1j*alpha + beta)*sin_t, cos_t + 1j*gamma*sin_t]
                ], dtype=dtype)
            else:
                # Real case: use exp(iY) which gives real matrices
                # Random rotation in SO(2) subgroup
                theta = np.random.uniform(0, 2*np.pi)
                U = np.array([
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]
                ], dtype=dtype)

            return U

        U1 = random_su2(dtype)
        U2 = random_su2(dtype)

        # Apply unitaries: (U1 ⊗ U2) |ψ⟩
        U_total = np.kron(U1, U2)
        psi = U_total @ psi

    # Convert to MPS format
    Psi = from_comp_basis(psi, L=2)

    return Psi


def apply_random_unitaries(state,
                           epsilon: float = 0.1,
                           sites: List[int] | None = None,
                           seed: int | None = None,
                           dtype: np.dtype = np.float64,
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


def get_random_near_id_unitary(eps: float = 5e-2, dtype: np.dtype = np.float64):
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


def apply_unitary(unitary: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Apply a unitary gate to the physical leg of an MPS tensor.

    Args:
        unitary: Unitary matrix of shape (2, 2)
        A: MPS tensor of shape (phys, χ_L, χ_R)

    Returns:
        np.ndarray: Transformed MPS tensor with same shape

    Implementation:
        Contracts unitary with physical index: A'[new_phys] = Σ_phys U[new_phys, phys] A[phys]
        Represents local quantum operations (gates) applied by each player.

    Example:
        >>> U = np.array([[0, 1], [1, 0]])  # Pauli-X
        >>> A = get_product_state(1, [0])[0]  # |0⟩ state
        >>> A_new = apply_unitary(U, A)      # Now |1⟩ state
    """
    A = einops.einsum(
        A, unitary, "phys chi_l chi_r, new_phys phys -> new_phys chi_l chi_r"
    )
    return A
