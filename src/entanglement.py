"""
Entanglement parameter calculations for quantum states.

This module provides functions for computing entanglement invariants
and measures for 3-qubit and 4-qubit quantum states.
"""

import numpy as np
import einops


def compute_purity(rho: np.ndarray) -> float:
    """
    Compute purity Tr(rho^2) of a density matrix.

    Args:
        rho: Density matrix (square array)

    Returns:
        Purity value in [0, 1]. Equals 1 for pure states.
    """
    return np.real(np.trace(np.linalg.matrix_power(rho, 2)))


def partial_trace(psi: np.ndarray, keep: list[int]) -> np.ndarray:
    """
    Compute reduced density matrix by tracing out specified subsystems.

    Args:
        psi: Quantum state as tensor with shape (d1, d2, ..., dn)
        keep: List of subsystem indices to keep

    Returns:
        Reduced density matrix for the kept subsystems
    """
    n = psi.ndim
    trace_out = [i for i in range(n) if i not in keep]

    # Contract psi with psi* over traced-out indices
    # Build einsum string dynamically
    psi_indices = list(range(n))
    psi_conj_indices = list(range(n, 2*n))

    # For kept indices, use different labels for ket and bra
    # For traced indices, use same labels
    for i in trace_out:
        psi_conj_indices[i] = psi_indices[i]

    # Build output indices (kept indices from both ket and bra)
    out_indices = [psi_indices[i] for i in keep] + [psi_conj_indices[i] for i in keep]

    # Use einops for the contraction
    keep_str = ' '.join([f'i{i}' for i in keep])
    trace_str = ' '.join([f'j{i}' for i in trace_out])

    if trace_out:
        # Build einsum specification
        ket_spec = ' '.join([f'i{i}' if i in keep else f'j{i}' for i in range(n)])
        bra_spec = ' '.join([f'k{i}' if i in keep else f'j{i}' for i in range(n)])
        out_spec = ' '.join([f'i{i}' for i in keep] + [f'k{i}' for i in keep])
        spec = f'{ket_spec}, {bra_spec} -> {out_spec}'
        rho = einops.einsum(psi, psi.conj(), spec)
    else:
        # Keep all - just outer product
        rho = np.outer(psi.flatten(), psi.conj().flatten())

    # Reshape to matrix form
    kept_dim = int(np.prod([psi.shape[i] for i in keep]))
    return rho.reshape(kept_dim, kept_dim)


def compute_D(psi: np.ndarray, i: int, j: int) -> complex:
    """
    Compute the D matrix determinant for a pair of qubits (4-qubit states only).

    Helper function for 4-qubit SLOCC invariants.
    Uses precomputed formulas for efficiency.

    Args:
        psi: 4-qubit state vector (16 elements)
        i, j: Qubit pair indices (0-3)

    Returns:
        Determinant of the D matrix for qubits (i, j)
    """
    a = psi.flatten()

    # Dispatch table for all 6 possible pairs
    # Maps sorted (i,j) pairs to their corresponding D matrix computation
    pair = tuple(sorted([i, j]))

    if pair == (0, 1):  # xy
        D = np.array([
            [-a[1]*a[2] + a[0]*a[3], a[3]*a[4] - a[2]*a[5] - a[1]*a[6] + a[0]*a[7], -a[5]*a[6] + a[4]*a[7]],
            [a[3]*a[8] - a[2]*a[9] - a[1]*a[10] + a[0]*a[11], a[7]*a[8] - a[6]*a[9] - a[5]*a[10] + a[4]*a[11] + a[3]*a[12] - a[2]*a[13] - a[1]*a[14] + a[0]*a[15], a[7]*a[12] - a[6]*a[13] - a[5]*a[14] + a[4]*a[15]],
            [-a[9]*a[10] + a[8]*a[11], a[11]*a[12] - a[10]*a[13] - a[9]*a[14] + a[8]*a[15], -a[13]*a[14] + a[12]*a[15]]
        ])
    elif pair == (0, 2):  # xz
        D = np.array([
            [-a[1]*a[4] + a[0]*a[5], -a[3]*a[4] + a[2]*a[5] - a[1]*a[6] + a[0]*a[7], -a[3]*a[6] + a[2]*a[7]],
            [a[5]*a[8] - a[4]*a[9] - a[1]*a[12] + a[0]*a[13], a[7]*a[8] - a[6]*a[9] + a[5]*a[10] - a[4]*a[11] - a[3]*a[12] + a[2]*a[13] - a[1]*a[14] + a[0]*a[15], a[7]*a[10] - a[6]*a[11] - a[3]*a[14] + a[2]*a[15]],
            [-a[9]*a[12] + a[8]*a[13], -a[11]*a[12] + a[10]*a[13] - a[9]*a[14] + a[8]*a[15], -a[11]*a[14] + a[10]*a[15]]
        ])
    elif pair == (0, 3):  # xt
        D = np.array([
            [-a[2]*a[4] + a[0]*a[6], -a[3]*a[4] - a[2]*a[5] + a[1]*a[6] + a[0]*a[7], -a[3]*a[5] + a[1]*a[7]],
            [a[6]*a[8] - a[4]*a[10] - a[2]*a[12] + a[0]*a[14], a[7]*a[8] + a[6]*a[9] - a[5]*a[10] - a[4]*a[11] - a[3]*a[12] - a[2]*a[13] + a[1]*a[14] + a[0]*a[15], a[7]*a[9] - a[5]*a[11] - a[3]*a[13] + a[1]*a[15]],
            [-a[10]*a[12] + a[8]*a[14], -a[11]*a[12] - a[10]*a[13] + a[9]*a[14] + a[8]*a[15], -a[11]*a[13] + a[9]*a[15]]
        ])
    elif pair == (1, 2):  # yz
        D = np.array([
            [-a[1]*a[8] + a[0]*a[9], -a[3]*a[8] + a[2]*a[9] - a[1]*a[10] + a[0]*a[11], -a[3]*a[10] + a[2]*a[11]],
            [-a[5]*a[8] + a[4]*a[9] - a[1]*a[12] + a[0]*a[13], -a[7]*a[8] + a[6]*a[9] - a[5]*a[10] + a[4]*a[11] - a[3]*a[12] + a[2]*a[13] - a[1]*a[14] + a[0]*a[15], -a[7]*a[10] + a[6]*a[11] - a[3]*a[14] + a[2]*a[15]],
            [-a[5]*a[12] + a[4]*a[13], -a[7]*a[12] + a[6]*a[13] - a[5]*a[14] + a[4]*a[15], -a[7]*a[14] + a[6]*a[15]]
        ])
    elif pair == (1, 3):  # yt
        D = np.array([
            [-a[2]*a[8] + a[0]*a[10], -a[3]*a[8] - a[2]*a[9] + a[1]*a[10] + a[0]*a[11], -a[3]*a[9] + a[1]*a[11]],
            [-a[6]*a[8] + a[4]*a[10] - a[2]*a[12] + a[0]*a[14], -a[7]*a[8] - a[6]*a[9] + a[5]*a[10] + a[4]*a[11] - a[3]*a[12] - a[2]*a[13] + a[1]*a[14] + a[0]*a[15], -a[7]*a[9] + a[5]*a[11] - a[3]*a[13] + a[1]*a[15]],
            [-a[6]*a[12] + a[4]*a[14], -a[7]*a[12] - a[6]*a[13] + a[5]*a[14] + a[4]*a[15], -a[7]*a[13] + a[5]*a[15]]
        ])
    elif pair == (2, 3):  # zt
        D = np.array([
            [-a[4]*a[8] + a[0]*a[12], -a[5]*a[8] - a[4]*a[9] + a[1]*a[12] + a[0]*a[13], -a[5]*a[9] + a[1]*a[13]],
            [-a[6]*a[8] - a[4]*a[10] + a[2]*a[12] + a[0]*a[14], -a[7]*a[8] - a[6]*a[9] - a[5]*a[10] - a[4]*a[11] + a[3]*a[12] + a[2]*a[13] + a[1]*a[14] + a[0]*a[15], -a[7]*a[9] - a[5]*a[11] + a[3]*a[13] + a[1]*a[15]],
            [-a[6]*a[10] + a[2]*a[14], -a[7]*a[10] - a[6]*a[11] + a[3]*a[14] + a[2]*a[15], -a[7]*a[11] + a[3]*a[15]]
        ])
    else:
        raise ValueError(f"Invalid qubit pair: ({i}, {j})")

    return np.linalg.det(D)


def compute_four_qubit_SLOCC_invariants(psi: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute SLOCC (Stochastic Local Operations and Classical Communication) invariants
    for 4-qubit states.

    Args:
        psi: 4-qubit state vector, shape (16,) or (2,2,2,2)

    Returns:
        tuple: (H, L, M, Dxt) - Four SLOCC invariants
            - H: Hyperdeterminant (degree-4 polynomial invariant)
            - L: Simple determinant of 4×4 reshaping
            - M: Determinant of specific index permutation
            - Dxt: D-invariant for qubits (0,3)

    Reference:
        These invariants characterize 4-qubit entanglement classes under SLOCC.
    """
    psi = psi.flatten()
    assert psi.shape == (16,), f"Expected 16-element state vector, got shape {psi.shape}"

    # Hyperdeterminant H
    H = (psi[0] * psi[15] - psi[1] * psi[14] - psi[2] * psi[13] + psi[3] * psi[12] -
         psi[4] * psi[11] + psi[5] * psi[10] + psi[6] * psi[9] - psi[7] * psi[8])

    # Simple determinant L
    L = np.linalg.det(psi.reshape(4, 4))

    # Permuted determinant M
    M = np.linalg.det([
        [psi[0], psi[8], psi[2], psi[10]],
        [psi[1], psi[9], psi[3], psi[11]],
        [psi[4], psi[12], psi[6], psi[14]],
        [psi[5], psi[13], psi[7], psi[15]],
    ])

    # D-invariant for qubits (0, 3) - x-t pair
    Dxt = compute_D(psi, 0, 3)

    return abs(H), abs(L), abs(M), abs(Dxt)


def compute_entanglement_params(state: np.ndarray, option: str = 'I') -> np.ndarray:
    """
    Compute entanglement parameters characterizing the quantum state structure.

    Args:
        state: Quantum state of shape (2,2,2) for 3 qubits or (2,2,2,2) for 4 qubits,
               or flattened (8,) or (16,) as numpy array
        option: Return 'I' invariants or 'J' parameters (default: 'I')
                Only applicable for 3-qubit states. 4-qubit states return SLOCC invariants.

    Returns:
        np.ndarray: Entanglement parameters
            - For 3 qubits: shape (5,) with I1-I5 or J1-J5
            - For 4 qubits: shape (4,) with (H, L, M, Dxt) SLOCC invariants

    For 3-qubit states with option='I' (invariants):
        - I1: Tr(ρ_1²) - Single-party purity for player 1
        - I2: Tr(ρ_2²) - Single-party purity for player 2
        - I3: Tr(ρ_3²) - Single-party purity for player 3
        - I4: Tr((ρ_1 ⊗ ρ_2) ρ_12) - Two-party correlation measure
        - I5: |det₃(ψ)|² - Three-party entanglement (generalized concurrence)

    For 3-qubit states with option='J' (derived parameters):
        - J1, J2, J3: Transformed purity measures
        - J4: √I5 - Concurrence
        - J5: Higher-order correlation measure

    For 4-qubit states:
        - H: Hyperdeterminant
        - L: Simple determinant
        - M: Permuted determinant
        - Dxt: D-invariant for qubits (0,3)

    Implementation:
        For 3 qubits: Computes reduced density matrices for all subsystems and uses
        Levi-Civita tensor for determinant computation.

        For 4 qubits: Computes SLOCC invariants that classify entanglement under
        stochastic local operations and classical communication.
    """
    # Detect number of qubits
    if state.ndim == 1:
        n_qubits = int(np.log2(len(state)))
        if n_qubits == 4:
            # 4-qubit case
            return np.array(compute_four_qubit_SLOCC_invariants(state))
        elif n_qubits == 3:
            state = state.reshape(2, 2, 2)
        else:
            raise ValueError(f"Unsupported number of qubits: {n_qubits}. Only 3 or 4 qubits supported.")
    elif state.shape == (2, 2, 2, 2):
        # 4-qubit case
        return np.array(compute_four_qubit_SLOCC_invariants(state))

    # 3-qubit case continues below
    # Compute reduced density matrices
    rho_1 = einops.einsum(state, state.conj(), 'x i j, y i j -> x y')
    rho_2 = einops.einsum(state, state.conj(), 'i x j, i y j -> x y')
    rho_3 = einops.einsum(state, state.conj(), 'i j x, i j y -> x y')
    rho_12 = einops.einsum(state, state.conj(), 'x1 x2 i, y1 y2 i -> x1 y1 x2 y2')
    rho_12 = einops.rearrange(rho_12, 'x1 y1 x2 y2 -> (x1 x2) (y1 y2)')

    # Compute invariants
    I1 = np.trace(np.linalg.matrix_power(rho_1, 2))
    I2 = np.trace(np.linalg.matrix_power(rho_2, 2))
    I3 = np.trace(np.linalg.matrix_power(rho_3, 2))
    I4 = np.trace(np.kron(rho_1, rho_2) @ rho_12)

    # Compute 3-party entanglement using Levi-Civita tensor
    eps = np.array([[0, 1], [-1, 0]], dtype=state.dtype)
    det3 = 1/2 * einops.einsum(
        eps, eps, eps, eps, eps, eps, state, state, state, state,
        'i1 j1, i2 j2, k1 l1, k2 l2, i3 k3, j3 l3, i1 i2 i3, j1 j2 j3, k1 k2 k3, l1 l2 l3 ->'
    )
    I5 = np.abs(det3) ** 2

    if option == 'I':
        return np.stack([I1, I2, I3, I4, I5])
    elif option == 'J':
        J1 = 1/4 * (1 + I1 - I2 - I3 - 2 * np.sqrt(I5))
        J2 = 1/4 * (1 - I1 + I2 - I3 - 2 * np.sqrt(I5))
        J3 = 1/4 * (1 - I1 - I2 + I3 - 2 * np.sqrt(I5))
        J4 = np.sqrt(I5)
        J5 = 1/4 * (3 - 3 * I1 - 3 * I2 - I3 + 4 * I4 - 2 * np.sqrt(I5))
        return np.stack([J1, J2, J3, J4, J5])
    else:
        raise ValueError("Invalid option")


# Alias for backward compatibility
compute_ent_params_from_state = compute_entanglement_params
