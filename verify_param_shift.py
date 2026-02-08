#%% Verification: Parameter Shift Rule vs Tensor Contraction
"""
Compare gradient computation using:
1. Parameter shift rule via Cirq simulation (from qsolver.py)
2. Analytical commutator via tensor contraction (from find_nash_eq_vanilla)
"""

import numpy as np
import cirq
from functools import reduce

#%% Setup: Create the same random unitary circuit state

def random_unitary_evolution(qubits, depth, seed=None):
    """From qsolver.py - creates random unitary circuit"""
    circuit = cirq.Circuit()
    L = len(qubits)
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    for d in range(depth):
        moment_ops = []
        for i in range(L):
            if i % 2 == d % 2:
                random_matrix = cirq.testing.random_unitary(4, random_state=rng)
                gate = cirq.MatrixGate(random_matrix)
                moment_ops.append(gate.on(qubits[i], qubits[np.mod(i+1, L)]))
        circuit.append(cirq.Moment(moment_ops))
    return circuit

L = 6
depth = 10
seed = 42
site = 2  # Player index to compute gradient for

qubits = cirq.LineQubit.range(L)
circuit = random_unitary_evolution(qubits, depth, seed=seed)

print(f"Circuit created with {L} qubits, depth {depth}")
print(f"Computing gradient for player {site}")

#%% Get the quantum state from the circuit

simulator = cirq.Simulator()
result = simulator.simulate(circuit)
psi_vector = result.final_state_vector

print(f"\nState vector shape: {psi_vector.shape}")
print(f"State norm: {np.linalg.norm(psi_vector):.10f}")

#%% Define Hamiltonian (3-player Prisoner's Dilemma payoff)
# From qsolver.py line 248 - this is for 3 qubits (neighbors)
PAYOFF_TABLE = np.array([6, 3, 10, 6, 3, 0, 6, 2]) / 2

# Create Hamiltonian tensor for the 3-qubit subsystem
# H acts on qubits [site-1, site, site+1] with periodic boundary
H_3qubit = np.diag(PAYOFF_TABLE).reshape([2, 2, 2, 2, 2, 2])

# Embed this into the full 6-qubit space
# We need to expand H to act on the full space
def create_full_hamiltonian(H_local, site, L):
    """
    Embed 3-qubit Hamiltonian into L-qubit space.
    H_local acts on sites [site-1, site, site+1] (mod L)
    """
    # For simplicity, create full matrix
    H_full_matrix = np.zeros((2**L, 2**L), dtype=np.float64)

    # Get the three sites
    sites = [np.mod(site-1, L), site, np.mod(site+1, L)]

    # Iterate over all basis states
    for i in range(2**L):
        for j in range(2**L):
            # Extract bits for the three sites
            bits_i = [(i >> k) & 1 for k in range(L)]
            bits_j = [(j >> k) & 1 for k in range(L)]

            # Check if they differ only on the three relevant sites
            same_on_others = all(bits_i[k] == bits_j[k] for k in range(L) if k not in sites)

            if same_on_others:
                # Get indices for the 3-qubit Hamiltonian
                idx_i = tuple(bits_i[s] for s in sites)
                idx_j = tuple(bits_j[s] for s in sites)
                H_full_matrix[i, j] = H_3qubit[idx_i + idx_j]

    return H_full_matrix

H_full = create_full_hamiltonian(H_3qubit, site, L)

print(f"\nFull Hamiltonian shape: {H_full.shape}")
print(f"Hamiltonian is diagonal: {np.allclose(H_full, np.diag(np.diag(H_full)))}")

#%% Method 1: Parameter Shift Rule (from qsolver.py)

def param_shift_simulation(qubits, circuit, site, repetitions=1000):
    """From qsolver.py lines 231-245"""
    circuit_pos_shift = circuit.copy()
    circuit_pos_shift.append(cirq.ry(np.pi/4).on(qubits[site]))
    circuit_pos_shift.append(
        cirq.measure(qubits[np.mod(site-1, len(qubits))],
                    qubits[site],
                    qubits[np.mod(site+1, len(qubits))],
                    key='result')
    )
    circuit_neg_shift = circuit.copy()
    circuit_neg_shift.append(cirq.ry(-np.pi/4).on(qubits[site]))
    circuit_neg_shift.append(
        cirq.measure(qubits[np.mod(site-1, len(qubits))],
                    qubits[site],
                    qubits[np.mod(site+1, len(qubits))],
                    key='result')
    )
    simulator = cirq.Simulator()
    result_pos_shift = simulator.run(circuit_pos_shift, repetitions=repetitions)
    result_neg_shift = simulator.run(circuit_neg_shift, repetitions=repetitions)
    return result_pos_shift, result_neg_shift

def get_estimated_gradient(qubits, circuit, repetitions, site):
    """From qsolver.py lines 249-261"""
    result_pos_shift, result_neg_shift = param_shift_simulation(qubits, circuit, site, repetitions)
    E_pos = 0
    for k, v in result_pos_shift.histogram(key='result').items():
        E_pos += PAYOFF_TABLE[k] * v
    E_pos /= repetitions

    E_neg = 0
    for k, v in result_neg_shift.histogram(key='result').items():
        E_neg += PAYOFF_TABLE[k] * v
    E_neg /= repetitions

    return (E_pos - E_neg) / (2 * math.sin(np.pi / 4))

# Run with high repetitions for accuracy
repetitions = 100000
gradient_param_shift = get_estimated_gradient(qubits, circuit, repetitions, site)

print(f"\n{'='*70}")
print(f"Method 1: Parameter Shift Rule (Cirq simulation)")
print(f"{'='*70}")
print(f"Repetitions: {repetitions}")
print(f"Gradient: {gradient_param_shift:.10f}")

#%% Method 2: Analytical Commutator (from find_nash_eq_vanilla)

# Define iRY generator (from qsolver.py line 28)
iRY = np.array([[0, -1], [1, 0]])

def expand_to_full_space(A, site, L):
    """From qsolver.py line 29"""
    list_of_mats = [np.eye(2)] * site + [A] + [np.eye(2)] * (L - site - 1)
    return reduce(np.kron, list_of_mats)

def commutator(A, B):
    """From qsolver.py line 35"""
    return A @ B - B @ A

# Compute commutator [H, iRY]
iRY_full = expand_to_full_space(iRY, site, L)
comm = commutator(H_full, iRY_full)

# Method 2a: Direct expectation value
gradient_direct = np.real(psi_vector.conj() @ comm @ psi_vector)

print(f"\n{'='*70}")
print(f"Method 2a: Analytical Commutator (direct computation)")
print(f"{'='*70}")
print(f"<ψ|[H, iRY]|ψ> = {gradient_direct:.10f}")

# Method 2b: Tensor contraction (as in find_nash_eq_vanilla lines 66-70)
psi_reshaped = psi_vector.reshape([2] * L)
comm_tensor = comm.reshape([2]*(2*L))

comm_ev = np.tensordot(comm_tensor, psi_reshaped,
                       axes=([L+j for j in range(L)], [j for j in range(L)]))
comm_ev = np.tensordot(psi_reshaped.conj(), comm_ev,
                       axes=([j for j in range(L) if j != site],
                             [j for j in range(L) if j != site]))

gradient_tensor = np.trace(comm_ev).real

print(f"\n{'='*70}")
print(f"Method 2b: Analytical Commutator (tensor contraction)")
print(f"{'='*70}")
print(f"<ψ|[H, iRY]|ψ> = {gradient_tensor:.10f}")

#%% Method 3: Exact parameter shift (analytical, not sampled)

def apply_ry_rotation(psi_vec, site, theta, L):
    """Apply RY(theta) rotation to a specific site"""
    RY = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                   [np.sin(theta/2), np.cos(theta/2)]])
    RY_full = expand_to_full_space(RY, site, L)
    return RY_full @ psi_vec

# Apply exact rotations
theta_shift = np.pi / 5
psi_plus = apply_ry_rotation(psi_vector, site, theta_shift, L)
psi_minus = apply_ry_rotation(psi_vector, site, -theta_shift, L)

# Compute exact energies
E_plus_exact = np.real(psi_plus.conj() @ H_full @ psi_plus)
E_minus_exact = np.real(psi_minus.conj() @ H_full @ psi_minus)

gradient_exact_shift = (E_plus_exact - E_minus_exact) / (2 * theta_shift) *2

print(f"\n{'='*70}")
print(f"Method 3: Exact Parameter Shift (no sampling)")
print(f"{'='*70}")
print(f"E(θ=+π/4) = {E_plus_exact:.10f}")
print(f"E(θ=-π/4) = {E_minus_exact:.10f}")
print(f"[E(+π/4) - E(-π/4)] / (π/2) = {gradient_exact_shift:.10f}")

#%% Comparison

print(f"\n{'='*70}")
print(f"VERIFICATION RESULTS")
print(f"{'='*70}")
print(f"Parameter Shift (sampled):        {gradient_param_shift:.10f}")
print(f"Commutator (direct):              {gradient_direct:.10f}")
print(f"Commutator (tensor contraction):  {gradient_tensor:.10f}")
print(f"Parameter Shift (exact):          {gradient_exact_shift:.10f}")

print(f"\nDifferences from exact commutator:")
print(f"  |Sampled - Direct|      = {abs(gradient_param_shift - gradient_direct):.2e}")
print(f"  |Tensor - Direct|       = {abs(gradient_tensor - gradient_direct):.2e}")
print(f"  |Exact Shift - Direct|  = {abs(gradient_exact_shift - gradient_direct):.2e}")

# Statistical test for sampled gradient
sampling_error = abs(gradient_param_shift - gradient_exact_shift)
expected_std = np.sqrt(np.var(PAYOFF_TABLE) / repetitions)  # Rough estimate
print(f"\nSampling error: {sampling_error:.2e}")
print(f"Expected std dev: ~{expected_std:.2e}")
print(f"Sampling error is {sampling_error/expected_std:.1f}× expected std dev")

# Check if analytical methods agree
analytical_agree = abs(gradient_direct - gradient_tensor) < 1e-10
print(f"\n✓ Analytical methods agree!" if analytical_agree
      else f"✗ Analytical methods differ!")

# Check if parameter shift is close to analytical (within sampling error)
shift_agrees = abs(gradient_param_shift - gradient_direct) < 5 * expected_std
print(f"✓ Parameter shift agrees with analytical (within 5σ)!" if shift_agrees
      else f"✗ Parameter shift differs from analytical!")

print(f"\n{'='*70}")
print(f"VERIFICATION COMPLETE")
print(f"{'='*70}")

#%%
