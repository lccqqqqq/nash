#%%
# Generating and running a Python simulation of scrambling dynamics for n-qubit systems
# - Brickwork of random two-qubit Haar gates
# - Track bipartite entanglement entropy of the state (initial |0...0>)
# - Track operator spreading of a local Pauli (Z on qubit 0) by decomposing into Pauli strings
# - Produce two plots: entanglement entropy vs layer, and operator weight heatmap (layers x qubits)
# This code is self-contained and runs without internet.

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm, eig
from scipy.linalg import qr

# Helper: single-qubit Pauli matrices and identity
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
paulis = [I, X, Y, Z]
pauli_labels = ['I','X','Y','Z']

def haar_random_unitary(dim):
    # Generate Haar random unitary via QR of complex gaussian matrix
    z = (np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)) / np.sqrt(2.0)
    q, r = qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph

def kron_n(ops):
    # Kronecker product of list ops in given order
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out

def embed_two_qubit_gate(n, gate, pair):
    # Embed a 4x4 two-qubit gate into n-qubit Hilbert space at qubit indices pair (i,j) where i<j
    assert gate.shape == (4,4)
    i, j = pair
    ops = []
    for q in range(n):
        if q == i or q == j:
            # placeholder, we'll insert combined after loop
            ops.append(None)
        else:
            ops.append(I)
    # Now build by inserting gate on positions i and j using kron
    # Construct list of factors where at positions i and j we will place single-qubit factors forming the 4x4 gate
    # Easier method: start with identity and tensor in correct order using swap if needed.
    # We'll build full matrix by summing basis |a b><c d| mapped to computational basis positions.
    dim = 2**n
    full = np.zeros((dim, dim), dtype=complex)
    # iterate over computational basis states
    for a in range(2**n):
        # binary representation list of bits
        bits = [(a >> (n-1-k)) & 1 for k in range(n)]
        for b in range(2**n):
            bits2 = [(b >> (n-1-k)) & 1 for k in range(n)]
            # Only positions other than i,j must match
            ok = True
            for q in range(n):
                if q!=i and q!=j and bits[q] != bits2[q]:
                    ok = False
                    break
            if not ok:
                continue
            # Now rows/cols indices for the two-qubit subspace
            r_idx = (bits[i] << 1) | bits[j]
            c_idx = (bits2[i] << 1) | bits2[j]
            full[a,b] = gate[r_idx, c_idx]
    return full

def build_brickwork_layer(n):
    # Build a layer of random two-qubit Haar gates in brickwork (even pairs then odd pairs)
    # We'll return two sublayers (even, odd) to apply sequentially for a full layer
    even_gates = []
    odd_gates = []
    for i in range(0, n-1, 2):
        U = haar_random_unitary(4)
        even_gates.append((i, i+1, U))
    for i in range(1, n-1, 2):
        U = haar_random_unitary(4)
        odd_gates.append((i, i+1, U))
    # Create full unitaries for the sublayers (product of embedded gates)
    U_even = np.eye(2**n, dtype=complex)
    for i, j, U in even_gates:
        U_even = embed_two_qubit_gate(n, U, (i,j)) @ U_even
    U_odd = np.eye(2**n, dtype=complex)
    for i, j, U in odd_gates:
        U_odd = embed_two_qubit_gate(n, U, (i,j)) @ U_odd
    return U_even @ U_odd  # single full layer (even then odd)

def entropy_half_chain(state, n):
    # compute von Neumann entropy of left half (first n//2 qubits)
    m = n//2
    dimL = 2**m
    dimR = 2**(n-m)
    psi = state.reshape((dimL, dimR))
    # compute Schmidt singular values
    s = svd(psi, compute_uv=False)
    p = (s**2) / np.sum(s**2)
    # filter tiny probabilities
    p = p[p>1e-12]
    return -np.sum(p * np.log2(p))

def pauli_string(n, label_indices):
    # label_indices: list of ints 0..3 per qubit to choose I,X,Y,Z
    ops = [paulis[idx] for idx in label_indices]
    return kron_n(ops)

def all_pauli_strings(n):
    # returns list of (labels, matrix)
    for labels in itertools.product(range(4), repeat=n):
        yield labels, pauli_string(n, labels)

def pauli_decomposition(operator, n):
    # compute coefficients c_P = (1/2^n) Tr[P^\dagger O] for Pauli basis
    dim = 2**n
    coeffs = {}
    for labels in itertools.product(range(4), repeat=n):
        P = pauli_string(n, labels)
        c = np.trace(P.conj().T @ operator) / dim
        coeffs[labels] = c
    return coeffs

def weight_per_qubit_from_coeffs(coeffs, n):
    # For each qubit q, compute total weight sum |c_P|^2 over Pauli strings where P acts non-identity on q
    weights = np.zeros(n)
    for labels, c in coeffs.items():
        prob = np.abs(c)**2
        for q in range(n):
            if labels[q] != 0:
                weights[q] += prob
    return weights

# Simulation parameters (you can adjust)
n = 6  # number of qubits (keep moderate for full matrix ops)
layers = 12  # number of brickwork layers
np.random.seed(42)

# initial state |0..0>
psi = np.zeros(2**n, dtype=complex)
psi[0] = 1.0

entropies = []
operator_weights = []  # list of arrays size n per step

# initial operator: Z on qubit 0
# build Z_0 operator
labels_init = tuple([0]*n)
# replace qubit 0 with Z label 3
labels_list = list(labels_init)
labels_list[0] = 3  # Z
labels_init = tuple(labels_list)
O = pauli_string(n, labels_init)  # initial operator matrix

# track initial operator coefficients
coeffs = pauli_decomposition(O, n)
operator_weights.append(weight_per_qubit_from_coeffs(coeffs, n))
entropies.append(entropy_half_chain(psi, n))

# evolve through layers
U_total = np.eye(2**n, dtype=complex)
for t in range(layers):
    U_layer = build_brickwork_layer(n)
    # Schroedinger evolution for state
    psi = U_layer @ psi
    # Heisenberg evolution for operator: O -> U^\dagger O U (note: full unitary is U_layer)
    O = U_layer.conj().T @ O @ U_layer
    coeffs = pauli_decomposition(O, n)
    operator_weights.append(weight_per_qubit_from_coeffs(coeffs, n))
    entropies.append(entropy_half_chain(psi, n))

operator_weights = np.array(operator_weights)  # shape (layers+1, n)
entropies = np.array(entropies)  # length layers+1

# Display results: entanglement entropy vs layer
plt.figure(figsize=(6,4))
plt.plot(range(layers+1), entropies, marker='o')
plt.xlabel('Layer')
plt.ylabel('Entanglement entropy (bits) of left half')
plt.title(f'Entanglement growth (n={n})')
plt.grid(True)
plt.show()

# Operator spreading heatmap (layers x qubits)
plt.figure(figsize=(7,4))
plt.imshow(operator_weights, aspect='auto', interpolation='nearest')
plt.colorbar(label='Pauli weight (sum |c|^2)')
plt.xlabel('Qubit index')
plt.ylabel('Layer')
plt.title('Operator spreading of Z_0 (Pauli weight per qubit)')
plt.show()

# Print small summary
print("Final entanglement entropy:", entropies[-1])
print("Operator weight (last layer) per qubit:", operator_weights[-1])

# Provide the operator weight table for inspection
import pandas as pd
df = pd.DataFrame(operator_weights, columns=[f"q{j}" for j in range(n)])
df.index.name = "layer"
# Display dataframe to user
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("Operator weights per qubit by layer", df)

# Save matrices for user to inspect if desired
np.savez('/mnt/data/scrambling_simulation_n6_layers12.npz',
         operator_weights=operator_weights, entropies=entropies, final_state=psi)
print("[Saved simulation data to] /mnt/data/scrambling_simulation_n6_layers12.npz")
