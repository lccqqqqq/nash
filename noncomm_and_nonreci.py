#%% Interplay between non-commutativity and non-reciprocity in the QPD game
# Construct the Hamiltonian c.f. spin chains
%load_ext autoreload
%autoreload 2
from src.solver import PAULIS as sigma
from src.game import get_default_cyclic_players
import numpy as np
from src.mps_utils import get_product_state, to_comp_basis, from_comp_basis
from src.solver import find_nash_eq1, kick_with_u
from copy import deepcopy
from utils.misc import plot_energy_and_exploitability, get_near_identity_unitary, get_random_two_qubit_unitary

def get_TI_ising(g_L, g_R):
    H_comm = np.kron(sigma[2], sigma[2])
    V_L = g_L * np.kron(sigma[0], np.eye(2))
    V_R = g_R * np.kron(np.eye(2), sigma[0])
    Hs = [H_comm + V_L, H_comm + V_R]
    return Hs

def get_asym_heisenberg():
    a = np.random.randn(3, 2)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    xx = np.kron(sigma[0], sigma[0])
    yy = np.kron(sigma[1], sigma[1])
    zz = np.kron(sigma[2], sigma[2])
    H_L = a[0, 0] * xx + a[1, 0] * yy + a[2, 0] * zz
    H_R = a[0, 1] * xx + a[1, 1] * yy + a[2, 1] * zz
    Hs = [H_L, H_R]
    return Hs

#%% Initialize the state
L = 3
Psi = get_product_state(L=L, state_per_site=[0]*L, dtype=np.float64)
Psi = kick_with_u(Psi)
def perturb_state(Psi, epsilon, L, seed=42):
    rand_u = get_near_identity_unitary(n_qubits=L, epsilon=epsilon, dtype=np.complex128, seed=seed)
    psi = rand_u @ to_comp_basis(Psi)
    Psi = from_comp_basis(psi, L=L)
    return Psi

# Perturbation with random two-qubit unitary
rand_2q_u = get_random_two_qubit_unitary(seed=42)
psi = np.kron(rand_2q_u, np.eye(2)) @ to_comp_basis(Psi)
Psi = from_comp_basis(psi, L=L)
Psi = kick_with_u(Psi)

# Psi = perturb_state(Psi, epsilon=0.3, L=L, seed=42)
Hs = get_asym_heisenberg()
H = get_default_cyclic_players(L=L, Hs=Hs, dtype=np.float64)

init_state = deepcopy(Psi)
result = find_nash_eq1(init_state, H, max_iter=1000, alpha=0.04, expl_threshold=1e-8, expl_check_interval=100, expl_maxiter=300, real_strategies=False, return_history=True, use_tqdm=True)
plot_energy_and_exploitability(result, relative_to_init=False)
print(result['energy'][-1])
print(init_state)

#%% Two player games with non-commutativity and non-reciprocity parts

