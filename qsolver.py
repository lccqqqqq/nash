#%% Imports
import numpy as np
from tqdm import tqdm
from src.game import get_default_cyclic_players
from experiments.compute_eq import rand_unitary_circuit
from src.mps_utils import from_comp_basis, to_comp_basis, apply_unitary
from src.solver import kick_with_u, compute_exploitability
from functools import reduce
from utils.misc import plot_energy_and_exploitability

# %% Set up game
L = 4
H = get_default_cyclic_players(L=L, dtype=np.float64)

def init_mps_state_with_rcs(L, depth=10, seed=None):
    state = np.zeros([2]*L)
    state[(0,)*L] = 1.0
    state = rand_unitary_circuit(state, depth=depth, seed=seed)
    state_mps = from_comp_basis(state.flatten(), L=L)
    state_mps = kick_with_u(state_mps)
    return state_mps

state_mps = init_mps_state_with_rcs(L, depth=10, seed=42)
print(state_mps)

#%% Write Vanilla Gradient solver
import math
iRY = np.array([[0, -1], [1, 0]])
def expand_to_full_space(A, site, L):
    list_of_mats = [np.eye(2)] * site + [A] + [np.eye(2)] * (L - site - 1)
    return reduce(np.kron, list_of_mats)



def commutator(A, B):
    return A @ B - B @ A

def find_nash_eq_vanilla(
    Psi: list[np.ndarray] | np.ndarray, # allowing for both MPS and computational basis input
    H: list[np.ndarray],
    max_iter: int = 1000,
    alpha: float = 0.01,
    convergence_threshold: float = 1e-7,
    expl_threshold: float = 5e-4,
    use_tqdm: bool = False,
    expl_check_interval: int = 50,
    return_history: bool = False,
    expl_maxiter: int = 60,
    expl_seed: int = 42,
    real_strategies: bool = True,
):
    Es = []
    psi_list = [] if return_history else None
    Psi_list = [] if return_history else None
    local_converged = False
    global_converged = False
    expl_list = []
    for n in tqdm(range(max_iter), disable=not use_tqdm):
        if isinstance(Psi, list):
            psi = to_comp_basis(Psi).reshape([2] * L)
        else:
            psi = Psi.reshape([2] * L)
        RY_coefs = []
        E = []
        for i in range(L):
            comm = commutator(H[i].reshape(2**L, 2**L), expand_to_full_space(iRY, i, L)).reshape([2]*(2*L))
            comm_ev = np.tensordot(comm, psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            comm_ev = np.tensordot(psi.conj(), comm_ev, axes=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))

            RY_coefs.append(np.trace(comm_ev).real)
            
            # Also compute the energy at this step
            dE = np.tensordot(H[i], psi, axes=([L+j for j in range(L)], [j for j in range(L)]))
            dE = np.tensordot(psi.conj(), dE, axes=([j for j in range(L) if j != i], [j for j in range(L) if j != i]))
            E.append(np.trace(dE).real)

        Es.append(np.array(E))
        if return_history:
            psi_list.append(psi)
            Psi_list.append(Psi)
        for i in range(L):
            # Here the convention is made sure to be the same as in `apply_u`
            RY_site_i = np.eye(2) * math.cos(RY_coefs[i] * alpha) - iRY * math.sin(RY_coefs[i] * alpha)
            if isinstance(Psi, list):
                Psi[i] = apply_unitary(RY_site_i.T, Psi[i])
            else:
                Psi = reduce(np.kron, RY_site_i) @ Psi
        
        if n > 2 and not local_converged:
            local_converged = sum([abs(E[i] - Es[-2][i]) for i in range(L)]) < convergence_threshold
            if local_converged:
                print(f"Converged to Nash state at iteration {n}")


        if n % expl_check_interval == 0:
            expl = [compute_exploitability(psi, H, i, maxiter=expl_maxiter, seed=expl_seed, real_strategies=real_strategies) for i in range(L)]
            expl_list.append(expl)
            if sum(expl) < expl_threshold:
                global_converged = True
                break


    result = {
        'nash_state': local_converged,
        'nash_equilibrium': global_converged,
        'energy': np.stack(Es) if return_history else Es[-1],  # Only return final energy if not tracking history
        'state': psi_list if return_history else psi,
        'state_': Psi_list if return_history else Psi,
        'num_iters': n,
        'expl': np.array(expl_list) if return_history else expl_list[-1],
    }

    return result

#%% Run Solver
from src.solver import find_nash_eq1
L = 6
H = get_default_cyclic_players(L=L, dtype=np.float64)
Psi = init_mps_state_with_rcs(L, depth=10, seed=43)

result = find_nash_eq_vanilla(
    Psi=Psi,
    H=H,
    max_iter=3000,
    alpha=0.06,
    convergence_threshold=1e-7,
    expl_threshold=1e-7,
    expl_check_interval=50,
    expl_maxiter=60,
    expl_seed=42,
    real_strategies=True,
    return_history=True,
    use_tqdm=True
)
plot_energy_and_exploitability(result)


#%%
Psi = init_mps_state_with_rcs(L, depth=10, seed=42)
result_1 = find_nash_eq1(Psi=Psi, H=H, max_iter=1000, alpha=0.3, convergence_threshold=1e-7, expl_threshold=5e-4, expl_check_interval=50, expl_maxiter=60, expl_seed=42, real_strategies=True, return_history=True)
plot_energy_and_exploitability(result_1)

#%% Demonstration Circuit Implementation

import cirq as cq
L = 6
qubits = cq.LineQubit.range(L)
circuit = cq.Circuit(
    [cq.H(qubits[i]) for i in range(L)],
    [cq.CNOT(qubits[i], qubits[i+1]) for i in range(L-1)],
    [cq.CNOT(qubits[i], qubits[i+2]) for i in range(L-2)],
    [cq.CNOT(qubits[i], qubits[i+3]) for i in range(L-3)],
    [cq.CNOT(qubits[i], qubits[i+4]) for i in range(L-4)],
    [cq.CNOT(qubits[i], qubits[i+5]) for i in range(L-5)],
    cq.measure(qubits[0], qubits[1], key='result')
)
simulator = cq.Simulator()
result = simulator.run(circuit, repetitions=100000)
import matplotlib.pyplot as plt

hist = result.histogram(key='result')
plt.bar(hist.keys(), hist.values())
plt.xlabel('Measurement outcome')
plt.ylabel('Counts')
plt.title('Histogram of Measurement Outcomes')
plt.show()

#%% Counting number of gates
import cirq
import collections

# 1. Define your abstract circuit
qubits = cirq.LineQubit.range(4)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.X(qubits[2])**0.5,
    cirq.CNOT(qubits[2], qubits[3])
)

# 2. Transpile to a native gate set (e.g., Google's Sycamore)
from cirq_google.transformers import SycamoreTargetGateset
transpiled_circuit = cirq.optimize_for_target_gateset(
    circuit, 
    gateset=SycamoreTargetGateset()
)

# 3. Count the gates
gate_counts = collections.Counter(type(op.gate) for op in transpiled_circuit.all_operations())

print(f"Native Gate Counts: {gate_counts}")
print(f"Total Native Gates: {len(list(transpiled_circuit.all_operations()))}")

#%% Random Unitary Evolution Circuit
import numpy as np

def random_unitary_evolution(qubits, depth, seed=None):
    circuit = cirq.Circuit()
    L = len(qubits)
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState() 
    for d in range(depth):
        moment_ops = []
        for i in range(L):
            if i % 2 == d % 2:
                # Generate a random 4x4 unitary (Haar random)
                random_matrix = cirq.testing.random_unitary(4, random_state=rng)
                gate = cirq.MatrixGate(random_matrix)
                
                moment_ops.append(gate.on(qubits[i], qubits[np.mod(i+1, L)]))
        circuit.append(cirq.Moment(moment_ops))
        
    return circuit

# Example Usage
L = 16
layers = 10
q_list = cirq.LineQubit.range(L)
random_circuit = random_unitary_evolution(q_list, layers, seed=42)
transpiler = SycamoreTargetGateset()
decomposed_circuit = cirq.optimize_for_target_gateset(random_circuit, gateset=transpiler)
gate_counts = collections.Counter(type(op.gate) for op in decomposed_circuit.all_operations())
print(f"Native Gate Counts: {gate_counts}")
print(f"Total Native Gates: {len(list(decomposed_circuit.all_operations()))}")

#%% Retrieving gradient of the circuit
import cirq
L = 4
qubits = cirq.LineQubit.range(L)
depth = 10
circuit = random_unitary_evolution(qubits, depth, seed=42)
# print(circuit)
def param_shift_simulation(qubits, circuit, site, repetitions=1000):
    circuit_pos_shift = circuit.copy()
    circuit_pos_shift.append(cirq.ry(np.pi/4).on(qubits[site]))
    circuit_pos_shift.append(
        cirq.measure(qubits[np.mod(site-1, len(qubits))], qubits[site], qubits[np.mod(site+1, len(qubits))], key='result')
    )
    circuit_neg_shift = circuit.copy()
    circuit_neg_shift.append(cirq.ry(-np.pi/4).on(qubits[site]))
    circuit_neg_shift.append(
        cirq.measure(qubits[np.mod(site-1, len(qubits))], qubits[site], qubits[np.mod(site+1, len(qubits))], key='result')
    )
    simulator = cirq.Simulator()
    result_pos_shift = simulator.run(circuit_pos_shift, repetitions=repetitions)
    result_neg_shift = simulator.run(circuit_neg_shift, repetitions=repetitions)
    return result_pos_shift, result_neg_shift

import math
PAYOFF_TABLE = np.array([6, 3, 10, 6, 3, 0, 6, 2]) / 2
def get_estimated_gradient(qubits, circuit, repetitions, site):
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

def compute_payoffs(qubits, circuit, repetitions):
    simulator = cirq.Simulator()
    circuit_measure_all = circuit.copy()
    circuit_measure_all.append(cirq.measure(*qubits, key='result'))
    result = simulator.run(circuit_measure_all, repetitions=repetitions)
    result_counts = result.histogram(key='result')
    result_counts = {format(key, '0{}b'.format(len(qubits))): value for key, value in result_counts.items()}

    payoffs = np.zeros(len(qubits))
    for key, value in result_counts.items():
        for i in range(len(qubits)):
            payoffs[i] += PAYOFF_TABLE[
                int("".join([key[np.mod(i-1, len(qubits))],
                             key[i], key[np.mod(i+1, len(qubits))]]), 2)
            ] * value
    return payoffs / repetitions

def compute_payoff_for_player(qubits, circuit, repetitions, player):
    simulator = cirq.Simulator()
    circuit_measure_player = circuit.copy()
    circuit_measure_player.append(cirq.measure([qubits[np.mod(player-1, len(qubits))], qubits[player], qubits[np.mod(player+1, len(qubits))]], key='result'))
    result = simulator.run(circuit_measure_player, repetitions=repetitions)
    result_counts = result.histogram(key='result')
    payoff = 0
    for key, value in result_counts.items():
        payoff += PAYOFF_TABLE[key] * value
    return payoff / repetitions

def compute_exact_payoffs(qubits, circuit):
    H = get_default_cyclic_players(L=len(qubits), dtype=np.float64)
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    psi_vector = result.final_state_vector
    payoffs = np.zeros(len(qubits))
    for i in range(len(qubits)):
        h = H[i].reshape(2**len(qubits), 2**len(qubits))
        payoff = psi_vector.conj() @ h @ psi_vector
        payoffs[i] = payoff.real
    return payoffs


# Example usage:
# hist = result.histogram(key='result')
# binary_hist = histogram_keys_to_binary(hist, L)
# print(binary_hist)

y_grad = get_estimated_gradient(qubits, circuit, repetitions=1000, site=2)
print(y_grad)

payoff_results = compute_payoffs(qubits, circuit, repetitions=1000)
print(payoff_results)
exact_payoffs = compute_exact_payoffs(qubits, circuit)
print(exact_payoffs)

#%% Statistics of payoff estimation

repetitions = 1000
num_trails = 1000
payoff_results = []
exact_payoffs = compute_exact_payoffs(qubits, circuit)
for i in tqdm(range(num_trails)):
    payoff_results.append(compute_payoffs(qubits, circuit, repetitions=repetitions))

payoff_results = np.array(payoff_results)
# Plot payoff histograms for each player, using the sampled payoff results

import matplotlib.pyplot as plt

num_players = payoff_results.shape[1] if payoff_results.ndim > 1 else len(payoff_results[0])

# Use a 2x2 layout for histograms (up to 4 players; adjust as needed)
import math

ncols = 2
nrows = math.ceil(num_players / 2)
plt.figure(figsize=(12, 4 * nrows))
for i in range(num_players):
    plt.subplot(nrows, ncols, i + 1)
    plt.hist(payoff_results[:, i], bins=30, alpha=0.7, color='royalblue', edgecolor='black')
    plt.axvline(exact_payoffs[i], color='crimson', linestyle='--', linewidth=2, label='Exact Payoff')
    plt.xlabel(f'Player {i} Payoff', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title(f'Histogram of Estimated Payoff for Player {i}', fontsize=15)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



#%% Compare with actual gradient

simulator = cirq.Simulator()
result = simulator.simulate(circuit)
psi_vector = result.final_state_vector.reshape([2] * L)
site = 2
H = get_default_cyclic_players(L=L, dtype=np.float64)
comm = commutator(H[site].reshape(2**L, 2**L), expand_to_full_space(iRY, site, L)).reshape([2]*(2*L))
comm_ev = np.tensordot(comm, psi_vector, axes=([L+j for j in range(L)], [j for j in range(L)]))
comm_ev = np.tensordot(psi_vector.conj(), comm_ev, axes=([j for j in range(L) if j != site], [j for j in range(L) if j != site]))

print(np.trace(comm_ev).real / 2)
#%% Gather some statistics on gradient estimation
import matplotlib.pyplot as plt
from tqdm import tqdm
num_trails = 1000
repetitions = 10000
grad_list = []
for trail in tqdm(range(num_trails)):
    y_grad = get_estimated_gradient(qubits, circuit, repetitions=repetitions, site=2)
    grad_list.append(y_grad)

#%% Plot the Histogram of Gradient Estimates
plt.figure(figsize=(8,6))
n, bins, patches = plt.hist(grad_list, bins=20, density=True, alpha=0.75, color='cornflowerblue', edgecolor='black')
plt.axvline(x=np.trace(comm_ev).real / 2, color='crimson', linewidth=2, linestyle='--', label='Exact Gradient')
plt.xlabel('Estimated Gradient (from sampling the parameter shift rule)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('Distribution of Estimated Gradients\nvs Exact Gradient', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# Add text box with repetitions in the plot
reps_text = f'Repetitions per run: {repetitions}; Number of qubits: {L}'
plt.gca().text(0.98, 0.95, reps_text, fontsize=12, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
               transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

#%% Needed 10^5 evaluations per gradient... very costly!
result_pos_shift, result_neg_shift = param_shift_simulation(qubits, circuit, site=2, repetitions=100000)
print(result_pos_shift.histogram(key='result').keys())
print(result_neg_shift.histogram(key='result').values())


#%% Using COBYLA for gradient-free optimization
from scipy.optimize import minimize

def compute_regrets(qubits, circuit, repetitions, eps, num_eps=3):
    baseline = compute_payoffs(qubits, circuit, repetitions)
    
    total_regret = 0
    for i in range(len(qubits)):
        regret_i = 0
        for trail in range(1, num_eps+1):
            circuit_pos_shift = circuit.copy()
            circuit_pos_shift.append(cirq.ry(trail*eps).on(qubits[i]))
            circuit_neg_shift = circuit.copy()
            circuit_neg_shift.append(cirq.ry(-trail*eps).on(qubits[i]))
            payoff_pos = compute_payoff_for_player(qubits, circuit_pos_shift, repetitions, i)
            payoff_neg = compute_payoff_for_player(qubits, circuit_neg_shift, repetitions, i)
            regret_i = max(regret_i, payoff_pos - baseline[i], payoff_neg - baseline[i])

        total_regret += regret_i
    return total_regret

regret_list = []
num_trails = 100
for trail in tqdm(range(num_trails)):
    total_regret = compute_regrets(qubits, circuit, repetitions=1000, eps=0.1)
    regret_list.append(total_regret)

regret_list = np.array(regret_list)
plt.hist(regret_list, bins=30, alpha=0.7, color='royalblue', edgecolor='black')
plt.xlabel('Total Regret', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.title('Histogram of Total Regret', fontsize=15)
plt.show()

#%% Minimizing regret using COBYLA
from scipy.optimize import minimize

def regret_function(params, qubits, rcs_circuit, repetitions=1000, eps=0.1):
    """
    The parameters are the Ry angles for each player's qubit.
    """
    total_circuit = rcs_circuit.copy()
    for i in range(len(params)):
        total_circuit.append(cirq.ry(params[i]).on(qubits[i]))
    
    regret = compute_regrets(qubits, total_circuit, repetitions, eps)
    return regret


L = 4
qubits = cirq.LineQubit.range(L)
circuit = random_unitary_evolution(qubits, depth=10, seed=42)
cons = []
for i in range(L):
    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i]})              # x[i] >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=i: 2*np.pi - x[i]})    # x[i] <= 2pi
res = minimize(regret_function, x0=np.random.uniform(0, 2*np.pi, L), args=(qubits, circuit, 10000, 0.3), method='COBYLA', options={'rhobeg': 0.3, 'maxiter': 1000, 'tol': 0.05, 'disp': 1}, constraints=cons)
print(res.x)
print(res.fun)
#%% Computing the optimal payoff with minimized regret
optimal_circuit = circuit.copy()
original_payoffs = compute_exact_payoffs(qubits, circuit)
for i in range(L):  
    optimal_circuit.append(cirq.ry(res.x[i]).on(qubits[i]))
optimal_payoffs = compute_payoffs(qubits, optimal_circuit, repetitions=1000)
print(original_payoffs)
print(optimal_payoffs)
exact_payoffs = compute_exact_payoffs(qubits, optimal_circuit)
print(exact_payoffs)

#%% Compare with the exact solver
from src.solver import find_nash_eq1
from utils.misc import plot_energy_and_exploitability
L = 4
H = get_default_cyclic_players(L=L, dtype=np.float64)
simulator = cirq.Simulator()
result = simulator.simulate(optimal_circuit)
psi_vector = result.final_state_vector
Psi = from_comp_basis(psi_vector, L=L)

exact_result = find_nash_eq1(Psi=Psi, H=H, max_iter=1000, alpha=0.1, convergence_threshold=1e-7, expl_threshold=5e-4, expl_check_interval=50, expl_maxiter=60, expl_seed=42, real_strategies=True, return_history=True, use_tqdm=True)
plot_energy_and_exploitability(exact_result)
#%% Report
"""
The shot noises seems to dominate for moderate-sized qubits (L=4) when computing the updates.

To use COBYLA we need to compute a single regret function as the cost landscape the optimizer seeks to minimize. The regret evaluations are quite noisy because each function evaluation for the regret takes the maximum of the payoff differences between the current strategy and the shifted strategies. The shot noise in payoff evaluations is often comparable with the payoff difference, so that the SNR is low.

On the other hand the gradient-based solver also appears noisy (from the evaluations using the parameter shift rule).

Once these are sorted out when we can use COBYLA to find locate the true Nash Equilibrium, the resource estimation appears straightforward... (using the functions native to cirq)
"""
