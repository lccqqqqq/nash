#%% DIAGNOSIS: Why COBYLA optimizer barely moves

"""
Analyzing the COBYLA optimizer configuration and regret function
"""

import numpy as np
import cirq
from scipy.optimize import minimize
from tqdm import tqdm

# ============================================================================
# CRITICAL BUG #1: compute_regrets has a useless loop
# ============================================================================

print("="*70)
print("CRITICAL BUG #1: compute_regrets function")
print("="*70)

# From qsolver.py lines 404-420:
def compute_regrets_BUGGY(qubits, circuit, repetitions, eps, num_eps=10):
    baseline = compute_payoffs(qubits, circuit, repetitions)

    total_regret = 0
    for i in range(len(qubits)):
        regret_i = 0
        for trail in range(1, num_eps+1):  # <-- This loop does nothing useful!
            circuit_pos_shift = circuit.copy()
            circuit_pos_shift.append(cirq.ry(eps).on(qubits[i]))
            circuit_neg_shift = circuit.copy()
            circuit_neg_shift.append(cirq.ry(-eps).on(qubits[i]))
            payoff_pos = compute_payoff_for_player(qubits, circuit_pos_shift, repetitions, i)
            payoff_neg = compute_payoff_for_player(qubits, circuit_neg_shift, repetitions, i)
            regret_i = max(0, payoff_pos - baseline[i], payoff_neg - baseline[i])
            # ^^ BUG: regret_i is OVERWRITTEN each iteration, not accumulated!

        total_regret += regret_i
    return total_regret

print("""
PROBLEM: The loop at line 410 runs num_eps=10 times but:
1. Always uses the SAME epsilon value (doesn't vary)
2. OVERWRITES regret_i each iteration instead of accumulating
3. Only the LAST iteration's result is used

This makes the loop completely useless and wastes 10x computation!

WHAT IT PROBABLY SHOULD BE:
Either remove the loop entirely, OR accumulate/average results for noise reduction.
""")

# ============================================================================
# CRITICAL BUG #2: Noise in objective function
# ============================================================================

print("\n" + "="*70)
print("CRITICAL BUG #2: Objective function is too noisy")
print("="*70)

# Setup
L = 4
qubits = cirq.LineQubit.range(L)
PAYOFF_TABLE = np.array([6, 3, 10, 6, 3, 0, 6, 2]) / 2

def random_unitary_evolution(qubits, depth, seed=None):
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
    circuit_measure_player.append(cirq.measure(
        qubits[np.mod(player-1, len(qubits))],
        qubits[player],
        qubits[np.mod(player+1, len(qubits))],
        key='result'))
    result = simulator.run(circuit_measure_player, repetitions=repetitions)
    result_counts = result.histogram(key='result')
    payoff = 0
    for key, value in result_counts.items():
        payoff += PAYOFF_TABLE[key] * value
    return payoff / repetitions

# Fixed version
def compute_regrets_FIXED(qubits, circuit, repetitions, eps):
    baseline = compute_payoffs(qubits, circuit, repetitions)

    total_regret = 0
    for i in range(len(qubits)):
        circuit_pos_shift = circuit.copy()
        circuit_pos_shift.append(cirq.ry(eps).on(qubits[i]))
        circuit_neg_shift = circuit.copy()
        circuit_neg_shift.append(cirq.ry(-eps).on(qubits[i]))
        payoff_pos = compute_payoff_for_player(qubits, circuit_pos_shift, repetitions, i)
        payoff_neg = compute_payoff_for_player(qubits, circuit_neg_shift, repetitions, i)
        regret_i = max(0, payoff_pos - baseline[i], payoff_neg - baseline[i])
        total_regret += regret_i
    return total_regret

circuit = random_unitary_evolution(qubits, depth=10, seed=42)

# Test noise level at a fixed parameter
params_test = np.pi/3 * np.ones(L)
test_circuit = circuit.copy()
for i in range(L):
    test_circuit.append(cirq.ry(params_test[i]).on(qubits[i]))

print(f"\nTesting objective function noise at params={params_test[0]:.3f}...")
print(f"Running 20 evaluations with 10000 repetitions each...")

regrets = []
for trial in range(20):
    regret = compute_regrets_FIXED(qubits, test_circuit, repetitions=10000, eps=0.1)
    regrets.append(regret)

regrets = np.array(regrets)
print(f"\nMean regret: {np.mean(regrets):.6f}")
print(f"Std dev:     {np.std(regrets):.6f}")
print(f"Coefficient of variation: {np.std(regrets)/np.mean(regrets)*100:.1f}%")

if np.std(regrets)/np.mean(regrets) > 0.5:
    print("\n⚠️  WARNING: Noise is >50% of signal! COBYLA will struggle!")
else:
    print("\n✓ Noise level seems acceptable")

# ============================================================================
# ISSUE #3: COBYLA parameter settings
# ============================================================================

print("\n" + "="*70)
print("ISSUE #3: COBYLA parameter analysis")
print("="*70)

print("""
Current settings (line 456):
- rhobeg: 0.8       (initial trust region size)
- maxiter: 1000     (max iterations)
- tol: 1e-5         (convergence tolerance)
- x0: π/3 * ones(L) (starting point)
- eps: 0.1          (deviation size for regret)

PROBLEMS:

1. rhobeg=0.8 might be TOO LARGE
   - Parameter space is [0, 2π] ≈ [0, 6.28]
   - Starting step of 0.8 is ~13% of the range
   - For a noisy objective, this might overshoot
   - Recommendation: Try 0.3-0.5

2. tol=1e-5 might be TOO TIGHT for noisy objective
   - With high noise, optimizer might declare convergence
     when objective changes are < tol due to noise, not true convergence
   - Recommendation: Try 1e-3 or 1e-4

3. Single starting point
   - COBYLA is local optimizer, can get stuck
   - Recommendation: Multi-start optimization

4. eps=0.1 is FIXED
   - Only checks one deviation size
   - True Nash requires checking ALL deviations
   - Recommendation: Use smaller eps (0.05) or multiple eps values
""")

# ============================================================================
# DIAGNOSTIC: Check if optimizer is actually improving anything
# ============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC: Is the optimizer actually working?")
print("="*70)

def regret_function(params, qubits, rcs_circuit, repetitions=10000, eps=0.1):
    total_circuit = rcs_circuit.copy()
    for i in range(len(params)):
        total_circuit.append(cirq.ry(params[i]).on(qubits[i]))
    regret = compute_regrets_FIXED(qubits, total_circuit, repetitions, eps)
    return regret

# Test at a few different points
test_points = [
    np.zeros(L),
    np.pi/4 * np.ones(L),
    np.pi/3 * np.ones(L),
    np.pi/2 * np.ones(L),
    np.random.uniform(0, 2*np.pi, L)
]

print("\nEvaluating regret at different parameter values:")
print("(Lower regret = closer to Nash equilibrium)\n")

for i, point in enumerate(test_points):
    regret = regret_function(point, qubits, circuit, 10000, 0.1)
    print(f"Point {i}: params={point[:2]}... → regret={regret:.6f}")

# ============================================================================
# RECOMMENDED FIX
# ============================================================================

print("\n" + "="*70)
print("RECOMMENDED FIXES")
print("="*70)

print("""
1. FIX compute_regrets: Remove useless loop

def compute_regrets(qubits, circuit, repetitions, eps):
    baseline = compute_payoffs(qubits, circuit, repetitions)
    total_regret = 0
    for i in range(len(qubits)):
        circuit_pos_shift = circuit.copy()
        circuit_pos_shift.append(cirq.ry(eps).on(qubits[i]))
        circuit_neg_shift = circuit.copy()
        circuit_neg_shift.append(cirq.ry(-eps).on(qubits[i]))
        payoff_pos = compute_payoff_for_player(qubits, circuit_pos_shift, repetitions, i)
        payoff_neg = compute_payoff_for_player(qubits, circuit_neg_shift, repetitions, i)
        total_regret += max(0, payoff_pos - baseline[i], payoff_neg - baseline[i])
    return total_regret

2. ADJUST COBYLA settings:

res = minimize(
    regret_function,
    x0=np.pi/3 * np.ones(L),
    args=(qubits, circuit, 20000, 0.05),  # More samples, smaller eps
    method='COBYLA',
    options={
        'rhobeg': 0.4,      # Smaller initial step
        'maxiter': 2000,    # More iterations
        'tol': 1e-3,        # Looser tolerance for noisy objective
        'disp': True        # Show progress
    },
    constraints=cons
)

3. USE MULTI-START:

best_result = None
best_regret = np.inf

for trial in range(10):
    x0 = np.random.uniform(0, 2*np.pi, L)
    res = minimize(regret_function, x0=x0, ...)
    if res.fun < best_regret:
        best_regret = res.fun
        best_result = res
        print(f"New best: regret={res.fun:.6f}")

4. ALTERNATIVE: Use your exact gradient-based solver instead!
   - It's much more reliable
   - COBYLA is struggling because objective is noisy
   - Your find_nash_eq_vanilla should work better
""")
