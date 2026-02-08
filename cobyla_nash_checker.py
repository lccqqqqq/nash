# REVIEW OF COBYLA NASH EQUILIBRIUM FINDER
# Issues found and fixes

import cirq
import numpy as np
from scipy.optimize import minimize

# --- ISSUES IDENTIFIED ---

"""
1. **CRITICAL BUG**: results indexing is wrong
   - results[0][0] should be results[0]
   - run_batch returns list of Result objects, not nested lists

2. **CRITICAL BUG**: Constraints only apply to Alice (player 0), not Bob
   - Need constraints for both theta_A and theta_B

3. **POTENTIAL ISSUE**: Noise buffer calculation
   - Using 1/sqrt(shots) is too simplistic
   - Should be based on actual variance of the payoff estimator
   - Current approach: threshold at ~1 standard error (68% confidence)
   - For more confidence, use 2/sqrt(shots) or 3/sqrt(shots)

4. **CONCEPTUAL ISSUE**: Fixed epsilon deviation
   - Only checks local deviations of size epsilon
   - Might miss better responses that require larger moves
   - For true Nash, should check all possible deviations (costly)

5. **STATISTICAL ISSUE**: Noise buffer subtraction
   - Subtracting noise BEFORE max() can give 0 even when true regret > 0
   - Better: Use confidence intervals or hypothesis testing
"""

# --- CORRECTED VERSION ---

qubits = cirq.LineQubit.range(2)
simulator = cirq.Simulator()

def get_payoffs_from_counts(counts, repetitions):
    """
    Custom payoff logic.
    Example: Alice gains on |00>, Bob gains on |11>.
    """
    p00 = counts.get(0, 0) / repetitions
    p01 = counts.get(1, 0) / repetitions
    p10 = counts.get(2, 0) / repetitions
    p11 = counts.get(3, 0) / repetitions

    # Example zero-sum or coordination payoff
    uA = p00 + 0.5 * p01 - p11
    uB = p11 + 0.5 * p10 - p00
    return uA, uB

def estimate_payoff_std(counts, payoff_coeffs, repetitions):
    """
    Estimate standard deviation of payoff estimator.
    payoff_coeffs: [c00, c01, c10, c11] coefficients for payoff = sum(c_i * p_i)
    """
    # Get probabilities
    probs = np.array([counts.get(i, 0) / repetitions for i in range(4)])

    # For linear combinations of multinomial probabilities:
    # Var(sum c_i p_i) ≈ sum c_i^2 * p_i(1-p_i)/N + cross terms
    # Simplified: sum c_i^2 * p_i / N (ignoring cross terms)
    variance = sum(payoff_coeffs[i]**2 * probs[i] for i in range(4)) / repetitions
    return np.sqrt(variance)

def objective_function(params, epsilon=0.15, shots=1000, confidence_sigma=2.0):
    """
    Calculates total regret Z = max(0, ΔuA) + max(0, ΔuB).

    Args:
        confidence_sigma: Number of std errors for noise threshold (2.0 = 95% confidence)
    """
    theta_A, theta_B = params

    # Define 3 configurations: [Current, Alice_Shift, Bob_Shift]
    configs = [
        (theta_A, theta_B),
        (theta_A + epsilon, theta_B),
        (theta_A, theta_B + epsilon)
    ]

    circuits = []
    for ta, tb in configs:
        c = cirq.Circuit(
            cirq.ry(ta)(qubits[0]),
            cirq.ry(tb)(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(qubits[0], qubits[1], key='m')
        )
        circuits.append(c)

    # Batch execution for efficiency
    results = simulator.run_batch(circuits, repetitions=shots)

    # FIX: Correct indexing - results is a list of Result objects
    u_now = get_payoffs_from_counts(results[0].histogram(key='m'), shots)
    u_A_move = get_payoffs_from_counts(results[1].histogram(key='m'), shots)
    u_B_move = get_payoffs_from_counts(results[2].histogram(key='m'), shots)

    # Calculate noise thresholds based on actual payoff variance
    # Alice's payoff coefficients: [1, 0.5, 0, -1] for [p00, p01, p10, p11]
    # Bob's payoff coefficients: [-1, 0, 0.5, 1]
    std_A_now = estimate_payoff_std(results[0].histogram(key='m'),
                                     [1, 0.5, 0, -1], shots)
    std_A_dev = estimate_payoff_std(results[1].histogram(key='m'),
                                     [1, 0.5, 0, -1], shots)
    std_B_now = estimate_payoff_std(results[0].histogram(key='m'),
                                     [-1, 0, 0.5, 1], shots)
    std_B_dev = estimate_payoff_std(results[2].histogram(key='m'),
                                     [-1, 0, 0.5, 1], shots)

    # Standard error of the difference
    std_diff_A = np.sqrt(std_A_now**2 + std_A_dev**2)
    std_diff_B = np.sqrt(std_B_now**2 + std_B_dev**2)

    # Noise buffer: confidence_sigma * std_error
    noise_buffer_A = confidence_sigma * std_diff_A
    noise_buffer_B = confidence_sigma * std_diff_B

    # Calculate Regret (Potential Gain)
    # Only count as regret if gain exceeds noise threshold
    regret_A = max(0, (u_A_move[0] - u_now[0]) - noise_buffer_A)
    regret_B = max(0, (u_B_move[1] - u_now[1]) - noise_buffer_B)

    total_regret = regret_A + regret_B
    return total_regret

# --- CORRECTED EXECUTION ---
x0 = [np.pi/2, np.pi/2]

# FIX: Add constraints for BOTH players
res = minimize(
    fun=objective_function,
    x0=x0,
    method='COBYLA',
    options={'rhobeg': 0.4, 'maxiter': 40, 'tol': 1e-3},
    constraints=[
        # Alice's constraints
        {'type': 'ineq', 'fun': lambda x: x[0]},           # theta_A >= 0
        {'type': 'ineq', 'fun': lambda x: 2*np.pi - x[0]}, # theta_A <= 2pi
        # Bob's constraints (MISSING IN ORIGINAL)
        {'type': 'ineq', 'fun': lambda x: x[1]},           # theta_B >= 0
        {'type': 'ineq', 'fun': lambda x: 2*np.pi - x[1]}, # theta_B <= 2pi
    ]
)

print(f"Optimization Success: {res.success}")
print(f"Equilibrium Strategies: Alice={res.x[0]:.3f}, Bob={res.x[1]:.3f}")
print(f"Final System Regret: {res.fun:.5f}")

print("\n" + "="*70)
print("CRITICAL FIXES MADE:")
print("="*70)
print("1. Fixed indexing: results[0][0] → results[0]")
print("2. Added missing constraints for Bob's angle")
print("3. Improved noise estimation using actual payoff variance")
print("4. Made confidence level configurable (default 2σ = 95%)")

# --- ADDITIONAL VERIFICATION ---
print("\n" + "="*70)
print("VERIFICATION: Check if solution is actually a Nash equilibrium")
print("="*70)

# Test a denser grid of deviations to verify local Nash
test_epsilons = np.linspace(-0.5, 0.5, 11)
theta_A_eq, theta_B_eq = res.x

print(f"\nTesting deviations around equilibrium:")
print(f"Current strategy: θ_A={theta_A_eq:.3f}, θ_B={theta_B_eq:.3f}")

for eps in test_epsilons:
    if eps == 0:
        continue
    # Test Alice's deviation
    c_A_dev = cirq.Circuit(
        cirq.ry(theta_A_eq + eps)(qubits[0]),
        cirq.ry(theta_B_eq)(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0], qubits[1], key='m')
    )
    result_A = simulator.run(c_A_dev, repetitions=5000)
    u_A_test = get_payoffs_from_counts(result_A.histogram(key='m'), 5000)

    # Get baseline
    c_base = cirq.Circuit(
        cirq.ry(theta_A_eq)(qubits[0]),
        cirq.ry(theta_B_eq)(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0], qubits[1], key='m')
    )
    result_base = simulator.run(c_base, repetitions=5000)
    u_base = get_payoffs_from_counts(result_base.histogram(key='m'), 5000)

    gain_A = u_A_test[0] - u_base[0]

    if abs(gain_A) > 0.01:  # Only print significant deviations
        print(f"  Δθ_A={eps:+.2f}: Alice gains {gain_A:+.4f}")

print("\n" + "="*70)
print("NOISE MITIGATION ASSESSMENT:")
print("="*70)
print("Current approach: Subtract confidence_sigma * std_error from gain")
print("Pros: Conservative, avoids false positives from noise")
print("Cons: Might be too aggressive, could miss small real regrets")
print("\nRecommendations:")
print("- Use confidence_sigma=2.0 (95% confidence) for production")
print("- Use confidence_sigma=1.0 (68% confidence) for exploration")
print("- Increase shots for tighter confidence intervals")
print("- Consider hypothesis testing instead of hard thresholds")
