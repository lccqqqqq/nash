"""
Cirq Demonstration Script
=========================
This script demonstrates basic usage of Google's Cirq quantum computing package.

Installation:
    pip install cirq

Key concepts demonstrated:
- Circuit construction
- Quantum gates (single and multi-qubit)
- Entanglement creation (relevant to Nash equilibrium research)
- Simulation and measurement
- State vector extraction
"""

import numpy as np

try:
    import cirq
except ImportError:
    print("Cirq is not installed. Please run: pip install cirq")
    exit(1)


def demo_basic_circuit():
    """Demonstrate basic single-qubit operations."""
    print("=" * 60)
    print("Demo 1: Basic Single-Qubit Circuit")
    print("=" * 60)

    # Create a qubit
    qubit = cirq.LineQubit(0)

    # Create a circuit
    circuit = cirq.Circuit(
        cirq.H(qubit),      # Hadamard gate: |0⟩ → (|0⟩ + |1⟩)/√2
        cirq.measure(qubit, key='result')
    )

    print("Circuit:")
    print(circuit)

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)

    print(f"\nMeasurement results (1000 shots):")
    print(result.histogram(key='result'))
    print()


def demo_entanglement():
    """Demonstrate two-qubit entanglement (Bell state)."""
    print("=" * 60)
    print("Demo 2: Entanglement - Bell State Creation")
    print("=" * 60)

    # Create two qubits
    q0, q1 = cirq.LineQubit.range(2)

    # Create Bell state: (|00⟩ + |11⟩)/√2
    circuit = cirq.Circuit(
        cirq.H(q0),           # Create superposition
        cirq.CNOT(q0, q1),    # Entangle q0 and q1
        cirq.measure(q0, q1, key='result')
    )

    print("Circuit:")
    print(circuit)

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)

    print(f"\nMeasurement results (1000 shots):")
    print("Note: Should only see |00⟩ and |11⟩ (perfect correlation)")
    print(result.histogram(key='result'))
    print()


def demo_ghz_state():
    """
    Demonstrate 3-qubit GHZ state creation.

    This is directly relevant to the 3-player quantum game research in this repo!
    GHZ state: (|000⟩ + |111⟩)/√2
    """
    print("=" * 60)
    print("Demo 3: Three-Qubit GHZ State")
    print("=" * 60)
    print("(Relevant to 3-player quantum games in this project!)")
    print()

    # Create three qubits
    q0, q1, q2 = cirq.LineQubit.range(3)

    # Create GHZ state
    circuit = cirq.Circuit(
        cirq.H(q0),            # Create superposition on q0
        cirq.CNOT(q0, q1),     # Entangle q0 and q1
        cirq.CNOT(q0, q2),     # Entangle q0 and q2
    )

    print("Circuit (without measurement):")
    print(circuit)

    # Get state vector
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state_vector = result.final_state_vector

    print(f"\nState vector:")
    print(f"Basis order: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩")
    print(f"Amplitudes: {state_vector}")
    print(f"\nNote: Only |000⟩ and |111⟩ have non-zero amplitudes (≈ 1/√2 ≈ 0.707)")

    # Now measure
    circuit_with_measurement = circuit + cirq.Circuit(
        cirq.measure(q0, q1, q2, key='result')
    )
    result = simulator.run(circuit_with_measurement, repetitions=1000)

    print(f"\nMeasurement results (1000 shots):")
    print("Note: Should only see |000⟩=0 and |111⟩=7")
    print(result.histogram(key='result'))
    print()


def demo_parameterized_circuit():
    """Demonstrate parameterized circuits (useful for optimization)."""
    print("=" * 60)
    print("Demo 4: Parameterized Circuit (Useful for Optimization)")
    print("=" * 60)

    # Create a qubit
    qubit = cirq.LineQubit(0)

    # Create a symbol for the rotation angle
    theta = cirq.Symbol('theta')

    # Create parameterized circuit
    circuit = cirq.Circuit(
        cirq.ry(theta)(qubit),  # Y-rotation by angle theta
        cirq.measure(qubit, key='result')
    )

    print("Parameterized circuit:")
    print(circuit)

    # Sweep over different values of theta
    print(f"\nSweeping rotation angle from 0 to π:")
    print(f"{'Angle (π)':<15} {'P(|1⟩)':<10}")
    print("-" * 25)

    simulator = cirq.Simulator()
    for angle_factor in [0, 0.25, 0.5, 0.75, 1.0]:
        angle = angle_factor * np.pi
        resolver = cirq.ParamResolver({'theta': angle})
        result = simulator.run(circuit, param_resolver=resolver, repetitions=1000)
        prob_one = np.sum(result.measurements['result']) / 1000
        print(f"{angle_factor:<15.2f} {prob_one:<10.3f}")

    print(f"\nNote: ry(θ) creates state cos(θ/2)|0⟩ + sin(θ/2)|1⟩")
    print(f"So P(|1⟩) = sin²(θ/2)")
    print()


def demo_unitary_extraction():
    """Demonstrate extracting the unitary matrix of a circuit."""
    print("=" * 60)
    print("Demo 5: Extracting Unitary Matrices")
    print("=" * 60)
    print("(Relevant for computing Nash equilibria with unitary strategies)")
    print()

    # Simple single-qubit circuit
    qubit = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.H(qubit),
        cirq.T(qubit),  # T gate (π/4 phase)
    )

    print("Circuit:")
    print(circuit)

    # Get unitary
    unitary = cirq.unitary(circuit)
    print(f"\nUnitary matrix (2x2 for 1 qubit):")
    print(unitary)

    # Verify it's unitary
    identity = unitary @ unitary.conj().T
    print(f"\nVerification: U × U† = I?")
    print(np.allclose(identity, np.eye(2)))

    # Two-qubit example
    print("\n" + "-" * 40)
    q0, q1 = cirq.LineQubit.range(2)
    circuit2 = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
    )

    print("Two-qubit circuit (Bell state prep):")
    print(circuit2)

    unitary2 = cirq.unitary(circuit2)
    print(f"\nUnitary matrix (4x4 for 2 qubits):")
    print(unitary2)
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 60)
    print("CIRQ DEMONSTRATION")
    print("Google's Quantum Computing Framework")
    print("*" * 60)
    print()

    demo_basic_circuit()
    demo_entanglement()
    demo_ghz_state()
    demo_parameterized_circuit()
    demo_unitary_extraction()

    print("=" * 60)
    print("Comparison with your current MPS approach:")
    print("=" * 60)
    print("""
Your current codebase uses Matrix Product States (MPS) to represent
quantum states, which is more efficient for:
- Large numbers of qubits (>10)
- States with limited entanglement
- Tensor network operations

Cirq is better suited for:
- Small-scale quantum circuits (<20 qubits)
- Gate-based quantum algorithms
- Near-term quantum hardware experiments
- Direct simulation of quantum computers

For your 3-player quantum game research, MPS is likely the right choice
for scalability, but Cirq could be useful for:
- Prototyping small examples
- Verifying MPS implementations against exact simulation
- Exploring gate-based strategies before converting to MPS form
    """)


if __name__ == "__main__":
    main()
