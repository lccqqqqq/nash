import pickle
import pandas as pd
from tqdm import tqdm
from mps_utils import to_comp_basis
import numpy as np
import os
import matplotlib.pyplot as plt
from opt_mps_fiducial_state import compute_energies, apply_unitary
from mps_utils import from_comp_basis

def plot_state(
    state,
    title: str = "State in Computational Basis",
    figure_size: tuple = (12, 5),
    x_ticks_rotation: int = 45,
):
    num_qubits = int(np.log2(len(state)))
    labels = [format(i, f'0{num_qubits}b') for i in range(len(state))]

    plt.figure(figsize=figure_size)

    if np.iscomplexobj(state):
        # Plot real and imaginary parts
        width = 0.35
        x = np.arange(len(state))
        plt.bar(x - width/2, np.real(state), width=width, label="Real part", alpha=0.8)
        plt.bar(x + width/2, np.imag(state), width=width, label="Imaginary part", alpha=0.8)
        plt.xlabel("Computation Basis States")
        plt.ylabel("Amplitude")
        plt.title(f"{title} (Real & Imag)")
        plt.xticks(x, labels, rotation=x_ticks_rotation)
        plt.legend()
    else:
        plt.bar(labels, np.abs(state))
        plt.xlabel("Computation Basis States")
        plt.ylabel("Amplitude")
        plt.title(f"{title}")
        plt.xticks(rotation=x_ticks_rotation)

    plt.tight_layout()
    plt.show()

def compute_exploitability(psi, H, num_samples=1000):
    if isinstance(psi, np.ndarray):
        psi = from_comp_basis(psi)

    alpha = np.linspace(0, np.pi, num_samples)
    unitaries = [
        np.array([[np.cos(alpha[i]), -np.sin(alpha[i])], [np.sin(alpha[i]), np.cos(alpha[i])]])
        for i in range(num_samples)
    ]
    baseline_energies = compute_energies(psi, H)
    expl = np.zeros((int(np.log2(psi.shape[0])) if isinstance(psi, np.ndarray) else len(psi), num_samples))
    for player in tqdm(range(len(psi)), desc="Computing exploitability for players"):
        for i in range(num_samples):
            Psi_ = [apply_unitary(unitaries[i], psi[j]) if j == player else psi[j] for j in range(len(psi))]
            energies_ = compute_energies(Psi_, H)
            expl[player, i] = energies_[player] - baseline_energies[player]
    return expl

def plot_exploitability_vs_angle(expl, num_samples=None):
    if num_samples is None:
        num_samples = expl.shape[1]
    num_players = expl.shape[0]
    alpha = np.linspace(0, np.pi, num_samples)

    plt.figure(figsize=(10, 6))
    for player in range(num_players):
        plt.plot(alpha, expl[player], label=f'Player {player}')
    plt.title('Exploitability vs Angle for Each Player')
    plt.xlabel('Rotation Angle (radians)')
    plt.ylabel('Exploitability')
    plt.legend(title='Player')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, np.pi)
    plt.gca().set_xlim(left=0, right=np.pi)
    plt.gca().margins(x=0)
    plt.axhline(y=0, color='red', linewidth=2.5, linestyle='--')
    plt.show()

def compute_and_validate_nash_equilibrium(
    psi,
    H,
    num_samples=1000,
    tolerance=1e-3,
    title="Nash Equilibrium State",
    figure_size=(12, 5),
):
    """
    Pipeline to validate a Nash equilibrium state.

    Args:
        psi: MPS representation of the state (list of tensors) or computational basis vector
        H: Hamiltonian tensor (payoff matrix)
        num_samples: Number of unitary samples for exploitability computation
        tolerance: Threshold for exploitability to consider state as Nash equilibrium
        title: Title for the state plot
        figure_size: Figure size for the state plot

    Returns:
        dict: Dictionary containing:
            - 'state': state in computational basis
            - 'exploitability': exploitability array (num_players x num_samples)
            - 'max_exploitability': maximum exploitability per player
            - 'is_nash': boolean indicating if state is a Nash equilibrium
    """
    # Step 1: Convert to computational basis if needed
    if isinstance(psi, list):
        state = to_comp_basis(psi)
    else:
        state = psi

    # Step 2: Plot the state in computational basis
    print(f"Plotting state in computational basis...")
    plot_state(state, title=title, figure_size=figure_size)

    # Step 3: Compute exploitability
    print(f"\nComputing exploitability with {num_samples} samples...")
    expl = compute_exploitability(psi, H, num_samples=num_samples)

    # Step 4: Plot exploitability vs angle
    print(f"Plotting exploitability vs angle...")
    plot_exploitability_vs_angle(expl, num_samples=num_samples)

    # Step 5: Verify exploitabilities are near zero
    max_expl = np.max(expl, axis=1)
    is_nash = np.all(max_expl < tolerance)

    print(f"\n{'='*60}")
    print(f"Nash Equilibrium Validation Results")
    print(f"{'='*60}")
    num_players = expl.shape[0]
    for player in range(num_players):
        status = "✓" if max_expl[player] < tolerance else "✗"
        print(f"Player {player}: max exploitability = {max_expl[player]:.6f} {status}")
    print(f"{'='*60}")
    print(f"Tolerance threshold: {tolerance}")
    print(f"Is Nash Equilibrium: {is_nash}")
    print(f"{'='*60}\n")

    return {
        'state': state,
        'exploitability': expl,
        'max_exploitability': max_expl,
        'is_nash': is_nash,
    }

if __name__ == "__main__":
    state = np.array([1, 0, 0, 0])
    plot_state(state, title="State in Computational Basis")