import numpy as np
import matplotlib.pyplot as plt

def get_rand_normalized_herm_matrix(d: int, dtype: np.dtype = np.float32):
    if dtype == np.float32:
        rand_herm_matrix = np.random.randn(d, d)
        rand_herm_matrix = rand_herm_matrix + rand_herm_matrix.T
        rand_herm_matrix = rand_herm_matrix / np.linalg.norm(rand_herm_matrix)
    else:
        rand_herm_matrix = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        rand_herm_matrix = rand_herm_matrix + rand_herm_matrix.T.conj()
        rand_herm_matrix = rand_herm_matrix / np.linalg.norm(rand_herm_matrix)

    return rand_herm_matrix

def plot_energy_and_exploitability(result, relative_to_init=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Energy
    if relative_to_init:
        axs[0].plot(result['energy']-result['energy'][0], linewidth=1, label=[f"energy {i}" for i in range(len(result['energy'][0]))])
    else:
        axs[0].plot(result['energy'], linewidth=1, label=[f"energy {i}" for i in range(len(result['energy'][0]))])
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Energy")
    axs[0].set_title("Energy Trajectory (Unsuccessful run: alpha=0.08)")
    axs[0].legend()

    # Right panel: Exploitability
    axs[1].plot(result['expl'])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Exploitability")
    axs[1].set_title("Exploitability Trajectory")

    plt.tight_layout()


