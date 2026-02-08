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

def plot_energy_and_exploitability(result, relative_to_init=True, expl_log_scale=True, cmap='coolwarm'):
    import matplotlib.cm as cm

    n_players = len(result['energy'][0])
    colormap = cm.get_cmap(cmap, n_players)
    colors = [colormap(i / max(n_players - 1, 1)) for i in range(n_players)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Energy
    energy = result['energy'] - result['energy'][0] if relative_to_init else result['energy']
    for i in range(n_players):
        axs[0].plot(energy[:, i], linewidth=1.5, color=colors[i], label=f"Player {i}")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Energy")
    axs[0].set_title("Energy Trajectory" + (" (relative)" if relative_to_init else ""))
    axs[0].legend()

    # Right panel: Exploitability
    expl = np.array(result['expl'])
    if expl.ndim == 2:
        for i in range(n_players):
            if expl_log_scale:
                axs[1].semilogy(expl[:, i], linewidth=1.5, color=colors[i], label=f"Player {i}")
            else:
                axs[1].plot(expl[:, i], linewidth=1.5, color=colors[i], label=f"Player {i}")
        axs[1].legend()
    else:
        if expl_log_scale:
            axs[1].semilogy(expl, linewidth=1.5, color=colors[0])
        else:
            axs[1].plot(expl, linewidth=1.5, color=colors[0])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Exploitability")
    axs[1].set_title("Exploitability" + (" (log scale)" if expl_log_scale else ""))

    plt.tight_layout()

def get_near_identity_unitary(n_qubits: int, epsilon: float = 0.01, dtype: np.dtype = np.complex128, seed=None):
    """Generate an n-qubit unitary that is close to identity.

    Uses the exponential map: U = exp(i * epsilon * H) where H is a random
    normalized Hermitian matrix. When epsilon is small, U ≈ I + i*epsilon*H,
    making U close to the identity matrix.

    Algorithm:
    1. Generate random normalized Hermitian matrix H (Frobenius norm = 1)
    2. Compute U = exp(i * epsilon * H) using matrix exponential
    3. Distance from identity: ||U - I||_F ≈ epsilon for small epsilon

    Args:
        n_qubits: Number of qubits (Hilbert space dimension = 2^n_qubits)
        epsilon: Perturbation strength controlling distance from identity.
                 epsilon=0 gives I, larger epsilon gives further deviation.
                 Typical range: [0.001, 0.1]
        dtype: NumPy dtype for the output unitary (complex64 or complex128)
        seed: Optional random seed for reproducibility

    Returns:
        U: (2^n_qubits, 2^n_qubits) unitary matrix close to identity

    Example:
        >>> U = get_near_identity_unitary(n_qubits=2, epsilon=0.05)
        >>> print(np.linalg.norm(U - np.eye(4)))  # Distance from identity
        ~0.05
    """
    if seed is not None:
        np.random.seed(seed)

    d = 2 ** n_qubits

    # Generate random normalized Hermitian matrix
    # Use float64 for intermediate computation for numerical stability
    H = get_rand_normalized_herm_matrix(d, dtype=np.complex128)

    # Compute U = exp(i * epsilon * H)
    # scipy.linalg.expm is more accurate than np.linalg.matrix_power for matrix exponentials
    from scipy.linalg import expm
    U = expm(1j * epsilon * H)

    # Cast to requested dtype
    U = U.astype(dtype)

    return U


def get_random_two_qubit_unitary(seed=None):
    """Sample a random two-qubit unitary from the Haar measure.

    Uses QR decomposition of a random complex Gaussian matrix to generate
    a uniformly distributed unitary matrix from U(4). This is the standard
    algorithm for sampling from the Haar measure on U(d).

    Algorithm:
    1. Generate random complex Gaussian matrix Z with i.i.d. N(0,1) entries
    2. QR decomposition: Z = QR
    3. Phase correction: multiply Q by phases from diag(R) to ensure uniformity

    Args:
        seed: Optional random seed for reproducibility

    Returns:
        U: 4x4 complex unitary matrix sampled uniformly from U(4)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random complex Gaussian matrix
    Z = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)

    # QR decomposition
    Q, R = np.linalg.qr(Z)

    # Multiply by phase factors from R to get Haar-random unitary
    Lambda = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    U = Q @ Lambda

    return U


