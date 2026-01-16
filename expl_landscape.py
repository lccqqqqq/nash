#%% Aiming to map out the exploitability landscape for a particular game
import numpy as np
from tqdm import tqdm
from src.solver import find_nash_eq1
from src.game import get_default_cyclic_players, get_perturbed_H_QPD
from src.mps_utils import to_comp_basis, to_canonical_form, get_product_state
from src.solver import compute_exploitability, apply_u, apply_unitary, kick_with_u, update_state_unitary, compute_bipartite_entanglement_entropies
from functools import reduce
import torch as t

#%% Exploitability landscape via grids
payoff_dtype = np.complex128
state_dtype = np.complex128
L = 2
def scramble(state, depth=10, unitary_eps=0.1, seed=42):
    L = len(state)
    np.random.seed(seed)
    for i in range(depth):
        state = update_state_unitary(state, np.random.randn(9), lr=unitary_eps, site=np.mod(i, L-1))
    return state

Hs = get_perturbed_H_QPD(eps=1, dtype=payoff_dtype, seed=43)
H = get_default_cyclic_players(L=L, Hs=Hs, dtype=payoff_dtype)
Psi = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
Psi = scramble(Psi, depth=1, unitary_eps=0.1, seed=42)

print(f"Initial state: {to_comp_basis(Psi)}")
print(f"Entanglement entropies: {compute_bipartite_entanglement_entropies(Psi)}")

#%% Plot Util
import matplotlib.pyplot as plt
def plot_energy_and_exploitability(result, relative_to_init=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Energy
    L = len(result['energy'][0])
    if relative_to_init:
        axs[0].plot(result['energy']-result['energy'][0], linewidth=1, label=[f"energy {i}" for i in range(L)])
    else:
        axs[0].plot(result['energy'], linewidth=1, label=[f"energy {i}" for i in range(L)])
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Energy")
    if relative_to_init:
        axs[0].set_title("Energy Trajectory (Relative to Initial)")
    else:
        axs[0].set_title("Energy Trajectory (Absolute)")
    axs[0].legend()

    # Right panel: Exploitability
    axs[1].plot(result['expl'])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Exploitability")
    axs[1].set_title("Exploitability Trajectory")
    axs[1].legend()
    axs[1].set_yscale('log')

    plt.tight_layout()

#%% Find Nash equilibrium
Psi = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
Psi = scramble(Psi, depth=1, unitary_eps=0.1, seed=42)
print(f"Initial State: {to_comp_basis(Psi)}")
result = find_nash_eq1(
    Psi,
    H,
    max_iter=1000,
    expl_threshold=1e-16,
    real_strategies=False,
    alpha=0.1,
    expl_check_interval=29,
    expl_maxiter=300,
    return_history=True,
    use_tqdm=True,
)
plot_energy_and_exploitability(result, relative_to_init=True)

#%% Fibonacci sphere sampling
def fibonacci_sphere(n_samples):
    """
    Generate n_samples points uniformly distributed on a unit sphere using the Fibonacci spiral method.

    Args:
        n_samples: Number of points to generate

    Returns:
        points: Array of shape (n_samples, 3) containing (x, y, z) coordinates
        theta: Array of polar angles [0, π]
        phi: Array of azimuthal angles [0, 2π]
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi / (golden_ratio ** 2)  # ~2.399963 radians

    indices = np.arange(n_samples)

    # y-coordinate: evenly spaced from 1 to -1
    y = 1 - (indices / (n_samples - 1)) * 2

    # Radius in xy-plane
    radius = np.sqrt(1 - y**2)

    # Azimuthal angle using golden angle
    phi = (golden_angle * indices) % (2 * np.pi)

    # Cartesian coordinates
    x = np.cos(phi) * radius
    z = np.sin(phi) * radius

    # Convert to spherical coordinates
    theta = np.arccos(y)  # polar angle [0, π]

    points = np.stack([x, y, z], axis=1)

    return points, theta, phi


def su2_from_spherical(alpha, theta, phi, dtype=np.complex128):
    """
    Construct SU(2) unitary from spherical coordinates.

    U = cos(α)I + i·sin(α)(n·σ)
    where n = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))

    Args:
        alpha: Rotation angle [0, π]
        theta: Polar angle [0, π]
        phi: Azimuthal angle [0, 2π]
        dtype: Data type for the unitary

    Returns:
        U: 2×2 unitary matrix
    """
    PAULIS = [
        np.array([[0, 1], [1, 0]], dtype=dtype),      # σ_x
        np.array([[0, -1j], [1j, 0]], dtype=dtype),   # σ_y
        np.array([[1, 0], [0, -1]], dtype=dtype),     # σ_z
    ]

    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)

    U = (np.eye(2, dtype=dtype) * np.cos(alpha) +
         1j * np.sin(alpha) * (nx * PAULIS[0] + ny * PAULIS[1] + nz * PAULIS[2]))

    return U


def generate_unitary_grid_fibonacci(n_directions, n_angles, dtype=np.complex128):
    """
    Generate a grid of SU(2) unitaries using Fibonacci sphere sampling.

    Args:
        n_directions: Number of directions (θ, φ) sampled on the Bloch sphere
        n_angles: Number of rotation angles α ∈ [0, π]
        dtype: Data type for unitaries

    Returns:
        unitaries: List of 2×2 unitary matrices
        params: Array of shape (N, 3) containing (α, θ, φ) for each unitary
    """
    # Sample directions using Fibonacci sphere
    _, thetas, phis = fibonacci_sphere(n_directions)

    # Sample rotation angles uniformly
    alphas = np.linspace(0, np.pi, n_angles)

    unitaries = []
    params = []

    for alpha in alphas:
        for theta, phi in zip(thetas, phis):
            U = su2_from_spherical(alpha, theta, phi, dtype=dtype)
            unitaries.append(U)
            params.append([alpha, theta, phi])

    return unitaries, np.array(params)


#%% Test: Visualize Fibonacci sampling
import plotly.graph_objs as go

def test_fibonacci_sampling():
    """Visualize the Fibonacci sphere sampling using plotly for interactivity"""
    n_samples = 200
    points, theta, phi = fibonacci_sphere(n_samples)

    # Create a 3D scatter plot using plotly
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=theta,  # coloring by theta or just use a uniform color
            colorscale='Viridis',
            opacity=0.7
        )
    )

    layout = go.Layout(
        title=f'Fibonacci Sphere Sampling ({n_samples} points)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

    print(f"Generated {n_samples} points on sphere")
    print(f"θ range: [{theta.min():.3f}, {theta.max():.3f}] (should be [0, π])")
    print(f"φ range: [{phi.min():.3f}, {phi.max():.3f}] (wraps around 2π)")

# Uncomment to test (plotly opens in Jupyter notebook or browser):
test_fibonacci_sampling()

#%% Mapping out the exploitability landscape

# Generate grid of unitaries using Fibonacci sampling
n_directions = 50  # Number of directions on Bloch sphere
n_angles = 10      # Number of rotation angles
print(f"Generating {n_directions * n_angles} unitaries...")

unitaries, params = generate_unitary_grid_fibonacci(n_directions, n_angles, dtype=state_dtype)
print(f"Total unitaries: {len(unitaries)}")


#%% Compute exploitability landscape
# Apply each unitary to player 0 and compute their exploitability
player_idx = 0
psi_ne = to_comp_basis(result['state_'][-1]).reshape([2] * L)  # Nash equilibrium state

Psi = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
Psi = scramble(Psi, depth=1, unitary_eps=0.1, seed=42)
print(f"Initial State: {to_comp_basis(Psi)}")
result = find_nash_eq1(
    Psi,
    H,
    max_iter=1000,
    expl_threshold=1e-16,
    real_strategies=False,
    alpha=0.1,
    expl_check_interval=29,
    expl_maxiter=300,
    return_history=True,
    use_tqdm=True,
)
import itertools
from copy import deepcopy
expls = []
A = deepcopy(to_comp_basis(Psi).reshape([2] * L))
for i, j in tqdm(itertools.product(range(len(unitaries)), range(len(unitaries)))):
    psi = apply_u(unitaries[i], A, 0)
    psi = apply_u(unitaries[j], psi, 1)
    expl = [compute_exploitability(psi, H, player,
                                  maxiter=300, seed=42, real_strategies=False) for player in range(L)]
    expls.append(np.sum(expl))


np.save('expl_landscape.npy', np.array(expls))

    