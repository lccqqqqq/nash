#%% Imports
%load_ext autoreload
%autoreload 2
from src.mps_utils import apply_random_unitaries, get_rand_state_as_mps, to_canonical_form, apply_two_qubit_gate
from src.solver import find_nash_eq1
from src.game import get_default_cyclic_players
from utils.misc import plot_energy_and_exploitability
import numpy as np
#%% Random Initialization

num_players = 8
H = get_default_cyclic_players(L=num_players, dtype=np.float64)
rand_state = get_rand_state_as_mps(L=num_players, max_bond_dim=128, dtype=np.float64)
result = find_nash_eq1(
    rand_state,
    H=H,
    max_iter=3000,
    alpha=0.3,
    expl_threshold=1e-8,
    expl_check_interval=100,
    expl_maxiter=50,
    real_strategies=True,
    return_history=True,
    use_tqdm=True,
)

#%% Plot result
plot_energy_and_exploitability(result, expl_log_scale=True)


#%% Two qubit unitaries

from utils.misc import get_random_two_qubit_unitary
from src.mps_utils import get_product_state, to_comp_basis
def apply_u(u, psi, idx):
    """Apply a multi-qubit unitary to specified indices of a wavefunction.

    Args:
        u: Unitary with shape (2, 2, ..., 2, 2, ..., 2) where first half are output indices,
           second half are input indices
        psi: Wavefunction tensor with shape (2, 2, ..., 2)
        idx: List of qubit indices to apply the unitary to
    """
    l = len(u.shape)//2
    # Contract the input indices (last l indices of u) with specified indices of psi
    psi = np.tensordot(u, psi, axes=(list(range(l, 2*l)), idx))
    # Move the output indices to the correct positions
    return np.moveaxis(psi, list(range(l)), idx)


def rand_unitary_circuit(psi, depth: int, seed: int | None = None):
    """Apply random unitary circuit to a wavefunction.

    Args:
        psi: Either 1D array of shape (2^L,) or tensor of shape (2, 2, ..., 2)
        depth: Circuit depth
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)  # Set seed once at the beginning

    # Determine number of qubits
    if psi.ndim == 1:
        L = int(np.log2(len(psi)))
        psi = psi.reshape([2]*L)  # Reshape to tensor form
    else:
        L = psi.ndim

    for d in range(depth):
        for i in range(L):
            if i % 2 == d % 2:
                U = get_random_two_qubit_unitary(seed=None)  # Let RNG evolve naturally
                psi = apply_u(U.reshape(2, 2, 2, 2), psi, [i, np.mod(i+1, L)])

    return psi



init_state = get_product_state(L=3, state_per_site=[0, 0, 0])
init_state = to_comp_basis(init_state)
init_state = rand_unitary_circuit(init_state, depth=10, seed=42)
print(init_state.flatten())

#%% Entanglement Spectrum across the middle cut, as one scrambles with random two-qubit unitaries
from src.solver import compute_bipartite_entanglement_entropies
from src.mps_utils import from_comp_basis
from tqdm import tqdm
depths = np.arange(0, 20)
num_trails = 100
L = 5

# Create initial |00...0⟩ state in tensor form
init_state_tensor = np.zeros([2]*L)
init_state_tensor[(0,)*L] = 1.0

entropies = np.zeros((len(depths), num_trails))
for d_idx, depth in enumerate(tqdm(depths)):
    for t_idx in range(num_trails):
        state = rand_unitary_circuit(init_state_tensor.copy(), depth=depth, seed=None)
        state_mps = from_comp_basis(state.flatten(), L=L)
        entropies[d_idx, t_idx] = compute_bipartite_entanglement_entropies(state_mps)[L//2]

#%% Plot the entropies
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))

mean_ent = np.mean(entropies, axis=1)
depths_for_box = np.arange(len(depths))

# Dot-line for mean
plt.plot(depths, mean_ent, 'o-', color='C0', label='Mean Entanglement Entropy', markersize=6, linewidth=2, zorder=10)

# Box plot statistics at each depth
box_data = [entropies[d_idx, :] for d_idx in range(len(depths))]
box = plt.boxplot(
    box_data,
    positions=depths,
    widths=0.6,
    patch_artist=True,
    showmeans=False,
    boxprops=dict(facecolor='C1', color='C1', alpha=0.15),
    whiskerprops=dict(color='C1', alpha=0.4),
    capprops=dict(color='C1', alpha=0.4),
    medianprops=dict(color='C1', linewidth=2),
    flierprops=dict(marker='o', color='C1', alpha=0.3, markersize=4)
)

plt.xlabel('Circuit Depth', fontsize=13)
plt.ylabel('Entanglement Entropy', fontsize=13)
plt.title('Entanglement Entropy vs. Circuit Depth (Mid-Cut)', fontsize=15)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Optional: Annotate median values
# for i, med in enumerate([item.get_ydata()[0] for item in box['medians']]):
#     plt.text(depths[i], med+0.03, f"{med:.2f}", ha='center', va='bottom', fontsize=8, color='C1')

plt.tight_layout()
plt.show()

print("Mean entanglement entropies at each depth:")
for d, mean in zip(depths, mean_ent):
    print(f"Depth {d}: {mean:.4f}")

#%% Solving for Nash Equilibrium using the states
from src.solver import kick_with_u

depths = 10
L = 6
state = np.zeros([2]*L)
state[(0,)*L] = 1.0
for depth in range(1, depths):
    state = rand_unitary_circuit(state, depth=depth, seed=None)
    state_mps = from_comp_basis(state.flatten(), L=L)
    state_mps = kick_with_u(state_mps)
    # print(state_mps)
    result = find_nash_eq1(
        state_mps,
        H=get_default_cyclic_players(L=L),
        max_iter=3000,
        alpha=0.3,
        expl_threshold=1e-8,
        expl_check_interval=50,
        expl_maxiter=50,
        real_strategies=True,
        return_history=True,
        use_tqdm=True,
    )
    print(result['energy'][-1])
    print(result['num_iters'])

#%% Data Analysis
import pickle
import pandas as pd

num_players = 10

data = pickle.load(open(f'data/rcs/results_{num_players}.pkl', 'rb'))
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import numpy as np

# Group by depth, prepare data for box plot
depths = sorted(df['depth'].unique())
# Scale down each energy value by num_players
energy_data_by_depth = [df[df['depth'] == d]['energy'].values / num_players for d in depths]
# Scale down number of iters by a factor of 10
iters_data_by_depth = [df[df['depth'] == d]['num_iters'].values / 10 for d in depths]

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# --- Energy boxplot (scaled) ---
ax = axs[0]
box1 = ax.boxplot(
    energy_data_by_depth,
    positions=depths,
    patch_artist=True,
    showmeans=False,
    medianprops=dict(color='red', linewidth=2),
    # meanprops removed
    boxprops=dict(facecolor='C0', color='C0', alpha=0.3),
    whiskerprops=dict(color='C0'),
    capprops=dict(color='C0'),
    flierprops=dict(marker='.', markerfacecolor='C0', markersize=4, alpha=0.4),
    widths=0.6
)
# Get median values and connect them with a line
energy_medians = [line.get_ydata()[0] for line in box1['medians']]
ax.plot(depths, energy_medians, color='red', linestyle='-', marker='o', linewidth=2, label='Median')

ax.set_xlabel("Circuit Depth")
ax.set_ylabel("Average Payoff (Social Welfare density)")
# ax.set_title("Energy vs Depth")

# Shift xticklabels right by 1
xticks = np.arange(-1, 20)
ax.set_xticks(xticks)
# Set the labels offset by +1, so tick at pos x reads label x+1, but careful not to exceed the number of depths
ax.set_xticklabels([str(x) for x in xticks])
ax.set_xlim(-0.5, 19.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend([box1["medians"][0]], ["Median"], loc='best')

# --- Num_iters boxplot ---
ax = axs[1]
box2 = ax.boxplot(
    iters_data_by_depth,
    positions=depths,
    patch_artist=True,
    showmeans=False,
    medianprops=dict(color='red', linewidth=2),
    # meanprops removed
    boxprops=dict(facecolor='C2', color='C2', alpha=0.3),
    whiskerprops=dict(color='C2'),
    capprops=dict(color='C2'),
    flierprops=dict(marker='.', markerfacecolor='C2', markersize=4, alpha=0.4),
    widths=0.6
)
# Get median values and connect them with a line
iters_medians = [line.get_ydata()[0] for line in box2['medians']]
ax.plot(depths, iters_medians, color='red', linestyle='-', marker='o', linewidth=2, label='Median')

ax.set_xlabel("Circuit Depth")
ax.set_ylabel("Num Iters to Convergence")
# ax.set_title("Num Iters vs Depth")

# Shift xticklabels right by 1
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])
ax.set_xlim(-0.5, 19.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend([box2["medians"][0]], ["Median"], loc='best')

plt.tight_layout()
plt.show()

#%% 