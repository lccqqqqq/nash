# Get the optimality of the GHZ state orbit for a set of players.
from src.game import get_default_cyclic_players
from src.solver import find_nash_eq1
from utils.misc import plot_energy_and_exploitability, get_near_identity_unitary
from src.mps_utils import get_ghz_state
from src.solver import kick_with_u
from src.mps_utils import to_comp_basis, from_comp_basis
import numpy as np
from tqdm import tqdm
import uuid
import os
import pickle
import argparse

def main(args):
    L = 6
    H = get_default_cyclic_players(L=L)
    state = get_ghz_state(L=L)
    state = kick_with_u(state)
    result = find_nash_eq1(state, H=H, max_iter=1000, alpha=0.3, convergence_threshold=1e-6, expl_threshold=1e-5, use_tqdm=True, expl_check_interval=30, return_history=True)
    plot_energy_and_exploitability(result, relative_to_init=False)
    print(sum(result['energy'][-1]))

    # Perturb the state using a near-identity unitary
    eps = args.eps
    pbar = tqdm(range(args.n_samples))
    U_list = []
    welfare_list = []
    perturbed_state_list = []
    for i in pbar:
        U = get_near_identity_unitary(L, epsilon=eps)
        state_comp_basis = to_comp_basis(state)
        perturbed_state = U @ state_comp_basis
        # print(f"Distance from identity: {np.linalg.norm(U - np.eye(2**L))}")
        # print(f"Perturbed state fidelity: {np.abs(np.vdot(state_comp_basis, perturbed_state))}")
        perturbed_state = np.abs(perturbed_state)
        perturbed_state = from_comp_basis(perturbed_state, L=L)
        perturbed_state = kick_with_u(perturbed_state)

        result = find_nash_eq1(perturbed_state, H=H, max_iter=1000, alpha=0.3, convergence_threshold=1e-8, expl_threshold=1e-5, use_tqdm=False, expl_check_interval=30, return_history=True, real_strategies=True)
        # plot_energy_and_exploitability(result, relative_to_init=False)
        welfare_list.append(sum(result['energy'][-1]))
        U_list.append(U)
        perturbed_state_list.append(perturbed_state)
        print(sum(result['energy'][-1]))
        pbar.set_postfix(welfare=sum(result['energy'][-1]))

    # Save data

    run_uuid = str(uuid.uuid4())[:8]
    data = {
        'welfare': welfare_list,
        'U': U_list,
        'perturbed_state': perturbed_state_list
    }
    os.makedirs('data/ghz_local_optimality', exist_ok=True)
    with open(f'data/ghz_local_optimality/L{L}_eps{eps}_n{len(welfare_list)}_{run_uuid}.pkl', 'wb') as f:
        pickle.dump(data, f)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Analyze GHZ state local optimality in quantum games',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )

#     parser.add_argument('--eps', type=float, default=0.3,
#                         help='Perturbation epsilon for near-identity unitaries')
#     parser.add_argument('--n-samples', type=int, default=100,
#                         help='Number of perturbed states to sample')

#     args = parser.parse_args()
#     main(args)
# #%% Load data
# import pickle
# import pandas as pd
# data = pickle.load(open('data/ghz_local_optimality/L6_eps0.3_n100_93ab0dca.pkl', 'rb'))
# data = pd.DataFrame(data)
# data.head()
# #%%
# filtered = data[data['welfare'] > 14.5]
# filtered['welfare'].hist()

#%% Load data
import pickle
import pandas as pd
import os
import re
from collections import defaultdict, OrderedDict
from tqdm import tqdm
welfare_list_by_eps = defaultdict(list)
for file in tqdm(os.listdir('data/ghz_local_optimality')):
    if file.endswith('.pkl'):

        # Example filename: 'L6_eps0.3_n100_93ab0dca.pkl'
        match = re.search(r'_eps([0-9.]+)_', file)
        if match:
            eps_val = float(match.group(1))
        else:
            eps_val = None  # or handle error
        # print(f"Parsed eps from filename: {eps_val}")

        data = pickle.load(open(os.path.join('data/ghz_local_optimality', file), 'rb'))
        welfare_list_by_eps[eps_val].extend(data['welfare'])



#%%
welfare_list_by_eps = OrderedDict(
    (eps, [w.item() if hasattr(w, 'item') else w for w in ws if w > 14.5])
    for eps, ws in sorted(welfare_list_by_eps.items())
)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
eps_values = list(welfare_list_by_eps.keys())
eps_labels = [f'{eps:g}' for eps in eps_values]
data = [welfare_list_by_eps[eps] for eps in eps_values]
non_empty = [(lbl, vals) for lbl, vals in zip(eps_labels, data) if len(vals) > 0]
if non_empty:
    labels, datasets = zip(*non_empty)
    positions = range(1, len(datasets) + 1)
    boxprops = dict(linewidth=1.5, color='#555555')
    medianprops = dict(color='#222222', linewidth=2)
    whiskerprops = dict(linewidth=1.2, color='#777777')
    capprops = dict(linewidth=1.2, color='#777777')
    bp = plt.boxplot(
        datasets,
        labels=labels,
        patch_artist=False,
        positions=positions,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=dict(marker='o', markersize=4, markerfacecolor='#bbbbbb', markeredgecolor='none'),
    )
    mean_vals = [np.mean(vals) for vals in datasets]
    plt.plot(positions, mean_vals, marker='o', color='#0d0887', linewidth=2, label='Mean')
    plt.xlabel('ε')
    plt.ylabel('Welfare')
    plt.title('GHZ Local Optimality vs. Perturbation ε')
    plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
else:
    print('No welfare samples available for plotting.')
