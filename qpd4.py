import numpy as np
from solver import find_nash_eq1, compute_exploitability
from game import get_default_4players, get_default_H
from mps_utils import get_rand_mps
import time
import uuid
import os
import pandas as pd
import mpi4py
from tqdm import tqdm

def gather_equilibrium_statistics(
    H: list[np.ndarray],
    L: int,
    num_samples: int = 50,
    max_iter: int = 1000,
    alpha: float = 0.06,
    convergence_threshold: float = 1e-6,
    expl_threshold: float = 1e-3,
    use_tqdm: bool = False,
    expl_check_interval: int = 10,
    return_history: bool = False,
    save_dir: str = "data/qpd4_results",
):
    results = []
    for i in tqdm(range(num_samples)):
        Psi = get_rand_mps(L=L, chi=32, dtype=np.float32)
        result = find_nash_eq1(Psi=Psi, H=H, max_iter=max_iter, alpha=alpha, convergence_threshold=convergence_threshold, expl_threshold=expl_threshold, use_tqdm=use_tqdm, expl_check_interval=expl_check_interval, return_history=return_history)
        results.append(result)
    
    # Saving results as dataframe to disk
    df = pd.DataFrame(results)
    os.makedirs(save_dir, exist_ok=True)
    df.to_pickle(os.path.join(save_dir, f"eqdata_randstates_{str(uuid.uuid4())[:8]}.pkl"))
    return results


if __name__ == "__main__":
    H = get_default_H(num_players=6, option='H', dtype=np.float32)
    results = gather_equilibrium_statistics(H=H, L=6, save_dir="data/qpd6_results")