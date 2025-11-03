import numpy as np
from functools import reduce
import math
from tqdm import tqdm
from mpi4py import MPI
from scipy.optimize import differential_evolution
import os
def spherical_meshgrid(dim_sphere, n_points):
    """
    Create a meshgrid on a d-dimensional sphere in Cartesian coordinates.
    
    Parameters:
    - dim_sphere: dimension of the embedding space (e.g., 3 for points on S^2)
    - n_points: number of points along each angular dimension
    
    Returns:
    - Cartesian coordinates of shape (dim_sphere, n_points^(dim_sphere-1))
    """
    # For d-dimensional sphere embedded in R^d, need (d-1) angles
    n_angles = dim_sphere - 1
    
    # Create angular grids
    angles = []
    for i in range(n_angles):
        if i < n_angles - 1 and i != 0:
            # Polar angles: [0, π]
            angles.append(np.linspace(0, np.pi, n_points))
        else:
            # Azimuthal angle: [0, 2π)
            angles.append(np.linspace(0, 2*np.pi, n_points))
    
    # Create meshgrid and flatten
    grids = np.meshgrid(*angles, indexing='ij')
    flat_grids = [g.flatten() for g in grids]
    
    n_total_points = n_points ** n_angles
    coords = np.zeros((dim_sphere, n_total_points))
    
    # Convert to Cartesian using spherical coordinate formulas
    for i in range(dim_sphere):
        if i < dim_sphere - 1:
            coords[i] = np.cos(flat_grids[i])
            for j in range(i):
                coords[i] *= np.sin(flat_grids[j])
        else:  # last coordinate
            coords[i] = np.ones(n_total_points)
            for j in range(n_angles):
                coords[i] *= np.sin(flat_grids[j])
    
    north_poles = np.eye(dim_sphere)
    south_poles = -np.eye(dim_sphere)
    
    coords = np.vstack([north_poles, coords.T, south_poles])
    
    return coords

def pure_state_grid(n_points):
    sphere_grid = spherical_meshgrid(dim_sphere=5, n_points=n_points)
    all_grid = []
    for i, grid in tqdm(enumerate(sphere_grid)):
        if abs(grid[1]) >= 1e-5 and abs(1 - abs(grid[1])) >= 1e-5:
            grid_rep = np.tile(grid, (n_points, 1))
            phi = np.linspace(0, np.pi, n_points)
            all_grid.append(np.concatenate([grid_rep, np.expand_dims(phi, axis=1)], axis=1))
        else:
            all_grid.append(
                np.concatenate([grid, np.array([0])])
            )
    
    return np.vstack(all_grid)

def get_canonical_form_state(input_params):
    """
    Convert parameters describing the canonical form of a 3-qubit state into a state tensor.
    The parametrisation follows the form used in the accompanying notebooks.
    """
    canonical_vec = np.array([
        input_params[0], 0, 0, 0,
        input_params[1] * np.exp(1j * input_params[-1]),
        input_params[2], input_params[3], input_params[4]
    ], dtype=np.complex128)
    return canonical_vec.reshape(2, 2, 2)


def apply_u(u,psi,idx):
    l = len(u.shape)//2
    psi = np.tensordot(u, psi, axes=(list(range(l)), idx))
    return np.moveaxis(psi, list(range(l)), idx)

def compute_exploitability(psi, H, player_idx):
    """
    Compute exploitability for a given player by optimising over single-qubit rotations.
    """
    L = psi.ndim
    j = (player_idx - 1) % L
    k = (player_idx + 1) % L

    def uni_dev_payoff(alpha_vec):
        alpha = alpha_vec[0]
        unitary = np.eye(2) * math.cos(alpha) + np.array([[0, 1], [-1, 0]]) * math.sin(alpha)
        psi_dev = apply_u(unitary, psi, [player_idx])
        dE = np.tensordot(H[player_idx], psi_dev, axes=([4, 5, 3], [1, 2, 0]))
        dE = np.tensordot(psi_dev.conj(), dE, axes=([j, k], [j, k]))
        return -float(np.trace(dE).real)

    result = differential_evolution(
        uni_dev_payoff,
        bounds=[(0, math.pi)],
        maxiter=100,
        seed=42,
        atol=1e-6,
        tol=1e-6,
    )
    return -result.fun + uni_dev_payoff(np.array([0.0]))

def equilibrium_finding(psi, H, max_iter=10000, alpha=0.06, exploit_threshold=1e-2, convergence_threshold=1e-6, symmetric=False):
    L = len(psi.shape)
    iter = 0
    converged = False
    # kick the state a bit
    U0, _ = np.linalg.qr(np.random.randn(2, 2))
    for i in range(3):
        U, _ = np.linalg.qr(np.random.randn(2, 2))
        psi = apply_u(U, psi, [i]) if not symmetric else apply_u(U0, psi, [i])
    
    # Es = []
    # psis = []
    while iter <= max_iter and not converged:
        unitaries = []
        E_old = []
        E_new = []
        for i in range(L):
            j = np.mod(i - 1, L)
            k = np.mod(i + 1, L)
            
            dE = np.tensordot(H[i], psi, axes=([4, 5, 3], [1, 2, 0]))
            dE = np.tensordot(psi.conj(), dE, axes=([j, k], [j, k]))
            
            E_old.append(np.trace(dE).real)
            dE = np.eye(2) - alpha * dE / np.linalg.norm(dE)
            # print(dE)
            
            Y, _, Z = np.linalg.svd(dE)
            unitaries.append((Y @ Z).T.conj())
        
        for i in range(L):
            psi = apply_u(unitaries[i], psi, [i])
        
        # for convergence check
        for i in range(L):
            j = np.mod(i - 1, L)
            k = np.mod(i + 1, L)
            dE = np.tensordot(H[i], psi, axes=([4, 5, 3], [1, 2, 0]))
            dE = np.tensordot(psi.conj(), dE, axes=([j, k], [j, k]))
            E_new.append(np.trace(dE).real)
        
        converged = sum([max(E_new[i] - E_old[i], 0) for i in range(len(E_old))]) < convergence_threshold
        iter = iter + 1
        
        # Es.append(E_old)
        # psis.append(psi)
    
    if not converged:
        print(f"Warning: the differential BR dynamics did not converge up to threshold {convergence_threshold}")
    
    # compute exploitability
    epl = np.array([compute_exploitability(psi, H, player) for player in range(len(psi.shape))])
    # if sum(epl) > exploit_threshold:
    #     print(f"Warning: the state found is not a global NE, exploitability {epl}")
    
    results = {
        'symmetric': symmetric,
        'converged': converged,
        'exploitability': epl,
        'energy': E_new,
        'psi': psi,
    }

    return results

def batch_eq_finding(psi, H, max_iter=10000, alpha=0.06, exploit_threshold=1e-2, convergence_threshold=1e-6, symmetric=False, n_trails=100):
    results = {
        'symmetric': symmetric,
        'converged': [],
        'exploitability': [],
        'energy': [],
        'psi': [],
    }
    for _ in range(n_trails):
        result = equilibrium_finding(psi, H, max_iter=max_iter, alpha=alpha, exploit_threshold=exploit_threshold, convergence_threshold=convergence_threshold, symmetric=symmetric)
        results['converged'].append(result['converged'])
        results['exploitability'].append(result['exploitability'])
        results['energy'].append(result['energy'])
        results['psi'].append(result['psi'])
        
    return results


def process_grid(input_grid, H, max_iter=10000, alpha=0.06, exploit_threshold=1e-2, convergence_threshold=1e-6, symmetric=False, n_trails=1, num_checkpoints_per_worker=2, save_dir="qpd3_results"):
    """
    Distribute the evaluation of the equilibrium finding routine over an MPI grid.

    Parameters mirror ``batch_eq_finding`` with the addition of ``input_grid`` which should
    contain the canonical parameters for each initial 3-qubit state (shape: 6 x N).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    input_grid_chunks = np.array_split(input_grid, size)
    worker_chunk = input_grid_chunks[rank]
    
    print(f"Worker {rank} processing {worker_chunk.shape} points")
    worker_results = []
    saved_checkpoints = 0
    for input_params in worker_chunk:
        psi = get_canonical_form_state(input_params)
        # print(psi)
        result = batch_eq_finding(psi, H, max_iter=max_iter, alpha=alpha, exploit_threshold=exploit_threshold, convergence_threshold=convergence_threshold, symmetric=symmetric, n_trails=n_trails)
        worker_results.append(
            {"input_params": input_params, "input_state": psi, "result": result}
        )
        if (len(worker_results)+1) % 10 == 0:
            print(f"Worker {rank} has {len(worker_results)} results")
        if len(worker_results) >= len(worker_chunk) // num_checkpoints_per_worker:
            np.save(os.path.join(save_dir, f"worker_{rank:05d}_checkpoints_{saved_checkpoints:05d}.npy"), worker_results)
            print(f"Worker {rank} saved checkpoint {saved_checkpoints} with {len(worker_results)} results")
            saved_checkpoints += 1
            worker_results = []

def test_grid():
    input_grids = pure_state_grid(n_points=6)
    print(input_grids[1000])
    psi = get_canonical_form_state(input_grids[1000])
    print(psi)




if __name__ == "__main__":
    np.set_printoptions(precision=2)
    H = [np.diag([6, 3, 3, 0, 10, 6, 6, 2]), 
         np.diag([6, 3, 10, 6, 3, 0, 6, 2]), 
         np.diag([6, 10, 3, 6, 3, 6, 0, 2])]
    H = [h.reshape(2, 2, 2, 2, 2, 2) for h in H]
    input_grids = pure_state_grid(n_points=8)
    # print(input_grids)
    
    process_grid(input_grids, H, max_iter=10000, alpha=0.06, exploit_threshold=1e-2, convergence_threshold=1e-8, symmetric=False, n_trails=200, num_checkpoints_per_worker=1, save_dir="qpd3_results_high_sym_points_spacing_8")
