#!/usr/bin/env python3
"""
Near-QPD 2-Player Nash Equilibrium Experiment

This script tests Nash equilibrium existence and welfare in 2-player quantum games
as a function of:
- Initial state entanglement (Schmidt eigenvalue λ₁)
- Hamiltonian non-commutativity (perturbation strength λ)
- Perturbation direction (random seed for V in H = h0 + λ·V)

Usage:
    python experiments/near_qpd_2player.py \
        --schmidt-lambda 0.95 \
        --noncomm-lambda 0.1 \
        --pert-seed 1000 \
        --basis-seed 1 \
        --save-dir data/near_qpd_2player_experiment/test \
        --expl-threshold 1e-4
"""

import os
import sys
import argparse
import pickle
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mps_utils import get_low_entanglement_2qubit_state, to_comp_basis
from game import get_fixed_perturbation_hamiltonian_2player
from solver import find_nash_eq1


def main():
    parser = argparse.ArgumentParser(
        description='Near-QPD 2-Player Nash Equilibrium Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required parameters
    parser.add_argument('--schmidt-lambda', type=float, required=True,
                        help='Schmidt eigenvalue λ₁ ∈ [1/√2, 1] (1=product, 0.707=Bell)')
    parser.add_argument('--noncomm-lambda', type=float, required=True,
                        help='Non-commutativity strength λ in H = h0 + λ·V')
    parser.add_argument('--pert-seed', type=int, required=True,
                        help='Perturbation seed (determines V direction)')
    parser.add_argument('--basis-seed', type=int, required=True,
                        help='Basis randomization seed (random Schmidt basis orientation)')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save results')

    # Nash equilibrium parameters
    parser.add_argument('--expl-threshold', type=float, default=1e-4,
                        help='Exploitability threshold for convergence (lower than default 5e-4)')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Maximum Nash equilibrium iterations')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Nash equilibrium learning rate')
    parser.add_argument('--expl-check-interval', type=int, default=50,
                        help='Check exploitability every N iterations')
    parser.add_argument('--expl-maxiter', type=int, default=60,
                        help='Max iterations for exploitability computation (differential evolution)')

    # Optional parameters
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'],
                        help='Data type for computations')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress information')

    args = parser.parse_args()

    # Setup
    dtype = np.float64 if args.dtype == 'float64' else np.float32
    os.makedirs(args.save_dir, exist_ok=True)

    if args.verbose:
        print("="*70)
        print("Near-QPD 2-Player Nash Equilibrium Experiment")
        print("="*70)
        print(f"Schmidt λ₁:        {args.schmidt_lambda:.4f}")
        print(f"Noncomm λ:         {args.noncomm_lambda:.4f}")
        print(f"Pert seed:         {args.pert_seed}")
        print(f"Basis seed:        {args.basis_seed}")
        print(f"Expl threshold:    {args.expl_threshold:.1e}")
        print("="*70)

    # Step 1: Initialize 2-qubit state with specified Schmidt eigenvalue
    if args.verbose:
        print("\n[1/5] Initializing 2-qubit state...")

    Psi = get_low_entanglement_2qubit_state(
        lambda1=args.schmidt_lambda,
        randomize_basis=True,
        seed=args.basis_seed,
        dtype=dtype
    )

    # Compute initial state properties
    psi_init = to_comp_basis(Psi)

    if args.verbose:
        print(f"  ✓ Created state with λ₁={args.schmidt_lambda:.4f}")
        print(f"  ✓ MPS bond dimensions: {[A.shape for A in Psi]}")
        print(f"  ✓ State norm: {np.linalg.norm(psi_init):.6f}")

    # Step 2: Generate Hamiltonian H = h0 + λ·V
    if args.verbose:
        print("\n[2/5] Generating perturbed Hamiltonian...")

    H = get_fixed_perturbation_hamiltonian_2player(
        lambda_noncomm=args.noncomm_lambda,
        perturbation_seed=args.pert_seed,
        dtype=dtype,
        cache=None  # No cache needed for single run
    )

    if args.verbose:
        print(f"  ✓ Generated H = h0 + {args.noncomm_lambda:.4f}·V")
        print(f"  ✓ Hamiltonian shapes: {[h.shape for h in H]}")

    # Step 3: Find Nash equilibrium using find_nash_eq1
    if args.verbose:
        print("\n[3/5] Finding Nash equilibrium...")
        print(f"  Max iterations: {args.max_iter}")
        print(f"  Alpha: {args.alpha}")
        print(f"  Expl threshold: {args.expl_threshold:.1e}")

    result = find_nash_eq1(
        Psi=Psi,
        H=H,
        max_iter=args.max_iter,
        alpha=args.alpha,
        convergence_threshold=1e-7,
        expl_threshold=args.expl_threshold,
        use_tqdm=args.verbose,
        expl_check_interval=args.expl_check_interval,
        return_history=True,  # Save full trajectory
        expl_maxiter=args.expl_maxiter,
        expl_seed=42,
        real_strategies=True
    )

    if args.verbose:
        converged_str = "✓ CONVERGED" if result['nash_equilibrium'] else "✗ NOT CONVERGED"
        print(f"\n  {converged_str}")
        print(f"  Iterations: {result['num_iters']}")
        if len(result['expl']) > 0:
            final_expl = result['expl'][-1] if isinstance(result['expl'], np.ndarray) else result['expl']
            if isinstance(final_expl, (list, np.ndarray)):
                print(f"  Final exploitability: {np.sum(final_expl):.2e} (players: {final_expl})")
            else:
                print(f"  Final exploitability: {final_expl:.2e}")

    # Step 4: Post-process results
    if args.verbose:
        print("\n[4/5] Computing welfare and statistics...")

    # Compute welfare history (sum of player energies)
    if result['energy'].ndim == 2:
        # Shape: (num_steps, num_players)
        welfare_history = result['energy'].sum(axis=-1)
        final_welfare = welfare_history[-1]
        final_energy_per_player = result['energy'][-1]
    else:
        # Shape: (num_players,) - only final energy
        final_energy_per_player = result['energy']
        final_welfare = result['energy'].sum()
        welfare_history = None

    # Extract exploitability history
    if isinstance(result['expl'], np.ndarray) and result['expl'].ndim == 2:
        # Shape: (num_checks, num_players)
        expl_history = result['expl']
        final_expl_per_player = result['expl'][-1]
        final_total_expl = expl_history[-1].sum()
    elif isinstance(result['expl'], (list, np.ndarray)):
        final_expl_per_player = np.array(result['expl']) if isinstance(result['expl'], list) else result['expl']
        final_total_expl = final_expl_per_player.sum() if hasattr(final_expl_per_player, 'sum') else sum(final_expl_per_player)
        expl_history = None
    else:
        final_expl_per_player = result['expl']
        final_total_expl = result['expl']
        expl_history = None

    if args.verbose:
        print(f"  Final welfare (E₁+E₂): {final_welfare:.6f}")
        print(f"  Final energy per player: {final_energy_per_player}")
        print(f"  ✓ Computed {len(result['state']) if isinstance(result['state'], list) else 1} state snapshots")

    # Step 5: Save results
    if args.verbose:
        print("\n[5/5] Saving results...")

    # Prepare output dictionary
    output = {
        # Convergence info
        'converged': result['nash_equilibrium'],
        'nash_state': result['nash_state'],
        'num_iters': result['num_iters'],

        # Energy and welfare
        'energy_history': result['energy'],
        'welfare_history': welfare_history,
        'final_energy_per_player': final_energy_per_player,
        'final_welfare': final_welfare,

        # Exploitability
        'expl_history': expl_history,
        'final_expl_per_player': final_expl_per_player,
        'final_total_expl': final_total_expl,

        # State history (MPS format)
        'state_history_mps': result['state_'],
        'state_history_comp': result['state'],
        'initial_state': psi_init,

        # Metadata
        'metadata': {
            'schmidt_lambda': args.schmidt_lambda,
            'noncomm_lambda': args.noncomm_lambda,
            'pert_seed': args.pert_seed,
            'basis_seed': args.basis_seed,
            'expl_threshold': args.expl_threshold,
            'max_iter': args.max_iter,
            'alpha': args.alpha,
            'dtype': str(dtype),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }
    }

    # Generate filename
    filename = f"result_s{args.schmidt_lambda:.2f}_nc{args.noncomm_lambda:.3f}_ps{args.pert_seed}_bs{args.basis_seed}.pkl"
    filepath = os.path.join(args.save_dir, filename)

    # Save
    with open(filepath, 'wb') as f:
        pickle.dump(output, f)

    if args.verbose:
        print(f"  ✓ Saved to: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
        print("\n" + "="*70)
        print("Experiment completed successfully!")
        print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
