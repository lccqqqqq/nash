"""
Run W&B sweeps for hyperparameter optimization.

Usage:
    # Initialize sweep from YAML config
    python run_sweep.py --config sweep_config.yaml --count 20

    # Or create sweep programmatically
    python run_sweep.py --create-sweep --count 20
"""

import wandb
import argparse
import yaml
import numpy as np
from mps_utils import get_rand_mps
from game import get_default_H
from solver import opt_fid_state


# Python-based sweep configuration (alternative to YAML)
# Bayesian optimization config - uses continuous distributions
SWEEP_CONFIG = {
    'method': 'bayes',  # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'welfare',
        'goal': 'maximize'
    },
    'parameters': {
        # State initialization
        'chi': {'value': 8},
        'num_players': {'value': 5},  # Support both 3 and 4 player games
        'seed': {'values': [42, 123, 456, 789, 1337]},  # Different initializations

        # Optimization parameters
        'eps': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.1
        },
        'num_perturbations': {'value': 20},
        'perturbation_method': {'value': 'unitary'},

        # Nash equilibrium subroutine
        'subroutine_lr': {
            'distribution': 'log_uniform_values',
            'min': 0.01,
            'max': 0.6
        },
        'subroutine_max_iter': {'value': 1000},

        # Fixed parameters
        'max_num_steps': {'value': 1000},
        'wandb_log_interval': {'value': 20},  # Log every 20 steps for efficiency
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10,
        'eta': 2
    }
}

# Grid search config - uses discrete values only
# To use this, change the main() function to use SWEEP_CONFIG_GRID instead
SWEEP_CONFIG_GRID = {
    'method': 'grid',
    'metric': {
        'name': 'welfare',
        'goal': 'maximize'
    },
    'parameters': {
        # State initialization
        'chi': {'value': 32},
        'num_players': {'value': 3},
        'seed': {'values': [42, 123, 456]},  # 3 seeds for grid

        # Optimization parameters - manually discretized
        'eps': {'values': [0.001, 0.003, 0.007, 0.01, 0.03, 0.07, 0.1]},  # 5 values (log-spaced)
        'num_perturbations': {'value': 20},
        'perturbation_method': {'value': 'unitary'},

        # Nash equilibrium subroutine - manually discretized
        'subroutine_lr': {'values': [0.01, 0.03, 0.06, 0.1, 0.3]},  # 5 values (log-spaced)
        'subroutine_max_iter': {'value': 1000},

        # Fixed parameters
        'max_num_steps': {'value': 1000},
        'wandb_log_interval': {'value': 20},  # Log every 20 steps for efficiency
    },
    # Note: Grid search doesn't use early termination
}


def train_sweep():
    """
    Training function for W&B sweep.
    This function is called by wandb.agent() for each sweep run.
    """
    # Initialize wandb run - config will be set by sweep
    with wandb.init() as run:
        # Get hyperparameters from sweep config
        config = wandb.config

        print(f"\n{'='*60}")
        print(f"Starting sweep run: {run.name}")
        print(f"Sweep ID: {run.sweep_id}")
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        # Set random seed if provided
        seed = config.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        # Initialize state and Hamiltonian
        num_players = config.get('num_players', 3)
        chi = config.get('chi', 4)
        Psi = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=seed)
        H = get_default_H(num_players=num_players)

        # Run optimization
        Psi, metric_logs = opt_fid_state(
            Psi, H,
            max_num_steps=config.get('max_num_steps', 100),
            eps=config.get('eps', 0.01),
            num_perturbations=config.get('num_perturbations', 10),
            perturbation_method=config.get('perturbation_method', 'unitary'),
            subroutine_max_iter=config.get('subroutine_max_iter', 1000),
            subroutine_lr=config.get('subroutine_lr', 0.03),
            use_wandb=True,  # Always true for sweeps
            wandb_project=config.get('wandb_project', 'nash-equilibrium'),
            wandb_config={'sweep_run': True, 'seed': seed},
            wandb_log_interval=config.get('wandb_log_interval', 20),  # Log every 20 steps by default for sweeps
            should_save_results=config.get('save_results', True),
            save_dir=config.get('save_dir', 'data'),
            seed=seed,  # Pass seed for metadata tracking
        )

        print(f"\nCompleted sweep run: {run.name}\n")


def main():
    parser = argparse.ArgumentParser(description='Run W&B sweeps for Nash equilibrium optimization')

    # Sweep initialization options
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML sweep configuration file')
    parser.add_argument('--create-sweep', action='store_true',
                        help='Create sweep using Python config instead of YAML')
    parser.add_argument('--method', type=str, choices=['bayes', 'grid'], default='bayes',
                        help='Sweep method: bayes (default) or grid')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing sweep ID to join (if not creating new)')

    # Agent options
    parser.add_argument('--count', type=int, default=10,
                        help='Number of sweep runs to execute')
    parser.add_argument('--project', type=str, default='nash-equilibrium',
                        help='W&B project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='W&B entity (team) name')

    args = parser.parse_args()

    # Initialize or join sweep
    if args.sweep_id:
        # Join existing sweep
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
    else:
        # Create new sweep
        if args.config:
            # Load from YAML file
            print(f"Creating sweep from config file: {args.config}")
            with open(args.config, 'r') as f:
                sweep_config = yaml.safe_load(f)
        elif args.create_sweep:
            # Use Python config - select based on method
            if args.method == 'grid':
                print("Creating GRID sweep from Python configuration")
                print(f"  Total runs: {3 * 5 * 5} (3 seeds × 5 eps × 5 subroutine_lr)")
                sweep_config = SWEEP_CONFIG_GRID
            else:
                print("Creating BAYES sweep from Python configuration")
                sweep_config = SWEEP_CONFIG
        else:
            print("Error: Must specify either --config, --create-sweep, or --sweep-id")
            return

        # Initialize sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity
        )
        print(f"Created sweep with ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/{args.entity or 'your-entity'}/{args.project}/sweeps/{sweep_id}")

    # Run sweep agent
    print(f"\nStarting sweep agent with {args.count} runs...")
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=args.count,
        project=args.project,
        entity=args.entity
    )

    print(f"\nSweep completed! View results at W&B dashboard.")


if __name__ == "__main__":
    main()
