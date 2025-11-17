#!/usr/bin/env python
"""
Create a W&B sweep and generate commands file for cluster multirun.

This script:
1. Creates a W&B sweep from a YAML config
2. Saves the sweep ID to a file
3. Generates a commands.txt file for your cluster's multirun system

Usage:
    # Basic usage
    python setup_cluster_sweep.py --config sweep_config.yaml

    # With custom name and workers
    python setup_cluster_sweep.py \
        --config sweep_config.yaml \
        --name "high-chi-experiment-v2" \
        --num-workers 20 \
        --count-per-worker 5

    # With entity (for team accounts)
    python setup_cluster_sweep.py \
        --config sweep_config.yaml \
        --name "baseline-test" \
        --project nash-equilibrium \
        --entity my-team \
        --num-workers 10 \
        --count-per-worker 10
"""

import wandb
import yaml
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Setup W&B sweep for cluster multirun',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Sweep configuration
    parser.add_argument('--config', required=True,
                        help='Path to sweep config YAML file')
    parser.add_argument('--name', default=None,
                        help='Custom sweep name (shown in W&B dashboard)')
    parser.add_argument('--project', default='nash-equilibrium',
                        help='W&B project name')
    parser.add_argument('--entity', default=None,
                        help='W&B entity (team) name')

    # Cluster configuration
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of parallel workers/machines')
    parser.add_argument('--count-per-worker', type=int, default=10,
                        help='Number of runs each worker should execute')

    # Output files
    parser.add_argument('--output', default='commands.txt',
                        help='Output file for multirun commands')
    parser.add_argument('--save-id', default='sweep_id.txt',
                        help='File to save sweep ID')

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found!")
        sys.exit(1)

    # Load sweep config
    print(f"Loading sweep configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            sweep_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Override name if provided via CLI
    if args.name:
        sweep_config['name'] = args.name
        print(f"Setting sweep name to: {args.name}")

    # Display sweep configuration summary
    print("\nSweep Configuration:")
    print(f"  Method: {sweep_config.get('method', 'N/A')}")
    print(f"  Metric: {sweep_config.get('metric', {}).get('name', 'N/A')} "
          f"({sweep_config.get('metric', {}).get('goal', 'N/A')})")

    # Count parameters being swept
    params = sweep_config.get('parameters', {})
    swept_params = [k for k, v in params.items() if 'value' not in v]
    fixed_params = [k for k, v in params.items() if 'value' in v]
    print(f"  Swept parameters: {len(swept_params)} {swept_params}")
    print(f"  Fixed parameters: {len(fixed_params)}")

    # Create sweep
    print(f"\nCreating sweep in project '{args.project}'...")
    try:
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity
        )
    except Exception as e:
        print(f"Error creating sweep: {e}")
        print("\nMake sure you're logged in to W&B:")
        print("  wandb login")
        sys.exit(1)

    # Build sweep URL
    entity_str = args.entity if args.entity else "YOUR-USERNAME"
    sweep_url = f"https://wandb.ai/{entity_str}/{args.project}/sweeps/{sweep_id}"

    # Print success message
    print(f"\n{'='*70}")
    print(f"✓ Sweep created successfully!")
    print(f"{'='*70}")
    print(f"Sweep ID:   {sweep_id}")
    print(f"Sweep Name: {sweep_config.get('name', 'Unnamed')}")
    print(f"Project:    {args.project}")
    if args.entity:
        print(f"Entity:     {args.entity}")
    print(f"URL:        {sweep_url}")
    print(f"{'='*70}\n")

    # Save sweep ID to file
    try:
        with open(args.save_id, 'w') as f:
            f.write(sweep_id)
        print(f"✓ Sweep ID saved to: {args.save_id}")
    except Exception as e:
        print(f"Warning: Could not save sweep ID to file: {e}")

    # Generate commands file for cluster
    print(f"✓ Generating commands file: {args.output}")
    try:
        with open(args.output, 'w') as f:
            for i in range(args.num_workers):
                cmd = f"/usr/bin/python3 run_sweep.py --sweep-id {sweep_id} --count {args.count_per_worker}"
                if args.project != 'nash-equilibrium':
                    cmd += f" --project {args.project}"
                if args.entity:
                    cmd += f" --entity {args.entity}"
                f.write(cmd + "\n")
    except Exception as e:
        print(f"Error generating commands file: {e}")
        sys.exit(1)

    total_runs = args.num_workers * args.count_per_worker
    print(f"  Workers: {args.num_workers}")
    print(f"  Runs per worker: {args.count_per_worker}")
    print(f"  Total runs: up to {total_runs}")

    # Print commands file preview
    print(f"\nCommands file preview ({args.output}):")
    print("-" * 70)
    with open(args.output, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:3], 1):
            print(f"  {i}. {line.strip()}")
        if len(lines) > 3:
            print(f"  ... ({len(lines) - 3} more lines)")
    print("-" * 70)

    # Print next steps
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print(f"1. Review the commands file:")
    print(f"   cat {args.output}")
    print(f"\n2. Submit to your cluster's multirun system:")
    print(f"   multirun {args.output}")
    print(f"\n3. Monitor sweep progress:")
    print(f"   {sweep_url}")
    print(f"\n4. (Optional) Join sweep from another machine:")
    print(f"   python run_sweep.py --sweep-id {sweep_id} --count 10")
    print("="*70)

    # Print sweep ID one more time for easy copying
    print(f"\nSweep ID (for reference): {sweep_id}")


if __name__ == '__main__':
    main()
