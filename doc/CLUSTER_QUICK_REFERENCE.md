# Cluster Sweep Quick Reference

One-page cheat sheet for running W&B sweeps on your cluster.

## The Three Commands

```bash
# 1. Setup sweep
python setup_cluster_sweep.py --config sweep_config.yaml --name "my-exp" --num-workers 20

# 2. Submit to cluster
multirun commands.txt

# 3. Monitor (URL shown in step 1 output)
```

## Common Configurations

### Quick Test (8 runs, ~5 min)
```bash
python setup_cluster_sweep.py \
    --config sweep_example_simple.yaml \
    --num-workers 4 --count-per-worker 2
```

### Standard Run (50 runs, ~2 hrs)
```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --num-workers 10 --count-per-worker 5
```

### Large Scale (200 runs, overnight)
```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --num-workers 20 --count-per-worker 10
```

## File Reference

| File | Purpose |
|------|---------|
| `sweep_config.yaml` | Full Bayesian optimization config |
| `sweep_example_simple.yaml` | Quick grid search (8 runs) |
| `setup_cluster_sweep.py` | Creates sweep + generates commands |
| `run_sweep.py` | Runs sweep agent (called by workers) |
| `commands.txt` | Generated commands for multirun |
| `sweep_id.txt` | Saved sweep ID for reference |

## Command Options

```bash
python setup_cluster_sweep.py \
    --config <yaml>          # Required: sweep config file
    --name <string>          # Optional: sweep name
    --num-workers <int>      # Default: 10
    --count-per-worker <int> # Default: 10
    --project <string>       # Default: nash-equilibrium
    --entity <string>        # Optional: W&B team name
    --output <file>          # Default: commands.txt
```

## Choosing Worker Counts

| Scenario | Workers | Runs/Worker | Total |
|----------|---------|-------------|-------|
| Quick test | 4 | 2 | 8 |
| Medium sweep | 10 | 5 | 50 |
| Large sweep | 20 | 10 | 200 |
| Few powerful nodes | 5 | 20 | 100 |
| Many light nodes | 50 | 2 | 100 |

**Rule of thumb:**
- Total runs = workers × runs/worker
- Use 70-80% of available cores
- Leave some cores for system processes

## Monitoring

### W&B Dashboard
```
https://wandb.ai/USERNAME/PROJECT/sweeps/SWEEP_ID
```

View:
- Best parameters found so far
- Parallel coordinates plot
- Parameter importance
- All run details

### Command Line
```bash
# View sweep info
wandb sweep SWEEP_ID

# Check saved results
ls -lh data/

# Read sweep ID from file
cat sweep_id.txt
```

## Adding More Workers to Existing Sweep

```bash
# Get sweep ID
SWEEP_ID=$(cat sweep_id.txt)

# Generate more commands
for i in {1..5}; do
    echo "python run_sweep.py --sweep-id $SWEEP_ID --count 10" >> more_commands.txt
done

# Submit
multirun more_commands.txt
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Config file not found" | Check file exists: `ls sweep_config.yaml` |
| "Authentication required" | Login: `wandb login` |
| Workers not starting | Check paths in `commands.txt` |
| Sweep finishes too fast | Grid search has limited configs; use Bayesian |
| Out of memory | Reduce `--num-workers` |

## File Outputs

Each run saves to `data/`:
```
data/opt_fid_state_chi4_lr1e-02_steps100_alpha3e-02_TIMESTAMP.csv
```

Contains:
- `welfare` - Total welfare
- `energy_player_0/1/2` - Individual energies
- `I1/I2/I3/I4/I5` - Entanglement parameters

## Parameter Ranges (Default Config)

| Parameter | Range/Values | Type |
|-----------|--------------|------|
| `chi` | [2, 4, 6, 8] | Discrete |
| `eps` | [0.001, 0.1] | Log-uniform |
| `num_perturbations` | [3, 5, 10, 15] | Discrete |
| `subroutine_lr` | [0.01, 0.1] | Log-uniform |
| `subroutine_max_iter` | [500, 1000, 2000] | Discrete |

## Complete Example

```bash
# 1. Create sweep
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --name "baseline-optimization-2025-01-17" \
    --num-workers 20 \
    --count-per-worker 5

# Output shows:
# ✓ Sweep created: a1b2c3d4
# ✓ URL: https://wandb.ai/user/nash-equilibrium/sweeps/a1b2c3d4
# ✓ Generated commands.txt

# 2. Review (optional)
cat commands.txt

# 3. Submit to cluster
multirun commands.txt

# 4. Monitor progress
# Open URL from step 1 in browser

# 5. Later: check results
ls data/*.csv
```

## Tips

✓ Always use `--name` for easy sweep identification
✓ Start with `sweep_example_simple.yaml` to test
✓ Use `--count-per-worker` to control total runs
✓ Save sweep_id.txt for future reference
✓ Monitor resource usage on cluster nodes
✓ Grid search = finite configs, Bayesian = infinite

## Getting Help

```bash
# Script help
python setup_cluster_sweep.py --help

# Full documentation
cat CLUSTER_SWEEP_TUTORIAL.md
cat SWEEPS_GUIDE.md

# W&B docs
https://docs.wandb.ai/guides/sweeps
```
