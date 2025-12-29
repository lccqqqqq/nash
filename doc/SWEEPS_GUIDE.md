# W&B Sweeps Guide for Nash Equilibrium Optimization

This guide explains how to use Weights & Biases (W&B) sweeps for hyperparameter optimization.

## What are W&B Sweeps?

W&B sweeps automate hyperparameter search by:
- Running multiple experiments with different parameter combinations
- Using smart search strategies (grid, random, Bayesian)
- Tracking and comparing all runs in one dashboard
- Supporting early termination of poor runs
- Finding optimal hyperparameters automatically

## Quick Start

### 1. Simple Grid Search (Recommended for first time)

```bash
# Run a simple 8-run grid search
python run_sweep.py --config sweep_example_simple.yaml --count 8
```

This will test all combinations of:
- `chi`: [2, 4]
- `eps`: [0.01, 0.02]
- `num_perturbations`: [5, 10]

Total: 2 × 2 × 2 = 8 runs

### 2. Bayesian Optimization (Recommended for real optimization)

```bash
# Run 20 Bayesian optimization runs
python run_sweep.py --config sweep_config.yaml --count 20
```

### 3. Using Python Config (No YAML needed)

```bash
# Create and run sweep from Python config
python run_sweep.py --create-sweep --count 15
```

## Sweep Methods

### Grid Search (`method: grid`)
- **Best for**: Small parameter spaces, exhaustive search
- **Pros**: Tests all combinations, guaranteed to find best in grid
- **Cons**: Exponential growth, expensive for many parameters
- **Use when**: You have 2-4 parameters with 2-5 values each

### Random Search (`method: random`)
- **Best for**: Large parameter spaces, initial exploration
- **Pros**: Better coverage than grid for high dimensions
- **Cons**: No learning from previous runs
- **Use when**: You have many parameters or continuous ranges

### Bayesian Optimization (`method: bayes`)
- **Best for**: Expensive evaluations, finding global optimum
- **Pros**: Learns from previous runs, efficient sampling
- **Cons**: Slower initialization, can get stuck in local optima
- **Use when**: Each run is expensive and you want to minimize total runs

## Sweep Configuration

### YAML Configuration Files

**sweep_example_simple.yaml** - Grid search, quick testing (8 runs)
**sweep_config.yaml** - Bayesian optimization, full search (~20-50 runs)

### Key Configuration Sections

```yaml
method: bayes  # grid, random, or bayes

metric:
  name: welfare  # Metric to optimize
  goal: maximize  # maximize or minimize

parameters:
  # Discrete values
  chi:
    values: [2, 4, 6, 8]

  # Continuous ranges (log-uniform for learning rates)
  eps:
    distribution: log_uniform_values
    min: 0.001
    max: 0.1

  # Fixed parameters
  max_num_steps:
    value: 100
```

### Parameter Distribution Types

1. **Discrete values**: `values: [2, 4, 6, 8]`
2. **Uniform**: `distribution: uniform, min: 0.01, max: 0.1`
3. **Log-uniform** (for learning rates): `distribution: log_uniform_values, min: 0.001, max: 0.1`
4. **Normal**: `distribution: normal, mu: 0.05, sigma: 0.01`
5. **Fixed**: `value: 100`

## Advanced Usage

### Early Termination (Save Compute!)

```yaml
early_terminate:
  type: hyperband
  min_iter: 10  # Minimum iterations before termination
  eta: 2        # Aggressiveness (higher = more aggressive)
```

This stops poorly performing runs early, saving computation time.

### Join Existing Sweep

```bash
# Get sweep ID from W&B dashboard
python run_sweep.py --sweep-id your-sweep-id --count 10
```

Useful for:
- Running sweep on multiple machines in parallel
- Adding more runs to existing sweep
- Resuming interrupted sweeps

### Custom Sweep from Python

Edit `run_sweep.py` and modify `SWEEP_CONFIG` dictionary:

```python
SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {'name': 'welfare', 'goal': 'maximize'},
    'parameters': {
        'chi': {'values': [4, 6, 8]},
        'eps': {'min': 0.01, 'max': 0.05, 'distribution': 'log_uniform_values'},
        # ... add your parameters
    }
}
```

Then run:
```bash
python run_sweep.py --create-sweep --count 20
```

## Parallel Sweeps (Multiple Machines)

### Step 1: Create sweep on first machine
```bash
python run_sweep.py --config sweep_config.yaml --count 5
# Note the sweep ID from output
```

### Step 2: Join sweep from other machines
```bash
# On machine 2
python run_sweep.py --sweep-id your-sweep-id --count 5

# On machine 3
python run_sweep.py --sweep-id your-sweep-id --count 5
```

All machines will work on the same sweep in parallel!

## Monitoring Sweeps

### View in W&B Dashboard
1. Go to https://wandb.ai
2. Navigate to your project
3. Click "Sweeps" tab
4. View:
   - Parallel coordinate plot
   - Parameter importance
   - Best run summary

### Command Line Monitoring
```bash
wandb sweep --help
wandb sweep <sweep-id>  # View sweep details
```

## Tips for Effective Sweeps

### 1. Start Small
```bash
# Test with simple config first
python run_sweep.py --config sweep_example_simple.yaml --count 8
```

### 2. Use Bayesian for Expensive Runs
If each run takes >10 minutes, use `method: bayes` with 20-50 runs.

### 3. Log-Uniform for Learning Rates
Always use `log_uniform_values` for learning rates and similar parameters:
```yaml
eps:
  distribution: log_uniform_values
  min: 0.001
  max: 0.1
```

### 4. Fix Expensive Parameters First
Keep `max_num_steps` low during sweeps, then run best config longer:
```yaml
max_num_steps:
  value: 100  # Use 100 for sweeps, then run best with 1000
```

### 5. Use Early Termination
Enable Hyperband to kill bad runs early:
```yaml
early_terminate:
  type: hyperband
  min_iter: 10
```

## Example Workflow

```bash
# 1. Quick grid search to explore (5 minutes)
python run_sweep.py --config sweep_example_simple.yaml --count 8

# 2. Bayesian optimization to refine (1-2 hours)
python run_sweep.py --config sweep_config.yaml --count 30

# 3. Run best configuration longer
python solver.py \
  --chi 6 \
  --eps 0.025 \
  --num-perturbations 10 \
  --subroutine-lr 0.045 \
  --max-num-steps 1000 \
  --use-wandb
```

## Metrics Logged

All metrics from `solver.py` are automatically logged:
- `welfare` - Total welfare (primary optimization target)
- `energy/player_0`, `energy/player_1`, `energy/player_2` - Individual energies
- `ent_params/I1` through `ent_params/I5` - Entanglement parameters

You can optimize for any of these by changing the `metric.name` in your config.

## Troubleshooting

### Sweep not starting?
```bash
# Check W&B login
wandb login

# Verify config
python -c "import yaml; print(yaml.safe_load(open('sweep_config.yaml')))"
```

### Runs failing?
- Check that all parameter names match CLI arguments in `solver.py`
- Verify parameter ranges are reasonable
- Test single run first: `python solver.py --use-wandb`

### Too many runs?
- Use `method: grid` to count exact runs: product of all value lists
- Start with `--count 5` to test
- Use early termination to save compute

## Further Reading

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Sweep Configuration Options](https://docs.wandb.ai/guides/sweeps/configuration)
- [Hyperband Early Termination](https://docs.wandb.ai/guides/sweeps/early-termination)
