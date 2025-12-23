# Wandb Sweep Methods Guide

## Quick Answer: Grid vs Distribution Parameters

**NO** - Grid search does NOT work with `distribution` parameters like `log_uniform_values`.

| Sweep Method | Distribution Params | Discrete Values | Use Case |
|--------------|---------------------|-----------------|----------|
| `grid` | ❌ Not supported | ✅ Required (`values: [...]`) | Exhaustive search over small parameter space |
| `random` | ✅ Supported | ✅ Supported | Random sampling from large space |
| `bayes` | ✅ Supported | ✅ Supported | Smart optimization of expensive functions |

## Available Configurations

### 1. Bayesian Optimization (Default - `SWEEP_CONFIG`)
Uses continuous distributions for efficient exploration.

**Parameters:**
- `seed`: [42, 123, 456, 789, 1337] (5 discrete values)
- `eps`: log-uniform from 0.001 to 0.1 (continuous)
- `subroutine_lr`: log-uniform from 0.01 to 0.6 (continuous)

**Pros:**
- Explores continuous parameter space
- Efficient for expensive optimizations
- Learns from previous runs

**Cons:**
- Requires many runs to converge
- Results not fully reproducible (stochastic)

### 2. Grid Search (`SWEEP_CONFIG_GRID`)
Exhaustive search over discrete parameter grid.

**Parameters:**
- `seed`: [42, 123, 456] (3 values)
- `eps`: [0.001, 0.003, 0.01, 0.03, 0.1] (5 values, log-spaced)
- `subroutine_lr`: [0.01, 0.03, 0.1, 0.3, 0.6] (5 values, log-spaced)

**Total runs:** 3 × 5 × 5 = **75 runs**

**Pros:**
- Tests all combinations systematically
- Fully reproducible
- No randomness (except in optimization itself)

**Cons:**
- Combinatorial explosion (75 runs for this config!)
- Can't explore continuous ranges
- Expensive for large parameter spaces

## Usage

### Bayesian Optimization
```bash
# Default - uses SWEEP_CONFIG
python run_sweep.py --create-sweep --count 50

# Explicit
python run_sweep.py --create-sweep --method bayes --count 50
```

### Grid Search
```bash
# All 75 combinations
python run_sweep.py --create-sweep --method grid --count 75

# Note: --count should equal total grid size for exhaustive search
```

### From YAML File
```bash
python run_sweep.py --config my_sweep.yaml --count 20
```

## Converting Distribution to Grid

If you want to use grid search, you must manually discretize continuous parameters:

**Before (Bayes/Random):**
```python
'eps': {
    'distribution': 'log_uniform_values',
    'min': 0.001,
    'max': 0.1
}
```

**After (Grid):**
```python
'eps': {
    'values': [0.001, 0.003, 0.01, 0.03, 0.1]  # Log-spaced
}
# Or linear-spaced:
# 'values': [0.001, 0.026, 0.051, 0.076, 0.1]
```

### Generating Log-Spaced Values in Python
```python
import numpy as np

# Generate 5 log-spaced values between 0.001 and 0.1
values = np.logspace(-3, -1, 5).tolist()
# [0.001, 0.00316..., 0.01, 0.0316..., 0.1]

# Round for cleaner config
values = [round(v, 4) for v in values]
# [0.001, 0.0032, 0.01, 0.0316, 0.1]
```

## Recommendations

### Use Bayesian when:
- Parameter space is large or continuous
- Budget is limited (can't run full grid)
- Want to find good hyperparameters efficiently

### Use Grid when:
- Parameter space is small (< 100 combinations)
- Need complete coverage of parameter space
- Want fully reproducible results
- Testing specific hypotheses about parameter interactions

### Use Random when:
- Just want to explore broadly
- Have no prior about parameter importance
- Want simple random baseline

## Wandb Logging Efficiency

To reduce wandb overhead for long runs, both configs use `wandb_log_interval: 20`:
- Logs at step 0, 20, 40, 60, ...
- **Always logs the last step** (even if not a multiple of 20)
- Reduces log volume by 20x for 1000-step runs
- Can be customized: set `wandb_log_interval: 1` for every-step logging

**Example:** For a 1000-step run:
- `wandb_log_interval=1`: 1000 logs
- `wandb_log_interval=20`: 51 logs (0, 20, 40, ..., 980, 999)
- `wandb_log_interval=50`: 21 logs (0, 50, 100, ..., 950, 999)

## Current Config Comparison

| Aspect | Bayes Config | Grid Config |
|--------|-------------|-------------|
| Seeds | 5 values | 3 values |
| eps | Continuous (∞) | 7 values |
| subroutine_lr | Continuous (∞) | 5 values |
| **Total grid size** | Unlimited | **105** (3×7×5) |
| **Recommended count** | 50-100 | 105 |
| **Wandb log interval** | 20 | 20 |

## Troubleshooting

### "Grid search requires discrete values" error
**Problem:** Used `distribution` parameter with `method: 'grid'`

**Solution:** Convert all `distribution` params to `values` lists:
```python
# Change this:
'eps': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.1}

# To this:
'eps': {'values': [0.001, 0.003, 0.01, 0.03, 0.1]}
```

### Grid search seems stuck
**Problem:** Grid search needs exactly `total_combinations` runs to complete

**Solution:** Set `--count` to match grid size:
```bash
# For 3 × 5 × 5 = 75 grid
python run_sweep.py --create-sweep --method grid --count 75
```
