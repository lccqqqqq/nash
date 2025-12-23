# Changelog: Wandb Log Interval Feature

## Summary

Added `wandb_log_interval` parameter to reduce wandb logging overhead for long-running optimizations. Default for sweeps is now **20 steps** instead of every step.

## Changes

### 1. **solver.py**
- **Line 844**: Added `wandb_log_interval: int = 1` parameter to `opt_fid_state()`
- **Line 954**: Modified logging condition to respect interval:
  ```python
  should_log = (i % wandb_log_interval == 0) or (i == max_num_steps - 1)
  if use_wandb and should_log:
      wandb.log(wandb_metrics, step=i)
  ```

### 2. **run_sweep.py**
- **Line 54**: Added `'wandb_log_interval': {'value': 20}` to `SWEEP_CONFIG` (Bayesian)
- **Line 88**: Added `'wandb_log_interval': {'value': 20}` to `SWEEP_CONFIG_GRID` (Grid)
- **Line 133**: Pass interval to `opt_fid_state()`:
  ```python
  wandb_log_interval=config.get('wandb_log_interval', 20)
  ```

### 3. **SWEEP_GUIDE.md**
- Added "Wandb Logging Efficiency" section explaining the feature
- Updated config comparison table to show log interval setting

### 4. **Test Files**
- Created `test_log_interval_simple.py` to verify logic
- Tests confirm correct behavior for various scenarios

## Behavior

### Logging Schedule

For `wandb_log_interval=20` with 1000 steps:
- **Logs at**: 0, 20, 40, 60, 80, ..., 980, **999** (last step always logged)
- **Total logs**: 51 (instead of 1000)
- **Reduction**: ~20x fewer logs

### Key Features

1. **Always logs first step** (i=0) - important for initialization tracking
2. **Always logs last step** (i=max_num_steps-1) - critical for final results
3. **Regular intervals** (i % log_interval == 0) - consistent spacing
4. **Configurable** - can set to 1 for every-step logging if needed

## Impact

### Before (wandb_log_interval=1)
```python
# 1000-step run
Total wandb logs: 1000
Network overhead: High
Wandb dashboard: Cluttered with data points
```

### After (wandb_log_interval=20)
```python
# 1000-step run
Total wandb logs: 51
Network overhead: 20x reduced
Wandb dashboard: Cleaner, still captures trends
File saving: Still saves all 1000 iterations in metric_logs
```

## Usage

### In Sweeps (Automatic)
```bash
python run_sweep.py --create-sweep --count 50
# Uses wandb_log_interval=20 by default
```

### Manual Override
```python
# In YAML config
parameters:
  wandb_log_interval:
    value: 10  # Log every 10 steps

# Or in Python
from solver import opt_fid_state
Psi, logs = opt_fid_state(
    Psi, H,
    wandb_log_interval=10,  # Custom interval
    use_wandb=True
)
```

### Disable Interval (Log Every Step)
```python
wandb_log_interval=1  # Back to original behavior
```

## Examples

| Scenario | Steps | Interval | Logs | Reduction |
|----------|-------|----------|------|-----------|
| Short run | 100 | 20 | 6 | 16.7x |
| Medium run | 500 | 20 | 26 | 19.2x |
| Long run | 1000 | 20 | 51 | 19.6x |
| Very long run | 5000 | 50 | 101 | 49.5x |

## Backward Compatibility

✅ **Fully backward compatible**
- Default is `wandb_log_interval=1` (log every step)
- Only sweep configs use interval=20
- Old code continues to work unchanged
- All 1000 iterations still saved in `metric_logs` (only wandb logging affected)

## Testing

Run tests to verify:
```bash
python test_log_interval_simple.py
```

Output confirms:
- ✅ Logs at step 0
- ✅ Logs every N steps
- ✅ Always logs last step
- ✅ Reduces log count by ~Nx

## Benefits

1. **Reduced network overhead**: 20x fewer wandb API calls
2. **Faster runs**: Less time spent logging
3. **Cleaner dashboards**: Better signal-to-noise ratio in plots
4. **Same data quality**: Final results unaffected, intermediate points still captured
5. **Configurable**: Easy to adjust based on run length

## Notes

- This only affects **wandb logging** (what gets sent to cloud)
- **Local `metric_logs`** still contains all iterations
- **File saving** (CSV/pickle) saves all iterations unchanged
- **Last step always logged** ensures final metrics are captured
