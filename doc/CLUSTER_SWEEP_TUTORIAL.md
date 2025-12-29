# Cluster Sweep Tutorial

Complete guide for running W&B sweeps on your cluster using the multirun system.

## Quick Start (3 Commands)

```bash
# 1. Create sweep and generate commands
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --name "my-experiment" \
    --num-workers 20 \
    --count-per-worker 5

# 2. Submit to cluster
multirun commands.txt

# 3. Monitor at URL shown in output
```

That's it! Your sweep is running on 20 machines in parallel.

## Complete Walkthrough

### Step 1: Prepare Your Sweep Configuration

Choose a config file based on your needs:

**For quick testing (8 runs):**
```bash
--config sweep_example_simple.yaml
```

**For full optimization (Bayesian, ~50 runs):**
```bash
--config sweep_config.yaml
```

**Or create your own** (see SWEEPS_GUIDE.md)

### Step 2: Create Sweep and Generate Commands

```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --name "baseline-chi-sweep-v1" \
    --num-workers 10 \
    --count-per-worker 5
```

**Output:**
```
Loading sweep configuration from: sweep_config.yaml
Setting sweep name to: baseline-chi-sweep-v1

Sweep Configuration:
  Method: bayes
  Metric: welfare (maximize)
  Swept parameters: 5 ['chi', 'eps', 'num_perturbations', 'subroutine_lr', 'subroutine_max_iter']
  Fixed parameters: 4

Creating sweep in project 'nash-equilibrium'...

======================================================================
✓ Sweep created successfully!
======================================================================
Sweep ID:   a1b2c3d4
Sweep Name: baseline-chi-sweep-v1
Project:    nash-equilibrium
URL:        https://wandb.ai/YOUR-USERNAME/nash-equilibrium/sweeps/a1b2c3d4
======================================================================

✓ Sweep ID saved to: sweep_id.txt
✓ Generating commands file: commands.txt
  Workers: 10
  Runs per worker: 5
  Total runs: up to 50

Commands file preview (commands.txt):
----------------------------------------------------------------------
  1. python run_sweep.py --sweep-id a1b2c3d4 --count 5
  2. python run_sweep.py --sweep-id a1b2c3d4 --count 5
  3. python run_sweep.py --sweep-id a1b2c3d4 --count 5
  ... (7 more lines)
----------------------------------------------------------------------

======================================================================
Next Steps:
======================================================================
1. Review the commands file:
   cat commands.txt

2. Submit to your cluster's multirun system:
   multirun commands.txt

3. Monitor sweep progress:
   https://wandb.ai/YOUR-USERNAME/nash-equilibrium/sweeps/a1b2c3d4

4. (Optional) Join sweep from another machine:
   python run_sweep.py --sweep-id a1b2c3d4 --count 10
======================================================================

Sweep ID (for reference): a1b2c3d4
```

### Step 3: Review Commands File

```bash
cat commands.txt
```

```
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
python run_sweep.py --sweep-id a1b2c3d4 --count 5
```

### Step 4: Submit to Cluster

```bash
multirun commands.txt
```

Your cluster's multirun system will:
- Read each line from `commands.txt`
- Assign each line to an available machine
- Run all commands in parallel
- Each machine pulls different configs from the W&B sweep queue

### Step 5: Monitor Progress

**Option 1: W&B Dashboard (Recommended)**

Open the URL from Step 2:
```
https://wandb.ai/YOUR-USERNAME/nash-equilibrium/sweeps/a1b2c3d4
```

You'll see:
- All 10 workers running in parallel
- Current best welfare and parameters
- Parallel coordinates plot
- Parameter importance ranking
- Individual run details

**Option 2: Command Line**

```bash
# Check how many runs completed
wandb sweep a1b2c3d4

# View sweep in browser
wandb sweep --url a1b2c3d4
```

**Option 3: Check Output Files**

Each worker saves results to `data/` directory:
```bash
ls -lh data/
# opt_fid_state_chi4_lr1e-02_steps100_alpha3e-02_20250117_101234.csv
# opt_fid_state_chi6_lr2e-02_steps100_alpha5e-02_20250117_101456.csv
# ...
```

## Advanced Usage

### Custom Project and Entity

```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --name "production-run-v3" \
    --project my-project \
    --entity my-team \
    --num-workers 50 \
    --count-per-worker 2
```

### Different Worker Counts

**Few powerful workers (e.g., 5 GPU nodes):**
```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --num-workers 5 \
    --count-per-worker 20  # Each does 20 runs
```

**Many lightweight workers (e.g., 50 CPU nodes):**
```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --num-workers 50 \
    --count-per-worker 2  # Each does 2 runs
```

### Join Existing Sweep

If you already created a sweep and want to add more workers:

```bash
# Get the sweep ID
SWEEP_ID=$(cat sweep_id.txt)

# Generate more commands
for i in {1..10}; do
    echo "python run_sweep.py --sweep-id $SWEEP_ID --count 10" >> more_commands.txt
done

# Submit additional workers
multirun more_commands.txt
```

### Continuous Sweeps (for Bayesian)

For ongoing Bayesian optimization, set high counts:

```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --num-workers 10 \
    --count-per-worker 1000  # Will run for a long time
```

Workers will keep requesting new configs until you stop them or they hit the count.

## Troubleshooting

### "Error: Config file not found"

Make sure the config file exists:
```bash
ls sweep_config.yaml
# If missing, use example:
cp sweep_example_simple.yaml my_config.yaml
```

### "Error creating sweep: authentication required"

Login to W&B first:
```bash
wandb login
# Paste your API key when prompted
```

### Workers not starting

Check commands.txt has correct paths:
```bash
cat commands.txt
# Make sure 'python' and 'run_sweep.py' paths are correct
# If needed, use absolute paths:
#   /path/to/python /path/to/run_sweep.py --sweep-id ...
```

### Sweep finishes too quickly

Your sweep might have fewer configs than total requested runs:

**Grid search example:**
```yaml
parameters:
  chi: {values: [2, 4]}      # 2 options
  eps: {values: [0.01, 0.02]} # 2 options
  # Total configs: 2 × 2 = 4
```

If you request 10 workers × 5 runs = 50 total, only 4 will actually run.

**Solution:** Use Bayesian/random with large parameter spaces.

### View which workers are active

Check W&B dashboard → Runs tab → See all active runs with their worker IDs.

## Example Workflows

### Workflow 1: Quick Test (8 runs, 5 minutes)

```bash
python setup_cluster_sweep.py \
    --config sweep_example_simple.yaml \
    --name "quick-test" \
    --num-workers 4 \
    --count-per-worker 2

multirun commands.txt
```

### Workflow 2: Full Optimization (50 runs, 2-3 hours)

```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --name "full-optimization-v1" \
    --num-workers 10 \
    --count-per-worker 5

multirun commands.txt
```

### Workflow 3: Large-Scale Search (200 runs, overnight)

```bash
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --name "large-scale-search" \
    --num-workers 20 \
    --count-per-worker 10

multirun commands.txt
```

### Workflow 4: Add More Runs to Existing Sweep

```bash
# Already have sweep a1b2c3d4, want 20 more runs
python setup_cluster_sweep.py \
    --config sweep_config.yaml \
    --num-workers 10 \
    --count-per-worker 2 \
    --output more_commands.txt

# Edit more_commands.txt to use existing sweep ID
sed -i 's/--sweep-id .* /--sweep-id a1b2c3d4 /' more_commands.txt

# Submit
multirun more_commands.txt
```

## Best Practices

### 1. Start Small
```bash
# Test with 2-3 workers first
python setup_cluster_sweep.py --num-workers 3 --count-per-worker 1
```

### 2. Name Your Sweeps
Always use `--name` for easy identification:
```bash
--name "chi-sweep-baseline-v1"
--name "2025-01-17-production-run"
--name "debug-test"
```

### 3. Save Sweep IDs
The script saves to `sweep_id.txt` by default. Keep track:
```bash
# Save with descriptive names
cp sweep_id.txt sweep_ids/baseline_v1.txt
echo "baseline-v1: $(cat sweep_id.txt)" >> sweep_log.txt
```

### 4. Use Appropriate Worker Counts

Match your cluster capacity:
- **8-core nodes:** 6-7 workers per node
- **16-core nodes:** 12-14 workers per node
- **GPU nodes:** 1 worker per GPU typically

### 5. Monitor Resource Usage

Check that workers aren't fighting for resources:
```bash
# On compute node
htop  # Check CPU and memory usage
```

## Summary

**Three-step process:**
1. **Create sweep:** `python setup_cluster_sweep.py --config ... --num-workers ...`
2. **Submit jobs:** `multirun commands.txt`
3. **Monitor:** Check W&B dashboard

**Key points:**
- ✓ All workers use the SAME sweep ID
- ✓ No duplicate runs - W&B coordinates everything
- ✓ One dashboard for all results
- ✓ Works with any number of machines
- ✓ Fault-tolerant - failed workers don't affect others

Happy sweeping!
