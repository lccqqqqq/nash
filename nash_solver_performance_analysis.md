# Nash Solver Performance Degradation Analysis

## Executive Summary

The 3-player complex strategy Nash equilibrium solver is failing to converge in recent runs, getting stuck in local minima. The root cause is a **learning rate adaptation strategy change** introduced in commit `4423fa5` that decreases LR when local convergence is detected, preventing escape from local minima.

## Comparison of Working vs Failing Runs

### Old Working Run (`output/misc/complex_2584691.out`)
- **Date**: Jan 2, 2026
- **Configuration**:
  - Players: 3, Chi: 4
  - Non-commutative norm: 0.3
  - Subroutine LR: 0.06
  - Max subroutine LR: 0.5
  - Expl maxiter: 100
  - Expl threshold: ~5e-4 (default from old code)

- **LR Adaptation Behavior** (Old Strategy):
  ```
  Retry 1/20 with higher learning rate: 0.0820
  Retry 2/20 with higher learning rate: 0.1040
  Success on retry 3 with LR 0.1040
  Updated working LR: 0.0600 → 0.1040
  ```
  - **Always increases** LR on each retry
  - LR progression: 0.06 → 0.104 → 0.124 → ... → 0.444
  - Nash equilibria found successfully throughout run

### New Failing Run (`2629936_hseed1000_nc0.010526315789473684_job1.out`)
- **Date**: Jan 12, 2026
- **Configuration**:
  - Players: 3, Chi: 6
  - Non-commutative norm: 0.0105
  - Subroutine LR: 0.02
  - Max subroutine LR: 0.6
  - **Min subroutine LR: 0.001**
  - Expl maxiter: 200
  - **Expl threshold: 1e-7**

- **LR Adaptation Behavior** (New Adaptive Strategy):
  ```
  Retry 1/20: No convergence - increasing LR to 0.0490
  Converged to Nash state at iteration 658
  Retry 2/20: Local convergence but not global - decreasing LR to 0.0200
  Converged to Nash state at iteration 3
  Retry 3/20: Local convergence but not global - decreasing LR to 0.0010
  [... repeats 17 more times with LR=0.0010 ...]
  Warning: Initial baseline NE not found after 20 retries
  ```
  - First retry **increases** LR → achieves local convergence at iteration 658
  - Subsequent retries **decrease** LR because locally converged but not globally
  - Gets stuck at LR=0.001 (too small to escape local minimum)
  - All 20 retries fail

## Root Cause Analysis

### 1. Learning Rate Adaptation Strategy Change (PRIMARY ISSUE)

**Commit**: `4423fa5` - "Refined solver logic for subroutine learning rate modulation"

**Old Strategy** (Always Increase):
```python
# Failed - increase LR for next retry
if retry_count < max_retries - 1:
    current_alpha = base_alpha + ((retry_count + 1) / max_retries) * (max_alpha - base_alpha)
    print(f"Retry {retry_count + 1}/{max_retries} with higher learning rate: {current_alpha:.4f}")
```

**New Strategy** (Adaptive Increase/Decrease):
```python
# Adaptive LR adjustment based on convergence state
if retry_count < max_retries - 1:
    if not result['nash_state'] and not result['nash_equilibrium']:
        # Not making progress → increase LR by fixed step
        current_alpha = min(current_alpha + alpha_step, max_alpha)
        print(f"Retry {retry_count + 1}/{max_retries}: No convergence - increasing LR to {current_alpha:.4f}")
    elif result['nash_state'] and not result['nash_equilibrium']:
        # Locally converged but overshooting globally → decrease LR by fixed step
        current_alpha = max(current_alpha - alpha_step, min_alpha)
        print(f"Retry {retry_count + 1}/{max_retries}: Local convergence but not global - decreasing LR to {current_alpha:.4f}")
```

**Key Definitions** (from `find_nash_eq1` in solver.py:318-328):
- **local_converged** (`nash_state`): Energy changes < convergence_threshold
  ```python
  local_converged = sum([abs(E[i] - Es[-2][i]) for i in range(L)]) < convergence_threshold
  ```
- **global_converged** (`nash_equilibrium`): Global exploitability < expl_threshold
  ```python
  if sum(expl) < expl_threshold:
      global_converged = True
  ```

**Why the New Strategy Fails**:
1. First retry increases LR → state converges **locally** (energy stabilizes) but **not globally** (exploitability still high)
2. Algorithm interprets this as "overshooting" and **decreases** LR
3. Decreased LR is too small → state stuck in local minimum
4. All subsequent retries converge locally in ~3 iterations but fail globally
5. LR keeps decreasing until hitting min_alpha=0.001
6. LR=0.001 cannot escape local minimum → all 20 retries fail

**Fundamental Issue**: The adaptive strategy assumes:
- Local convergence + global failure = overshooting → need smaller LR

But in practice:
- Local convergence + global failure = **local minimum** → need **larger** LR to escape

### 2. Exploitability Threshold Too Strict

**Old**: `expl_threshold ≈ 5e-4` (default)
**New**: `expl_threshold = 1e-7` (14x stricter)

- Stricter threshold requires higher precision Nash equilibria
- May reject states that are "good enough" equilibria
- Forces solver into tighter local minima

### 3. Minimum Learning Rate Too Low

**New parameter**: `min_subroutine_lr = 0.001`

- When adaptive strategy decreases LR, it bottoms out at 0.001
- LR=0.001 is too small for 3-player complex strategies with chi=6
- Cannot escape even shallow local minima

### 4. Other Configuration Differences (Secondary)

- **Chi increased**: 4 → 6 (larger bond dimension, more complex optimization landscape)
- **Non-commutative norm decreased**: 0.3 → 0.0105 (less perturbation, may have different landscape structure)
- **Base LR decreased**: 0.06 → 0.02 (starting point lower)
- **Expl maxiter increased**: 100 → 200 (more iterations for exploitability, but doesn't help if stuck in local minimum)

## Proposed Experiments to Test Root Causes

### Experiment 1: Revert to Old LR Strategy (RECOMMENDED FIRST)

**Hypothesis**: Old "always increase" strategy will restore performance

**Implementation**:
1. Create a new CLI flag: `--lr-adaptation-strategy` with options:
   - `'always_increase'` (old behavior)
   - `'adaptive'` (current behavior)
   - `'always_increase_multiplicative'` (geometric progression)

2. Modify `find_nash_eq1_with_retry()` to support both strategies

3. **Test configurations**:
   ```bash
   # Test A1: Old strategy, new hyperparameters
   --lr-adaptation-strategy always_increase
   --subroutine-lr 0.02
   --max-subroutine-lr 0.6
   --min-subroutine-lr 0.001
   --expl-threshold 1e-7

   # Test A2: Old strategy, old hyperparameters
   --lr-adaptation-strategy always_increase
   --subroutine-lr 0.06
   --max-subroutine-lr 0.5
   --expl-threshold 5e-4

   # Test A3: Adaptive strategy with modified logic (increase on local-only convergence)
   --lr-adaptation-strategy adaptive_modified
   ```

**Expected Outcome**: Test A1 should restore performance, confirming LR strategy is the primary issue

### Experiment 2: Relax Exploitability Threshold

**Hypothesis**: Stricter threshold forces solver into tighter local minima

**Test configurations**:
```bash
# Baseline (failing)
--expl-threshold 1e-7

# Relaxed thresholds
--expl-threshold 1e-6
--expl-threshold 1e-5
--expl-threshold 5e-4  # Old default
--expl-threshold 1e-3
```

**Metrics**:
- Success rate (% of runs finding NE)
- Number of retries needed
- Final exploitability value
- Convergence iteration count

**Expected Outcome**: More relaxed thresholds should increase success rate

### Experiment 3: Increase Minimum Learning Rate

**Hypothesis**: min_subroutine_lr=0.001 is too small to escape local minima

**Test configurations**:
```bash
# Current (failing)
--min-subroutine-lr 0.001

# Higher minimums
--min-subroutine-lr 0.005
--min-subroutine-lr 0.01
--min-subroutine-lr 0.02
--min-subroutine-lr 0.05
```

**Expected Outcome**: Higher minimum LR should reduce getting stuck at LR=0.001

### Experiment 4: Modified Adaptive Strategy

**Hypothesis**: Adaptive strategy can work if logic is corrected

**Proposed Modified Logic**:
```python
if not result['nash_state'] and not result['nash_equilibrium']:
    # No convergence at all → increase LR (make bigger steps)
    current_alpha = min(current_alpha + alpha_step, max_alpha)

elif result['nash_state'] and not result['nash_equilibrium']:
    # Local convergence but not global → INCREASE LR to escape local minimum
    current_alpha = min(current_alpha + alpha_step, max_alpha)

elif not result['nash_state'] and result['nash_equilibrium']:
    # Global converged but not local (rare edge case) → success anyway
    return result, True, current_alpha
```

**Key Change**: Local-only convergence now **increases** LR instead of decreasing

**Test**: Compare modified adaptive vs current adaptive vs always_increase

### Experiment 5: Chi and Non-Commutativity Sensitivity

**Hypothesis**: Performance degradation may be exacerbated by chi=6 or low non-commutativity

**Test matrix**:
```
Chi values: [2, 4, 6, 8]
Non-comm norms: [0.0, 0.05, 0.1, 0.2, 0.3]
LR strategies: [always_increase, adaptive]
```

**Expected Outcome**: Identify if chi or non-comm amplifies the LR strategy issue

### Experiment 6: Learning Rate Range Scan

**Hypothesis**: Optimal LR range may differ from old runs

**Test configurations**:
- **Subroutine LR**: 0.01, 0.02, 0.04, 0.06, 0.08, 0.1
- **Max LR**: 0.3, 0.4, 0.5, 0.6, 0.8, 1.0
- **Min LR**: 0.001, 0.005, 0.01, 0.02

**Expected Outcome**: Find optimal LR range for 3-player chi=6 systems

## Recommended Implementation Plan

### Phase 1: Quick Diagnosis (1-2 days)

1. **Implement LR strategy flag** in solver.py
2. **Run Experiment 1** (revert to old strategy)
   - If successful → confirms root cause
   - If still failing → investigate other factors

### Phase 2: Optimization (3-5 days)

1. **Run Experiment 2** (exploitability threshold scan)
2. **Run Experiment 3** (minimum LR scan)
3. **Implement and test Experiment 4** (modified adaptive strategy)

### Phase 3: Comprehensive Sweep (ongoing)

1. **Run Experiment 5** (chi and non-comm sensitivity)
2. **Run Experiment 6** (LR range optimization)
3. **Document optimal hyperparameters** for different system sizes

## Code Changes Required

### 1. Add LR Adaptation Strategy Flag

**Location**: `src/solver.py` DEFAULTS dict (~line 1670)
```python
'lr_adaptation_strategy': 'always_increase',  # Options: 'always_increase', 'adaptive', 'adaptive_modified'
```

**Location**: `src/solver.py` argparse (~line 1750)
```python
parser.add_argument('--lr-adaptation-strategy', type=str, default=DEFAULTS['lr_adaptation_strategy'],
                    choices=['always_increase', 'adaptive', 'adaptive_modified'],
                    help='Learning rate adaptation strategy for Nash solver retries')
```

### 2. Modify `find_nash_eq1_with_retry()`

**Location**: `src/solver.py` lines 344-420

Add `strategy` parameter and implement branching logic for different strategies.

### 3. Update Experiment Scripts

Add `--lr-adaptation-strategy always_increase` to existing scripts to restore old behavior.

## Success Criteria

- [ ] Success rate > 90% for 3-player chi=6 systems with non-comm norm ~0.01
- [ ] Convergence within 10 retries (not hitting max 20)
- [ ] Working LR stabilizes (not continuously increasing or decreasing)
- [ ] Performance comparable to old runs (Jan 2 baseline)

## References

- **Old working run**: `output/misc/complex_2584691.out`
- **New failing run**: `output/hamiltonian_sweep_3p_complex_v1/2629936_hseed1000_nc0.010526315789473684_job1.out`
- **Critical commit**: `4423fa5` - "Refined solver logic for subroutine learning rate modulation"
- **Solver code**: `src/solver.py` lines 344-420 (`find_nash_eq1_with_retry`)
