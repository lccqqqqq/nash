"""
Test that wandb_log_interval parameter works correctly.
"""

import numpy as np
from unittest.mock import Mock, patch
from solver import opt_fid_state
from mps_utils import get_rand_mps
from game import get_default_H

def test_log_interval():
    """Test that wandb logging respects the log interval."""
    print("="*60)
    print("Testing Wandb Log Interval")
    print("="*60)

    # Setup
    num_players = 3
    chi = 2
    max_num_steps = 50
    log_interval = 10

    Psi = get_rand_mps(L=num_players, chi=chi, d_phys=2, seed=42, dtype=np.float64)
    H = get_default_H(num_players=num_players, dtype=np.float64)

    print(f"\n1. Testing log_interval={log_interval} with {max_num_steps} steps...")
    print(f"   Expected logs at steps: 0, 10, 20, 30, 40, 49 (last step)")

    # Mock wandb to track calls
    with patch('solver.wandb') as mock_wandb:
        # Setup mock
        mock_run = Mock()
        mock_wandb.init.return_value.__enter__ = Mock(return_value=mock_run)
        mock_wandb.init.return_value.__exit__ = Mock(return_value=False)
        mock_wandb.run = None  # Simulate not already initialized

        # Run optimization with wandb
        Psi_result, metric_logs = opt_fid_state(
            Psi, H,
            max_num_steps=max_num_steps,
            eps=0.01,
            num_perturbations=3,
            subroutine_max_iter=100,
            subroutine_lr=0.03,
            use_wandb=True,
            wandb_log_interval=log_interval,
            save_results=False,
        )

        # Check how many times wandb.log was called
        log_calls = mock_wandb.log.call_count
        expected_logs = len([i for i in range(max_num_steps) if i % log_interval == 0 or i == max_num_steps - 1])

        print(f"\n2. Verification:")
        print(f"   wandb.log called: {log_calls} times")
        print(f"   Expected: {expected_logs} times")

        # Extract the steps that were logged
        logged_steps = [call[1]['step'] for call in mock_wandb.log.call_args_list]
        print(f"   Logged steps: {logged_steps}")

        # Verify
        assert log_calls == expected_logs, f"Expected {expected_logs} logs, got {log_calls}"
        assert 0 in logged_steps, "Step 0 should be logged"
        assert max_num_steps - 1 in logged_steps, "Last step should be logged"

        for i in range(max_num_steps):
            should_log = (i % log_interval == 0) or (i == max_num_steps - 1)
            if should_log:
                assert i in logged_steps, f"Step {i} should be logged"
            else:
                assert i not in logged_steps, f"Step {i} should NOT be logged"

        print(f"   ✓ All logged steps are correct")

    # Test interval=1 (every step)
    print(f"\n3. Testing log_interval=1 (every step)...")
    with patch('solver.wandb') as mock_wandb:
        mock_run = Mock()
        mock_wandb.init.return_value.__enter__ = Mock(return_value=mock_run)
        mock_wandb.init.return_value.__exit__ = Mock(return_value=False)
        mock_wandb.run = None

        Psi_result, metric_logs = opt_fid_state(
            Psi, H,
            max_num_steps=10,
            eps=0.01,
            num_perturbations=3,
            subroutine_max_iter=100,
            subroutine_lr=0.03,
            use_wandb=True,
            wandb_log_interval=1,  # Every step
            save_results=False,
        )

        log_calls = mock_wandb.log.call_count
        expected_logs = 10  # All steps

        print(f"   wandb.log called: {log_calls} times")
        print(f"   Expected: {expected_logs} times (all steps)")
        assert log_calls == expected_logs, f"Expected {expected_logs} logs, got {log_calls}"
        print(f"   ✓ Logs every step correctly")

    print("\n" + "="*60)
    print("✅ All log interval tests passed!")
    print("="*60)
    print("\nSummary:")
    print(f"  - log_interval=10: Logs at 0, 10, 20, ..., and last step")
    print(f"  - log_interval=1: Logs every step")
    print(f"  - Last step is always logged regardless of interval")

if __name__ == '__main__':
    try:
        test_log_interval()
    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
