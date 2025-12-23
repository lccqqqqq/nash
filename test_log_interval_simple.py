"""
Simple test to verify log interval logic.
"""

def test_log_interval_logic():
    """Test the logging condition logic."""
    print("="*60)
    print("Testing Log Interval Logic")
    print("="*60)

    max_num_steps = 50
    log_interval = 20

    print(f"\nTesting with max_num_steps={max_num_steps}, log_interval={log_interval}")

    logged_steps = []
    for i in range(max_num_steps):
        should_log = (i % log_interval == 0) or (i == max_num_steps - 1)
        if should_log:
            logged_steps.append(i)

    print(f"Steps that should be logged: {logged_steps}")

    # Verify expectations
    assert 0 in logged_steps, "Step 0 should be logged"
    assert max_num_steps - 1 in logged_steps, "Last step should be logged"
    assert 20 in logged_steps, "Step 20 should be logged"
    assert 40 in logged_steps, "Step 40 should be logged"
    assert 10 not in logged_steps, "Step 10 should NOT be logged"

    expected_count = len([i for i in range(max_num_steps) if i % log_interval == 0 or i == max_num_steps - 1])
    assert len(logged_steps) == expected_count

    print(f"✓ Logged {len(logged_steps)} steps as expected")

    # Test different scenarios
    scenarios = [
        (100, 20, [0, 20, 40, 60, 80, 99]),  # 6 logs
        (100, 10, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]),  # 11 logs
        (100, 1, list(range(100))),  # All 100 logs
        (50, 50, [0, 49]),  # 2 logs (first and last)
    ]

    print("\nTesting various scenarios:")
    for steps, interval, expected in scenarios:
        logged = [i for i in range(steps) if (i % interval == 0) or (i == steps - 1)]
        assert logged == expected, f"Failed for steps={steps}, interval={interval}"
        print(f"  ✓ steps={steps}, interval={interval}: {len(logged)} logs")

    print("\n" + "="*60)
    print("✅ All log interval logic tests passed!")
    print("="*60)

    print("\nSummary of behavior:")
    print("  - Always logs first step (i=0)")
    print("  - Logs every N steps (i % log_interval == 0)")
    print("  - Always logs last step (i == max_num_steps - 1)")
    print("  - Reduces log volume by ~(log_interval)x")

if __name__ == '__main__':
    try:
        test_log_interval_logic()
    except Exception as e:
        print(f"\n❌ Test failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
