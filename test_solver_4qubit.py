"""Test script for solver.py with 4-qubit systems."""

import numpy as np
import sys
from solver import (
    apply_u, perturb_state, batch_perturb, find_nash_eq1,
    compute_exploitability, estimate_gradient_ols,
    update_state, update_state_unitary,
    compute_four_qubit_SLOCC_invariants, compute_ent_params_from_state
)
from mps_utils import get_rand_mps, to_comp_basis, to_canonical_form, get_product_state
from game import get_default_H

np.set_printoptions(precision=4, suppress=True)

def test_apply_u():
    """Test apply_u function on 4-qubit state."""
    print("\n" + "="*60)
    print("TEST 1: apply_u function")
    print("="*60)

    # Create simple 4-qubit state
    psi = np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)
    psi = psi / np.linalg.norm(psi)

    # Create random unitary
    U = np.linalg.qr(np.random.randn(2, 2) + 1j * np.random.randn(2, 2))[0]

    # Apply to different qubits
    for idx in range(4):
        psi_new = apply_u(U, psi, [idx])

        # Check normalization preserved
        norm_original = np.linalg.norm(psi)
        norm_new = np.linalg.norm(psi_new)

        print(f"  Qubit {idx}: Original norm = {norm_original:.6f}, New norm = {norm_new:.6f}")
        assert np.abs(norm_new - 1.0) < 1e-10, f"Norm not preserved for qubit {idx}"

    print("  ✓ apply_u preserves normalization for all qubits")
    return True


def test_slocc_invariants():
    """Test 4-qubit SLOCC invariants."""
    print("\n" + "="*60)
    print("TEST 2: 4-qubit SLOCC invariants")
    print("="*60)

    # Test on product state |0000⟩
    psi_product = np.zeros(16, dtype=np.complex128)
    psi_product[0] = 1.0
    H, L, M, Dxt = compute_four_qubit_SLOCC_invariants(psi_product)
    print(f"  Product state |0000⟩:")
    print(f"    H={H:.6f}, L={L:.6f}, M={M:.6f}, Dxt={Dxt:.6f}")

    # Test on GHZ-like state (|0000⟩ + |1111⟩)/√2
    psi_ghz = np.zeros(16, dtype=np.complex128)
    psi_ghz[0] = 1/np.sqrt(2)
    psi_ghz[15] = 1/np.sqrt(2)
    H, L, M, Dxt = compute_four_qubit_SLOCC_invariants(psi_ghz)
    print(f"  GHZ state (|0000⟩+|1111⟩)/√2:")
    print(f"    H={H:.6f}, L={L:.6f}, M={M:.6f}, Dxt={Dxt:.6f}")

    # Test on random state
    psi_rand = np.random.randn(16) + 1j * np.random.randn(16)
    psi_rand = psi_rand / np.linalg.norm(psi_rand)
    H, L, M, Dxt = compute_four_qubit_SLOCC_invariants(psi_rand)
    print(f"  Random state:")
    print(f"    H={H:.6f}, L={L:.6f}, M={M:.6f}, Dxt={Dxt:.6f}")

    # Test compute_ent_params_from_state wrapper
    params = compute_ent_params_from_state(psi_ghz.reshape(2,2,2,2))
    print(f"  compute_ent_params_from_state output shape: {params.shape}")
    assert params.shape == (4,), f"Expected 4 parameters, got {params.shape}"
    print(f"  ✓ SLOCC invariant computation works")
    return True


def test_perturb_state():
    """Test state perturbation methods."""
    print("\n" + "="*60)
    print("TEST 3: State perturbation (schmidt and unitary)")
    print("="*60)

    # Create 4-qubit MPS with small bond dimension
    chi = 2
    Psi = get_rand_mps(L=4, chi=chi, d_phys=2, seed=42)

    for method in ['schmidt', 'unitary']:
        print(f"\n  Method: {method}")

        # Test single perturbation
        Psi_perturbed, orig_param, new_param = perturb_state(Psi, lr=0.01, site=0, method=method)

        # Check state is still normalized
        psi_comp = to_comp_basis(Psi_perturbed).reshape(2**4)
        norm = np.linalg.norm(psi_comp)
        print(f"    Single perturbation norm: {norm:.6f}")
        assert np.abs(norm - 1.0) < 1e-6, f"Norm not preserved in {method} perturbation"

        # Test batch perturbation
        batch_size = 5
        Psi_batch, orig_param, batch_param = batch_perturb(
            Psi, batch_size=batch_size, lr=0.01, site=0, method=method
        )

        print(f"    Batch perturbation size: {len(Psi_batch)} tensors, batch={batch_size}")
        print(f"    Psi_batch[0] shape: {Psi_batch[0].shape}")

        # Check all batch states are normalized
        for i in range(batch_size):
            psi_i = [Psi_batch[j][i] for j in range(len(Psi_batch))]
            psi_comp_i = to_comp_basis(psi_i).reshape(2**4)
            norm_i = np.linalg.norm(psi_comp_i)
            assert np.abs(norm_i - 1.0) < 1e-6, f"Batch state {i} not normalized in {method}"

        print(f"    ✓ All {batch_size} batch states normalized")

    print(f"\n  ✓ Both perturbation methods work correctly")
    return True


def test_gradient_estimation():
    """Test OLS gradient estimation."""
    print("\n" + "="*60)
    print("TEST 4: Gradient estimation (OLS)")
    print("="*60)

    # Create synthetic linear data: y = 2*x1 + 3*x2 + noise
    n_samples = 20
    n_dims = 5
    true_gradient = np.array([2.0, 3.0, 0.0, -1.0, 1.5])

    dX = np.random.randn(n_samples, n_dims)
    dy = dX @ true_gradient + 0.01 * np.random.randn(n_samples)

    # Estimate gradient
    g_est = estimate_gradient_ols(dX, dy)

    print(f"  True gradient:      {true_gradient}")
    print(f"  Estimated gradient: {g_est}")
    print(f"  Error:              {np.linalg.norm(g_est - true_gradient):.6f}")

    assert np.linalg.norm(g_est - true_gradient) < 0.1, "Gradient estimation error too large"
    print(f"  ✓ Gradient estimation accurate")

    # Test underdetermined case
    print(f"\n  Testing underdetermined case (fewer samples than dimensions):")
    dX_under = np.random.randn(3, 5)
    dy_under = dX_under @ true_gradient
    g_est_under = estimate_gradient_ols(dX_under, dy_under)
    print(f"  Estimated gradient (underdetermined): {g_est_under}")
    print(f"  ✓ Handles underdetermined case without crashing")

    return True


def test_update_functions():
    """Test state update functions."""
    print("\n" + "="*60)
    print("TEST 5: State update functions")
    print("="*60)

    chi = 2
    Psi = get_rand_mps(L=4, chi=chi, d_phys=2, seed=42)

    # Test schmidt update
    print(f"\n  Testing update_state (schmidt):")
    S_grad = np.random.randn(chi)
    Psi_updated = update_state(Psi, S_grad, lr=0.01, site=0)
    psi_comp = to_comp_basis(Psi_updated).reshape(2**4)
    norm = np.linalg.norm(psi_comp)
    print(f"    Updated state norm: {norm:.6f}")
    assert np.abs(norm - 1.0) < 1e-6, "update_state doesn't preserve norm"
    print(f"    ✓ update_state preserves normalization")

    # Test unitary update
    print(f"\n  Testing update_state_unitary:")
    coef_grad = np.random.randn(9)
    Psi_updated = update_state_unitary(Psi, coef_grad, lr=0.01, site=0)
    psi_comp = to_comp_basis(Psi_updated).reshape(2**4)
    norm = np.linalg.norm(psi_comp)
    print(f"    Updated state norm: {norm:.6f}")
    assert np.abs(norm - 1.0) < 1e-6, "update_state_unitary doesn't preserve norm"
    print(f"    ✓ update_state_unitary preserves normalization")

    return True


def test_exploitability():
    """Test exploitability computation."""
    print("\n" + "="*60)
    print("TEST 6: Exploitability computation")
    print("="*60)

    # Get 4-player Hamiltonian
    H = get_default_H(num_players=4)
    print(f"  Hamiltonian shape: {[h.shape for h in H]}")

    # Create random 4-qubit state
    chi = 2
    Psi = get_rand_mps(L=4, chi=chi, d_phys=2, seed=42)
    psi = to_comp_basis(Psi).reshape([2]*4)

    # Compute exploitability for each player
    print(f"\n  Computing exploitability for each player:")
    for player_idx in range(4):
        expl = compute_exploitability(psi, H, player_idx)
        print(f"    Player {player_idx}: exploitability = {expl:.6e}")
        assert expl >= -1e-10, f"Exploitability should be non-negative, got {expl}"

    print(f"  ✓ Exploitability computation works for all players")
    return True


def test_nash_equilibrium_finder():
    """Test Nash equilibrium finder on 4-qubit system."""
    print("\n" + "="*60)
    print("TEST 7: Nash equilibrium finder (find_nash_eq1)")
    print("="*60)

    # Get 4-player Hamiltonian
    H = get_default_H(num_players=4)

    # Create initial state with chi=16 for sufficient expressiveness
    chi = 16
    Psi = get_rand_mps(L=4, chi=chi, d_phys=2, seed=42)
    Psi = to_canonical_form(Psi, form='B')

    print(f"  Running Nash equilibrium finder...")
    print(f"    Bond dimension: {chi}")
    print(f"    Max iterations: 500")

    result = find_nash_eq1(
        Psi, H,
        max_iter=500,
        alpha=0.01,
        convergence_threshold=1e-6,
        expl_threshold=1e-3,  # Stricter threshold
        use_tqdm=True,  # Show progress
        expl_check_interval=5,
        return_history=False
    )

    print(f"\n  Results:")
    print(f"    Nash state converged: {result['nash_state']}")
    print(f"    Nash equilibrium (global): {result['nash_equilibrium']}")
    print(f"    Number of iterations: {result['num_iters']}")
    print(f"    Final energies: {result['energy']}")
    print(f"    Final exploitability per player:")
    for i, expl in enumerate(result['expl'][-1]):
        print(f"      Player {i}: {expl:.6e}")
    print(f"    Total exploitability: {sum(result['expl'][-1]):.6e}")

    # Verify state is normalized
    psi_final = to_comp_basis(result['state_']).reshape(2**4)
    norm = np.linalg.norm(psi_final)
    print(f"    Final state norm: {norm:.6f}")
    assert np.abs(norm - 1.0) < 1e-6, "Final state not normalized"

    print(f"\n  ✓ Nash equilibrium finder runs successfully")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# TESTING solver.py WITH 4-QUBIT SYSTEMS")
    print("#"*60)

    tests = [
        ("apply_u", test_apply_u),
        ("SLOCC invariants", test_slocc_invariants),
        ("State perturbation", test_perturb_state),
        ("Gradient estimation", test_gradient_estimation),
        ("State updates", test_update_functions),
        ("Exploitability", test_exploitability),
        ("Nash equilibrium finder", test_nash_equilibrium_finder),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED with error:")
            print(f"    {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))

    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")

    n_pass = sum(1 for _, s in results if s == "PASS")
    n_total = len(results)
    print(f"\nTotal: {n_pass}/{n_total} tests passed")

    return n_pass == n_total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
