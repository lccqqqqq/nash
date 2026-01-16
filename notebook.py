import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Interactive Nash Equilibrium Solver Testing

    Quick notebook for testing Nash equilibrium finding algorithms interactively without running the full optimizer.

    **Use this for:**
    - Quick experiments with different Hamiltonians
    - Debugging specific solver cases
    - Visualizing convergence trajectories
    - Comparing solver algorithms
    - Testing best response dynamics

    **Key Algorithms:**
    - **Differential Best Response (DBR)**: Gradient-based method using infinitesimal unitary perturbations
    - **Iterated Best Response (IBR)**: Sequential best response using differential evolution
    """)
    return


@app.cell
def _():
    # Imports
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm

    from src.solver import (
        find_nash_eq1,
        find_nash_eq1_with_retry,
        compute_exploitability,
        update_state_unitary,
        kick_with_u,
        compute_bipartite_entanglement_entropies
    )
    from src.game import (
        get_default_3players,
        get_default_cyclic_players,
        get_perturbed_H_QPD
    )
    from src.mps_utils import (
        get_rand_state_as_mps,
        get_ghz_state,
        get_product_state,
        to_comp_basis,
        to_canonical_form,
    )
    return (
        compute_bipartite_entanglement_entropies,
        compute_exploitability,
        find_nash_eq1,
        get_default_cyclic_players,
        get_perturbed_H_QPD,
        get_product_state,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Configuration: Game Setup

    Configure the quantum game parameters:
    - **L**: Number of players
    - **chi**: Bond dimension (controls entanglement capacity)
    - **Hamiltonian**: Payoff structure (QPD, cyclic, custom)
    - **Initial state**: Starting point for Nash search
    """)
    return


@app.cell
def _(get_default_cyclic_players, get_perturbed_H_QPD, get_product_state, np):
    # Game configuration
    L = 2  # Number of players
    chi = 8  # Bond dimension
    payoff_dtype = np.complex128
    state_dtype = np.complex128

    # Get Hamiltonian - Option 1: Perturbed Quantum Prisoner's Dilemma
    Hs = get_perturbed_H_QPD(eps=0.1, dtype=payoff_dtype, seed=43)
    H = get_default_cyclic_players(L=L, Hs=Hs, dtype=payoff_dtype)

    # Initialize state - start from product state |11...1⟩
    Psi_init = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)

    print(f"Game setup: L={L} players, chi={chi}, dtype={state_dtype}")
    print(f"Hamiltonian shape: {[h.shape for h in H]}")
    print(f"Initial state: Product state |{'1'*L}⟩")
    return H, L, Psi_init, state_dtype


@app.cell
def _(mo):
    mo.md("""
    ## 2. Run Nash Equilibrium Solver

    Test the differential best response (DBR) solver with configurable parameters:
    - **alpha**: Learning rate for gradient ascent
    - **expl_threshold**: Convergence threshold (exploitability tolerance)
    - **real_strategies**: Whether to restrict strategies to real unitaries
    """)
    return


@app.cell
def _(H, Psi_init, compute_bipartite_entanglement_entropies, find_nash_eq1):
    # Make a copy to avoid modifying the original
    Psi = [p.copy() for p in Psi_init]

    # Display initial entanglement
    print("Initial bipartite entanglement entropies:")
    print(compute_bipartite_entanglement_entropies(Psi))

    # Solver configuration
    alpha = 0.1  # Learning rate for gradient ascent
    expl_threshold = 5e-16  # Very tight convergence

    # Run the differential best response solver
    result = find_nash_eq1(
        Psi, H,
        max_iter=1000,
        alpha=alpha,
        expl_threshold=expl_threshold,
        expl_check_interval=50,
        expl_maxiter=300,
        real_strategies=False,
        return_history=True,
        use_tqdm=True
    )

    print(f"\n{'='*60}")
    print("Solver Results (Differential Best Response):")
    print(f"{'='*60}")
    print(f"  Converged: {result['nash_equilibrium']}")
    # print(result.keys())
    print(f"  Iterations: {result['num_iters']}")
    print(f"  Final exploitability: {result['expl'][-1]}")
    print(f"  Final energies: {result['energy'][-1]}")
    print(f"  Energy change: {result['energy'][-1] - result['energy'][0]}")
    return (result,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Visualize Convergence

    Plot the energy and exploitability trajectories to understand solver behavior.
    """)
    return


@app.cell
def _(np, plt, result):
    def plot_energy_and_exploitability(result):
        """Plot energy and exploitability convergence trajectories."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel: Energy trajectories (relative to initial)
        L = len(result['energy'][0])
        energy_array = np.array(result['energy'])
        energy_relative = energy_array - energy_array[0]

        for i in range(L):
            axs[0].plot(energy_relative[:, i], linewidth=1.5, label=f"Player {i}")

        axs[0].set_xlabel("Iteration", fontsize=11)
        axs[0].set_ylabel("Energy (relative to initial)", fontsize=11)
        axs[0].set_title("Energy Trajectories", fontsize=12, fontweight='bold')
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # Right panel: Exploitability (log scale)
        expl_array = np.array(result['expl'])
        axs[1].semilogy(expl_array, linewidth=2, color='C3')
        axs[1].set_xlabel("Iteration", fontsize=11)
        axs[1].set_ylabel("Exploitability (log scale)", fontsize=11)
        axs[1].set_title("Convergence Monitor", fontsize=12, fontweight='bold')
        axs[1].grid(alpha=0.3, which='both')
        axs[1].axhline(y=1e-3, color='red', linestyle='--',
                       linewidth=1, label='Typical threshold (1e-3)')
        axs[1].legend()

        plt.tight_layout()
        return fig

    # Generate the plot
    fig_convergence = plot_energy_and_exploitability(result)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Verify Exploitability

    Double-check the exploitability of the final state to ensure the solver found a true Nash equilibrium.
    """)
    return


@app.cell
def _(H, compute_exploitability, result):
    # Verify final state exploitability
    Psi_final = result['state_'][0]
    psi_final = result['state'][0]

    print(f"{'='*60}")
    print("Independent Exploitability Verification:")
    print(f"{'='*60}")

    # Compute exploitability using the verification function
    expl_check, best_payoffs, baseline_payoffs = compute_exploitability(
        Psi_final, H,
        real_strategies=False,
        maxiter=300,
        player_idx=1
    )

    print(f"  Solver's reported exploitability: {result['expl'][-1]:.2e}")
    print(f"  Independent verification: {expl_check:.2e}")
    print(f"\nPer-player breakdown:")
    for i, (best, baseline) in enumerate(zip(best_payoffs, baseline_payoffs)):
        gain = best - baseline
        print(f"  Player {i}: baseline={baseline:.6f}, best={best:.6f}, gain={gain:.2e}")
    return (Psi_final,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Entanglement Analysis

    Analyze the entanglement structure of the Nash equilibrium state.
    """)
    return


@app.cell
def _(Psi_final, compute_bipartite_entanglement_entropies):
    print(f"{'='*60}")
    print("Entanglement Structure of Nash Equilibrium:")
    print(f"{'='*60}")

    entropies = compute_bipartite_entanglement_entropies(Psi_final)

    print("\nBipartite entanglement entropies:")
    for i, entropy in enumerate(entropies):
        print(f"  Cut after site {i}: S = {entropy:.6f}")

    # Check if state is entangled
    max_entropy = max(entropies)
    if max_entropy < 1e-6:
        print("\n✓ State is essentially a product state (no entanglement)")
    else:
        print(f"\n✓ State has non-trivial entanglement (max S = {max_entropy:.6f})")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Multiple Initial Conditions Test

    Test solver robustness by running from different random initial states.
    """)
    return


@app.cell
def _(H, L, find_nash_eq1, get_product_state, pd, scramble, state_dtype):
    print(f"{'='*60}")
    print("Testing Multiple Initial Conditions:")
    print(f"{'='*60}\n")

    n_trials = 5
    results_multi = []

    for trial in range(n_trials):
        # Create different initial states
        if trial == 0:
            Psi_test = get_product_state(L=L, state_per_site=[0]*L, dtype=state_dtype)
            init_label = "|00⟩"
        elif trial == 1:
            Psi_test = get_product_state(L=L, state_per_site=[1]*L, dtype=state_dtype)
            init_label = "|11⟩"
        else:
            Psi_test = get_product_state(L=L, state_per_site=[0]*L, dtype=state_dtype)
            Psi_test = scramble(Psi_test, depth=2, unitary_eps=0.2, seed=trial)
            init_label = f"Scrambled (seed={trial})"

        # Run solver
        res = find_nash_eq1(
            Psi_test, H,
            max_iter=1000,
            alpha=0.1,
            expl_threshold=1e-3,
            expl_check_interval=50,
            expl_maxiter=300,
            real_strategies=False,
            return_history=True,
            use_tqdm=False
        )

        results_multi.append({
            'trial': trial,
            'initial_state': init_label,
            'converged': res['converged'],
            'iterations': res['num_iter'],
            'final_expl': res['expl'][-1],
            'final_energy': res['energy'][-1],
        })

        status = "✓" if res['converged'] else "✗"
        print(f"{status} Trial {trial} ({init_label:20s}): "
              f"converged={res['converged']}, iter={res['num_iter']:4d}, "
              f"expl={res['expl'][-1]:.2e}")

    # Summary table
    df_multi = pd.DataFrame(results_multi)
    print(f"\n{'='*60}")
    print("Summary Table:")
    print(f"{'='*60}")
    print(df_multi.to_string(index=False))
    return (df_multi,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Summary & Diagnostics

    Key takeaways from the Nash equilibrium solver testing.
    """)
    return


@app.cell
def _(df_multi, result):
    print(f"{'='*60}")
    print("SUMMARY & DIAGNOSTICS")
    print(f"{'='*60}\n")

    print("Main Solver Run:")
    print(f"  • Algorithm: Differential Best Response (DBR)")
    print(f"  • Converged: {result['converged']}")
    print(f"  • Final exploitability: {result['expl'][-1]:.2e}")
    print(f"  • Iterations: {result['num_iter']}")

    print("\nMulti-Initial Condition Test:")
    print(f"  • Total trials: {len(df_multi)}")
    print(f"  • Convergence rate: {df_multi['converged'].sum()}/{len(df_multi)}")
    print(f"  • Avg iterations (converged): {df_multi[df_multi['converged']]['iterations'].mean():.1f}")

    print("\nNotes:")
    print("  • DBR uses gradient-based local search with infinitesimal perturbations")
    print("  • Convergence depends on learning rate (alpha) and threshold")
    print("  • For global search, use iterated best response with retry mechanism")
    return


if __name__ == "__main__":
    app.run()
