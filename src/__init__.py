"""
Source code for quantum Nash equilibrium research.

Modules:
    mps_utils: Matrix Product State utilities (canonical forms, random states, etc.)
    tensor_ops: Tensor network operations (contraction, overlap, compression)
    game: Quantum game definitions (Hamiltonians, payoffs)
    entanglement: Entanglement parameter calculations
"""

from .mps_utils import (
    to_canonical_form,
    to_left_canonical_form,
    test_canonical_form,
    get_rand_mps,
    to_comp_basis,
    from_comp_basis,
    get_product_state,
    get_ghz_state,
    apply_random_unitaries,
    get_random_near_id_unitary,
    apply_unitary,
)

from .tensor_ops import (
    group_legs,
    ungroup_legs,
    mps_2form,
    mps_overlap,
    compress,
    mpo_on_mps,
)

from .game import (
    get_default_H,
    get_default_3players,
    get_default_4players,
    canonical_qpd,
    get_hamiltonian,
)

from .entanglement import (
    compute_entanglement_params,
    compute_purity,
    partial_trace,
)
