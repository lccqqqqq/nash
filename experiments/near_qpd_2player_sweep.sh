#!/bin/bash
# Near-QPD 2-Player Nash Equilibrium Parameter Sweep
#
# This script launches a systematic experiment testing Nash equilibrium existence
# and welfare in 2-player quantum games as functions of:
#   - Initial state entanglement (Schmidt eigenvalue λ₁)
#   - Hamiltonian non-commutativity (perturbation strength λ)
#   - Perturbation direction (random seeds for V)
#
# Total jobs: 10 (λ₁) × 10 (V seeds) × 20 (λ) × 5 (basis seeds) = 10,000 jobs

# ==============================================================================
# PARAMETERS
# ==============================================================================

# Schmidt eigenvalue range (low entanglement: 0.9 to 1.0)
SCHMIDT_LAMBDAS=($(python -c "import numpy as np; print(' '.join([f'{x:.2f}' for x in np.linspace(0.9, 1.0, 10)]))"))

# Non-commutativity range (0 to 0.2 with 20 points)
NONCOMM_LAMBDAS=($(python -c "import numpy as np; print(' '.join([f'{x:.3f}' for x in np.linspace(0, 0.2, 20)]))"))

# Perturbation seeds (10 different V directions)
PERTURBATION_SEEDS=(1000 1100 1200 1300 1400 1500 1600 1700 1800 1900)

# Basis randomization seeds (5 random orientations per configuration)
NUM_BASIS_SEEDS=5

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

EXPL_THRESHOLD=1e-4  # Lower threshold for tighter convergence
MAX_ITER=10000       # Maximum Nash equilibrium iterations
ALPHA=0.01           # Nash equilibrium learning rate
EXPL_CHECK_INTERVAL=50   # Check exploitability every N iterations
EXPL_MAXITER=60      # Max iterations for exploitability computation

# ==============================================================================
# CLUSTER SETTINGS
# ==============================================================================

QUEUE="long"
NCORES=1
MEM_PER_CORE=1.8  # GB

# ==============================================================================
# DIRECTORIES
# ==============================================================================

BASE_SAVE_DIR="data/near_qpd_2player_experiment"
OUTPUT_DIR="output/near_qpd_2p"

# Create directories
mkdir -p $OUTPUT_DIR

# ==============================================================================
# SWEEP EXECUTION
# ==============================================================================

echo "======================================================================"
echo "Near-QPD 2-Player Nash Equilibrium Parameter Sweep"
echo "======================================================================"
echo "Configuration:"
echo "  Schmidt λ₁ values:    ${#SCHMIDT_LAMBDAS[@]} ($(echo ${SCHMIDT_LAMBDAS[0]} ${SCHMIDT_LAMBDAS[-1]}))"
echo "  Non-comm λ values:    ${#NONCOMM_LAMBDAS[@]} ($(echo ${NONCOMM_LAMBDAS[0]} ${NONCOMM_LAMBDAS[-1]}))"
echo "  Perturbation seeds:   ${#PERTURBATION_SEEDS[@]}"
echo "  Basis seeds per config: $NUM_BASIS_SEEDS"
echo ""
echo "Hyperparameters:"
echo "  Exploitability threshold: $EXPL_THRESHOLD"
echo "  Max iterations:           $MAX_ITER"
echo "  Alpha (learning rate):    $ALPHA"
echo ""
echo "Cluster settings:"
echo "  Queue: $QUEUE"
echo "  Cores: $NCORES"
echo "  Memory: $MEM_PER_CORE GB per core"
echo "======================================================================"
echo ""

TOTAL_JOBS=0
SUBMITTED_JOBS=0

# Main sweep loop
for SCHMIDT_LAMBDA in "${SCHMIDT_LAMBDAS[@]}"; do
    echo "Schmidt λ₁ = $SCHMIDT_LAMBDA"

    for PERT_SEED in "${PERTURBATION_SEEDS[@]}"; do
        echo "  Perturbation seed: $PERT_SEED"

        for NONCOMM_LAMBDA in "${NONCOMM_LAMBDAS[@]}"; do
            # Create save directory for this configuration
            SAVE_DIR="${BASE_SAVE_DIR}/schmidt_${SCHMIDT_LAMBDA}/pert_seed_${PERT_SEED}/lambda_${NONCOMM_LAMBDA}"
            mkdir -p $SAVE_DIR

            # Launch jobs with different basis seeds
            for BASIS_SEED in $(seq 1 $NUM_BASIS_SEEDS); do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))

                # Submit job
                JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE \
                    -o ${OUTPUT_DIR}/%j_s${SCHMIDT_LAMBDA}_ps${PERT_SEED}_nc${NONCOMM_LAMBDA}_bs${BASIS_SEED}.out \
                    -c near_qpd_2p \
                    python3 experiments/near_qpd_2player.py \
                    --schmidt-lambda $SCHMIDT_LAMBDA \
                    --noncomm-lambda $NONCOMM_LAMBDA \
                    --pert-seed $PERT_SEED \
                    --basis-seed $BASIS_SEED \
                    --save-dir $SAVE_DIR \
                    --expl-threshold $EXPL_THRESHOLD \
                    --max-iter $MAX_ITER \
                    --alpha $ALPHA \
                    --expl-check-interval $EXPL_CHECK_INTERVAL \
                    --expl-maxiter $EXPL_MAXITER \
                    2>&1)

                # Check if submission succeeded
                if echo "$JOB_OUTPUT" | grep -q "python3-"; then
                    SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
                fi

                # Progress update every 100 jobs
                if [ $((TOTAL_JOBS % 100)) -eq 0 ]; then
                    echo "    Progress: $TOTAL_JOBS jobs processed, $SUBMITTED_JOBS submitted"
                fi
            done
        done

        echo "    ✓ Completed noncomm λ sweep for pert_seed=$PERT_SEED"
    done

    echo "  ✓ Completed pert_seed sweep for schmidt_λ₁=$SCHMIDT_LAMBDA"
    echo ""
done

echo ""
echo "======================================================================"
echo "Sweep completed!"
echo "======================================================================"
echo "Summary:"
echo "  Total jobs attempted:  $TOTAL_JOBS"
echo "  Successfully submitted: $SUBMITTED_JOBS"
echo ""
echo "Expected statistics:"
echo "  Schmidt λ₁ samples:     ${#SCHMIDT_LAMBDAS[@]}"
echo "  Perturbation seeds:     ${#PERTURBATION_SEEDS[@]}"
echo "  Non-comm λ samples:     ${#NONCOMM_LAMBDAS[@]}"
echo "  Basis seeds per config: $NUM_BASIS_SEEDS"
echo "  ───────────────────────────────"
echo "  Total configurations:   $TOTAL_JOBS"
echo ""
echo "Results will be saved to:"
echo "  $BASE_SAVE_DIR/"
echo ""
echo "To monitor progress:"
echo "  tail -f ${OUTPUT_DIR}/*.out"
echo ""
echo "To check job status:"
echo "  qstat -u \$USER"
echo "======================================================================"
