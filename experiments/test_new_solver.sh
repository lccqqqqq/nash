#!/bin/bash
# Test new solver features: gradient methods, ridge regularization, and eps scheduling
# Submits jobs for different CHI values and gradient method configurations
#
# Usage: ./test_new_solver.sh [NUM_PLAYERS] [NON_COMMUTATIVE_NORM] [ALPHA] [MAX_ALPHA] [MIN_ALPHA] [NUM_JOBS_PER_CONFIG] [EXPL_THRESHOLD]
# Defaults: NUM_PLAYERS=6, NON_COMMUTATIVE_NORM=0, ALPHA=0.01, MAX_ALPHA=0.8, MIN_ALPHA=0.001, NUM_JOBS_PER_CONFIG=10, EXPL_THRESHOLD=1e-8

# Configuration
NUM_PLAYERS=${1:-6}
NON_COMMUTATIVE_NORM=${2:-0}
ALPHA=${3:-0.02}
MAX_ALPHA=${4:-0.8}
MIN_ALPHA=${5:-0.01}  # Min LR for adaptive retry (should be < ALPHA)
NUM_JOBS_PER_CONFIG=${6:-20}

EXPL_THRESHOLD=${7:-1e-7}
EXPERIMENT_NAME=${8:-"newdtype-explmaxiter200"}

# CHI values to test (smaller set for testing new features)
CHIS=(2 4 6 8 16)

# Gradient method configurations to test
# Format: "gradient_method:ridge_lam:description"
CONFIGS=(
    "max_welfare:0:maxwelfare"
    "ols:0:ols_noridge"
    "ols:0.01:ols_ridge0.01"
)

# Queue settings
QUEUE="long"
NCORES=1  # Number of cores per job
MEM_PER_CORE=1.8  # GB per core

# Base save directory (new directory for testing new features)
BASE_SAVE_DIR="data/qpd6_new_solver_test_${EXPERIMENT_NAME}"

echo "======================================================================"
echo "Testing New Solver Features: Gradient Methods + Ridge + Eps Scheduling"
echo "======================================================================"
echo "Configuration:"
echo "  Number of players: $NUM_PLAYERS"
echo "  Non-commutative norm: $NON_COMMUTATIVE_NORM"
echo "  Alpha (base): $ALPHA"
echo "  Max alpha: $MAX_ALPHA"
echo "  Min alpha: $MIN_ALPHA"
echo "  Expl threshold: $EXPL_THRESHOLD"
echo "  Jobs per configuration: $NUM_JOBS_PER_CONFIG"
echo "  CHI values: ${CHIS[@]}"
echo "  Gradient configurations: ${#CONFIGS[@]}"
for config in "${CONFIGS[@]}"; do
    desc=$(echo "$config" | cut -d: -f3)
    echo "    - $desc"
done
echo "  Eps schedule: cosine (enabled for all jobs)"
echo "  Queue: $QUEUE"
echo "======================================================================"
echo ""

# Initialize job counter
TOTAL_JOBS=0

# Loop over CHI values
for CHI in "${CHIS[@]}"; do
    echo "Processing CHI=$CHI..."

    # Loop over gradient method configurations
    for config in "${CONFIGS[@]}"; do
        # Parse configuration
        GRADIENT_METHOD=$(echo "$config" | cut -d: -f1)
        RIDGE_LAM=$(echo "$config" | cut -d: -f2)
        CONFIG_DESC=$(echo "$config" | cut -d: -f3)

        echo "  Configuration: $CONFIG_DESC (method=$GRADIENT_METHOD, ridge_lam=$RIDGE_LAM)"

        # Save directory for this CHI and configuration
        SAVE_DIR="${BASE_SAVE_DIR}/chi_${CHI}/${CONFIG_DESC}"
        mkdir -p $SAVE_DIR

        # Submit jobs for this configuration
        for i in $(seq 1 $NUM_JOBS_PER_CONFIG); do
            # Each job gets a unique seed = 1000000 * CHI + config_index * 10000 + i
            # This ensures different initializations across CHI, configs, and jobs
            CONFIG_INDEX=$(printf "%s\n" "${CONFIGS[@]}" | grep -n "^${config}$" | cut -d: -f1)
            SEED=$((1000001 * CHI + CONFIG_INDEX * 10000 + i))

            JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE \
                -o output/qpd_new_solver_test_${EXPERIMENT_NAME}/%j_chi${CHI}_${CONFIG_DESC}_job${i}.out \
                -c solver_test \
                /usr/bin/python3 src/solver.py \
                --non-commutative-norm $NON_COMMUTATIVE_NORM \
                --seed $SEED \
                --chi $CHI \
                --num-players $NUM_PLAYERS \
                --init-state product \
                --dtype real \
                --max-num-steps 1000 \
                --eps 0.02 \
                --eps-schedule cosine \
                --num-perturbations 20 \
                --perturbation-method unitary \
                --gradient-method $GRADIENT_METHOD \
                --ridge-lam $RIDGE_LAM \
                --subroutine-max-iter 1000 \
                --subroutine-lr $ALPHA \
                --max-subroutine-lr $MAX_ALPHA \
                --min-subroutine-lr $MIN_ALPHA \
                --expl-check-interval 50 \
                --expl-maxiter 200 \
                --expl-threshold $EXPL_THRESHOLD \
                --wandb-project quantum-nash-new-solver \
                --wandb-experiment $EXPERIMENT_NAME \
                --save-dir $SAVE_DIR \
                --real-strategies \
                --no-compute-distance-to-ghz \
                2>&1)

            # Extract job ID
            JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)

            if [ -n "$JOB_ID" ]; then
                # Print progress every 10 jobs
                if [ $((i % 10)) -eq 0 ]; then
                    echo "    Submitted $i/$NUM_JOBS_PER_CONFIG jobs (latest: $JOB_ID)"
                fi
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            else
                echo "    WARNING: Job $i failed to submit"
                echo "$JOB_OUTPUT"
            fi
        done

        echo "    Completed - submitted $NUM_JOBS_PER_CONFIG jobs for $CONFIG_DESC"
        echo ""
    done

    echo "  CHI=$CHI: All configurations completed"
    echo ""
done

echo "======================================================================"
echo "All jobs submitted!"
echo "======================================================================"
echo "Summary:"
echo "  Total CHI values: ${#CHIS[@]}"
echo "  Gradient configurations: ${#CONFIGS[@]}"
echo "  Jobs per configuration: $NUM_JOBS_PER_CONFIG"
echo "  Total jobs submitted: $TOTAL_JOBS"
echo ""
echo "Features tested:"
echo "  ✓ Gradient methods: max_welfare, ols"
echo "  ✓ Ridge regularization: 0, 0.01"
echo "  ✓ Eps scheduling: cosine"
echo "  ✓ Adaptive retry: LR ∈ [$MIN_ALPHA, $MAX_ALPHA]"
echo ""
echo "Results will be saved to:"
for CHI in "${CHIS[@]}"; do
    echo "  $BASE_SAVE_DIR/chi_${CHI}/<config>/"
done
echo ""
echo "Use cat_pkl.py to concatenate results after jobs complete"
echo "======================================================================"
