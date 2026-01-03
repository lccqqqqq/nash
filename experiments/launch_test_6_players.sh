#!/bin/bash
# Submit 1000 jobs for each CHI value
# Each job runs gather_equilibrium_stats.py with a single random initialization

# Configuration
NUM_PLAYERS=${1:-6}
NON_COMMUTATIVE_NORM=${2:-0}
ALPHA=${3:-0.01}
MAX_ALPHA=${4:-0.8}
NUM_JOBS_PER_CHI=${5:-30}

# CHI values to sweep over
CHIS=(2 3 4 5 6 7 8 9)

# Queue settings
QUEUE="long"
NCORES=1  # Number of cores per job
MEM_PER_CORE=1  # GB per core

# Base save directory
BASE_SAVE_DIR="data/qpd6"

echo "======================================================================"
echo "Submitting Chi Sweep Jobs for Equilibrium Statistics"
echo "======================================================================"
echo "Configuration:"
echo "  Number of players: $NUM_PLAYERS"
echo "  Non-commutative norm: $NON_COMMUTATIVE_NORM"
echo "  Alpha (base): $ALPHA"
echo "  Max alpha: $MAX_ALPHA"
echo "  Jobs per CHI: $NUM_JOBS_PER_CHI"
echo "  CHI values: ${CHIS[@]}"
echo "  Queue: $QUEUE"
echo "======================================================================"
echo ""

# Loop over CHI values
for CHI in "${CHIS[@]}"; do
    echo "Submitting $NUM_JOBS_PER_CHI jobs for CHI=$CHI..."

    # Save directory for this CHI
    SAVE_DIR="${BASE_SAVE_DIR}/chi_${CHI}"
    mkdir -p $SAVE_DIR

    # Submit jobs
    for i in $(seq 1 $NUM_JOBS_PER_CHI); do
        # Each job gets a unique seed = 1000000 * CHI + i
        # This ensures: (1) same Hamiltonian for same CHI, (2) different states across jobs
        # Since we only have 1 MPI rank per job, rank=0, so state_seed = SEED + 0 = SEED
        SEED=$((1000000 * CHI + i))

        JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE \
            -o output/qpd6_chi${CHI}_job${i}_%j.out \
            /usr/bin/python3 src/solver.py \
            --non-commutative-norm $NON_COMMUTATIVE_NORM \
            --seed $SEED \
            --chi $CHI \
            --num-players $NUM_PLAYERS \
            --dtype complex \
            --max-num-steps 800 \
            --eps 0.01 \
            --num-perturbations 20 \
            --perturbation-method unitary \
            --subroutine-max-iter 1000 \
            --subroutine-lr $ALPHA \
            --max-subroutine-lr $MAX_ALPHA \
            --expl-check-interval 50 \
            --expl-maxiter 100 \
            --wandb-project quantum-nash-1 \
            --save-dir $SAVE_DIR \
            --no-real-strategies
            2>&1)

        # Extract job ID
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)

        if [ -n "$JOB_ID" ]; then
            # Print progress every 50 jobs
            if [ $((i % 50)) -eq 0 ]; then
                echo "  CHI=$CHI: Submitted $i/$NUM_JOBS_PER_CHI jobs (latest: $JOB_ID)"
            fi
        else
            echo "  WARNING: Job $i for CHI=$CHI failed to submit"
            echo "$JOB_OUTPUT"
        fi
    done

    echo "  CHI=$CHI: Completed - submitted $NUM_JOBS_PER_CHI jobs"
    echo ""
done

echo "======================================================================"
echo "All jobs submitted!"
echo "Total jobs: $((${#CHIS[@]} * NUM_JOBS_PER_CHI))"
echo ""
echo "Results will be saved to: $BASE_SAVE_DIR/chi_*/"
echo "Use cat_pkl.py to concatenate results after jobs complete"
echo "======================================================================"
