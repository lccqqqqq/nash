#!/bin/bash
# Hamiltonian Perturbation Sweep for 3-Player Complex Strategy Experiments
# Sweeps over Hamiltonian seeds and non-commutativity levels

# Configuration
NUM_PLAYERS=3
DTYPE="complex"
REAL_STRATEGIES_FLAG="--no-real-strategies"  # Use complex strategies (SU(2))

# Hamiltonian seed range
HAMILTONIAN_SEEDS=(1000 2000)  # 5 different random Hamiltonians

# Non-commutativity levels for each Hamiltonian
NON_COMM_NORMS=($(python -c "import numpy as np; print(' '.join([str(x) for x in np.linspace(0, 0.2, 20)]))"))  # 20 values from 0 to 0.2 inclusive

# State initialization seeds (multiple random starts per config)
NUM_STATE_SEEDS_PER_CONFIG=10

# CHI values to test
CHI=6

# Other hyperparameters
ALPHA=0.02
MAX_ALPHA=0.6
MIN_ALPHA=0.001
EXPL_THRESHOLD=1e-7
MAX_NUM_STEPS=1000

# Queue settings
QUEUE="long"
NCORES=1
MEM_PER_CORE=1.8



# Enable distance to GHZ tracking
COMPUTE_DISTANCE_TO_GHZ="--compute-distance-to-ghz"
COMPUTE_DISTANCE_TO_GHZ_INTERVAL=10

# Experiment name
EXPERIMENT_NAME="hamiltonian_sweep_3p_complex_v1"
# Base save directory
BASE_SAVE_DIR="data/$EXPERIMENT_NAME"

echo "======================================================================"
echo "Hamiltonian Perturbation Sweep - 3-Player Complex Strategies"
echo "======================================================================"
echo "Configuration:"
echo "  Number of players: $NUM_PLAYERS"
echo "  Dtype: $DTYPE (complex strategies enabled)"
echo "  Hamiltonian seeds: ${HAMILTONIAN_SEEDS[@]}"
echo "  Non-commutativity levels: ${NON_COMM_NORMS[@]}"
echo "  State seeds per config: $NUM_STATE_SEEDS_PER_CONFIG"
echo "  CHI value: $CHI"
echo "  Distance to GHZ: Enabled (interval=$COMPUTE_DISTANCE_TO_GHZ_INTERVAL)"
echo "======================================================================"
echo ""

TOTAL_JOBS=0


for H_SEED in "${HAMILTONIAN_SEEDS[@]}"; do
    echo "  Hamiltonian seed: $H_SEED"

    for NON_COMM_NORM in "${NON_COMM_NORMS[@]}"; do
        echo "    Non-commutativity: $NON_COMM_NORM"

        # Save directory for this configuration
        SAVE_DIR="${BASE_SAVE_DIR}/hseed_${H_SEED}/noncomm_${NON_COMM_NORM}"
        mkdir -p $SAVE_DIR

        # Submit jobs with different state initialization seeds
        for i in $(seq 1 $NUM_STATE_SEEDS_PER_CONFIG); do
            # State seed = base + offset
            STATE_SEED=$((10000 * CHI + 100 * i))

            JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE \
                -o output/$EXPERIMENT_NAME/%j_hseed${H_SEED}_nc${NON_COMM_NORM}_job${i}.out \
                -c hamiltonian_sweep \
                /usr/bin/python3 src/solver.py \
                --num-players $NUM_PLAYERS \
                --dtype $DTYPE \
                $REAL_STRATEGIES_FLAG \
                --seed $STATE_SEED \
                --hamiltonian-seed $H_SEED \
                --non-commutative-norm $NON_COMM_NORM \
                --chi $CHI \
                --max-num-steps $MAX_NUM_STEPS \
                --eps 0.02 \
                --eps-schedule cosine \
                --num-perturbations 20 \
                --perturbation-method unitary \
                --gradient-method ols \
                --ridge-lam 0 \
                --subroutine-max-iter 1000 \
                --subroutine-lr $ALPHA \
                --max-subroutine-lr $MAX_ALPHA \
                --min-subroutine-lr $MIN_ALPHA \
                --expl-check-interval 50 \
                --expl-maxiter 200 \
                --expl-threshold $EXPL_THRESHOLD \
                $COMPUTE_DISTANCE_TO_GHZ \
                --compute-distance-to-ghz-interval $COMPUTE_DISTANCE_TO_GHZ_INTERVAL \
                --wandb-project quantum-nash-hamiltonian-sweep \
                --wandb-experiment $EXPERIMENT_NAME \
                --save-dir $SAVE_DIR \
                2>&1)

            # Extract job ID and track progress
            JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)

            if [ -n "$JOB_ID" ]; then
                if [ $((i % 5)) -eq 0 ]; then
                    echo "      Submitted $i/$NUM_STATE_SEEDS_PER_CONFIG jobs (latest: $JOB_ID)"
                fi
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            else
                echo "      WARNING: Job $i failed to submit"
            fi
        done

        echo "      Completed - $NUM_STATE_SEEDS_PER_CONFIG jobs for noncomm=$NON_COMM_NORM"
    done

    echo "    Hamiltonian seed $H_SEED: All non-commutativity levels completed"
done


echo "======================================================================"
echo "All jobs submitted!"
echo "======================================================================"
echo "Summary:"
echo "  Total CHI values: ${#CHIS[@]}"
echo "  Hamiltonian seeds: ${#HAMILTONIAN_SEEDS[@]}"
echo "  Non-commutativity levels: ${#NON_COMM_NORMS[@]}"
echo "  State seeds per config: $NUM_STATE_SEEDS_PER_CONFIG"
echo "  Total jobs submitted: $TOTAL_JOBS"
echo ""
echo "Experiment details:"
echo "  ✓ 3-player quantum game"
echo "  ✓ Complex strategies (full SU(2) group)"
echo "  ✓ Distance to GHZ orbit tracked"
echo "  ✓ Independent Hamiltonian randomization"
echo "  ✓ Hamiltonian saved in results for non-commutativity analysis"
echo ""
echo "Results will be saved to:"
echo "  $BASE_SAVE_DIR/chi_<CHI>/hseed_<H_SEED>/noncomm_<NON_COMM_NORM>/"
echo "======================================================================"
