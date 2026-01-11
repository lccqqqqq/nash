#!/bin/bash
# Job submission script for CMA-ES fiducial state optimization
# Usage: ./experiments/run_cmaes.sh [NUM_PLAYERS] [CHI] [POP_SIZE] [N_GENERATIONS] [SEED] [STRICT_MODE]

# Configuration
NUM_PLAYERS=${1:-6}
CHI=${2:-4}
POP_SIZE=${3:-64}
N_GENERATIONS=${4:-1000}
SEED=${5:-42}
STRICT_MODE=${6:-1}  # Set to 1 to enable strict Nash validation (slow but accurate)

# Nash solver settings
NASH_ALPHA=${7:-0.01}
NASH_ITERS=${8:-3000}
VALIDATION_INTERVAL=${9:-50}
EXPL_THRESHOLD=${10:-1e-4}

# Queue settings
QUEUE="heraclesgpu"
NCORES=1
MEM_PER_CORE=4  # GB per core (CMA-ES can use more memory)

# Output directory
OUTPUT_DIR="output/cmaes"
mkdir -p $OUTPUT_DIR

# Save directory
SAVE_DIR="data/cmaes/L${NUM_PLAYERS}_chi${CHI}"
mkdir -p $SAVE_DIR

echo "======================================================================"
echo "Submitting CMA-ES Optimization Job"
echo "======================================================================"
echo "Configuration:"
echo "  Number of players (L): $NUM_PLAYERS"
echo "  Bond dimension (χ): $CHI"
echo "  Population size: $POP_SIZE"
echo "  Generations: $N_GENERATIONS"
echo "  Seed: $SEED"
if [ "$STRICT_MODE" -eq 1 ]; then
    echo "  Mode: STRICT (exploitability checked for every solution)"
else
    echo "  Mode: FAST (periodic validation only)"
fi
echo ""
echo "Nash Solver Settings:"
echo "  Learning rate (alpha): $NASH_ALPHA"
echo "  Iterations: $NASH_ITERS"
echo "  Validation interval: $VALIDATION_INTERVAL"
echo "  Exploitability threshold: $EXPL_THRESHOLD"
echo ""
echo "Queue: $QUEUE"
echo "Cores: $NCORES"
echo "Memory: ${MEM_PER_CORE}GB per core"
echo "======================================================================"
echo ""

# Build command with optional strict mode flag
CMD="/usr/bin/python3 cmaes.py \
    --L $NUM_PLAYERS \
    --chi $CHI \
    --pop-size $POP_SIZE \
    --n-generations $N_GENERATIONS \
    --nash-alpha $NASH_ALPHA \
    --nash-iters $NASH_ITERS \
    --validation-interval $VALIDATION_INTERVAL \
    --expl-threshold $EXPL_THRESHOLD \
    --seed $SEED \
    --save-dir $SAVE_DIR" 

# Add strict mode flag if enabled
if [ "$STRICT_MODE" -eq 1 ]; then
    CMD="$CMD --strict-nash --validation-interval 50"
fi

# Submit job
JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE --gpus 1 --gputype a100with80gb \
    -c cmaes_opt \
    $CMD \
    2>&1)


echo ""
echo "Monitor job status with: showq | grep $JOB_ID"
echo "View output with: tail -f $OUTPUT_DIR/${JOB_ID}_L${NUM_PLAYERS}_chi${CHI}_seed${SEED}.out"
echo "======================================================================"
