NUM_CORES=${1:-10}
NUM_PLAYERS=${2:-3}
NON_COMMUTATIVE_NORM=${3:-0.8}
NUM_SAMPLES_PER_CORE=${4:-1}
ALPHA=${5:-0.15}
MAX_ALPHA=${6:-0.4}


SAVE_DIR="data/equilibrium_stats_v1/noncomm_norm_${NON_COMMUTATIVE_NORM//./p}"

JOB_OUTPUT=$(addqueue -q long -n $NUM_CORES -m 1 -c "v1 5hrs" /usr/bin/python3 src/gather_equilibrium_stats.py \
    --num-players $NUM_PLAYERS \
    --seed 42 \
    --chi 4 \
    --dtype complex64 \
    --num-samples $NUM_SAMPLES_PER_CORE \
    --save-dir $SAVE_DIR \
    --max-iter 1000 \
    --alpha $ALPHA \
    --max-alpha $MAX_ALPHA \
    --max-retries 20 \
    --convergence-threshold 1e-6 \
    --expl-threshold 1e-3 \
    --expl-check-interval 50 \
    --expl-maxiter 100 \
    --rand-measure haar \
    --no-real-strategies \
    --non-commutative-norm $NON_COMMUTATIVE_NORM 2>&1)

# Extract job ID from output like "python3-2571502.out"
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)

if [ -n "$JOB_ID" ]; then
    echo "  Job ID: $JOB_ID"
else
    echo "  WARNING: Could not extract job ID from output"
    echo "$JOB_OUTPUT"
    exit 1
fi

