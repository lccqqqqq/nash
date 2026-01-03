#!/bin/bash
# Gather equilibrium statistics for Haar random states
#
# This script runs equilibrium statistics gathering using Haar random initial states.
# The bond dimension is fixed to 16 to represent <=8 qubits faithfully (without truncation loss).
#
# Usage:
#   bash experiments/gather_equilibrium_stats_haar.sh [NUM_CORES] [NUM_PLAYERS] [NUM_SAMPLES_PER_CORE] [ALPHA] [TIMEOUT_MINUTES]
#
# Examples:
#   # Use defaults (10 cores, 3 players, 1 sample per core, alpha=0.06, 1800 min timeout)
#   bash experiments/gather_equilibrium_stats_haar.sh
#
#   # Custom parameters
#   bash experiments/gather_equilibrium_stats_haar.sh 20 3 100 0.06 240

# Parse command-line arguments with defaults
NUM_CORES=${1:-10}
NUM_PLAYERS=${2:-3}
NON_COMMUTATIVE_NORM=${3:-0.0}
NUM_SAMPLES_PER_CORE=${4:-1}
ALPHA=${5:-0.06}
TIMEOUT_MINUTES=${6:-1800}

# Generate UUID for this run
RUN_UUID=$(python3 -c "import uuid; print(str(uuid.uuid4())[:8])")

# Set save directory with UUID
SAVE_DIR="data/equilibrium_stats/haar_${RUN_UUID}"

echo "========================================================================"
echo "Equilibrium Statistics Gathering - Haar Random States"
echo "========================================================================"
echo "Configuration:"
echo "  Run UUID: $RUN_UUID"
echo "  Number of cores: $NUM_CORES"
echo "  Number of players: $NUM_PLAYERS"
echo "  Non-commutative norm: $NON_COMMUTATIVE_NORM"
echo "  Bond dimension (chi): 16 (fixed for Haar measure)"
echo "  Samples per core: $NUM_SAMPLES_PER_CORE"
echo "  Alpha: $ALPHA"
echo "  Random measure: haar"
echo "  Save directory: $SAVE_DIR"
echo "  Timeout: $TIMEOUT_MINUTES minutes"
echo "========================================================================"
echo ""

# Submit job
echo "Submitting Haar random state job..."
JOB_OUTPUT=$(addqueue -q gpulong --gpus 1 -n $NUM_CORES -m 1 -c "haar 5hrs" \
    /usr/bin/python3 src/gather_equilibrium_stats.py \
    --num-players $NUM_PLAYERS \
    --chi 16 \
    --num-samples $NUM_SAMPLES_PER_CORE \
    --save-dir $SAVE_DIR \
    --verbose \
    --max-iter 1000 \
    --alpha $ALPHA \
    --convergence-threshold 1e-6 \
    --expl-threshold 1e-3 \
    --rand-measure haar \
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
echo ""

echo "========================================================================"
echo "Monitoring job progress..."
echo "========================================================================"

# Monitor job with timeout
CHECK_INTERVAL=60  # Check every 60 seconds
ELAPSED=0

while [ $ELAPSED -lt $((TIMEOUT_MINUTES * 60)) ]; do
    # Check if job is still running
    if ! squeue -j $JOB_ID &>/dev/null; then
        echo ""
        echo "Job completed after $ELAPSED seconds!"
        break
    fi

    # Show status every minute
    echo "[$((ELAPSED / 60)) min] Job $JOB_ID still running..."
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

# Check if we timed out
if squeue -j $JOB_ID &>/dev/null; then
    echo ""
    echo "WARNING: Timeout reached after $TIMEOUT_MINUTES minutes."
    echo "Job may still be running. Check manually with: squeue -u $(whoami)"
    exit 1
fi

# Consolidate results
echo ""
echo "========================================================================"
echo "Consolidating results..."
echo "========================================================================"

# Check if save directory exists and has files
if [ -d "$SAVE_DIR" ] && ls $SAVE_DIR/eqdata_randstates_*.pkl 1> /dev/null 2>&1; then
    echo "Processing Haar random state results..."

    # Count files to determine if concatenation is needed
    NUM_FILES=$(ls -1 $SAVE_DIR/eqdata_randstates_*.pkl 2>/dev/null | wc -l)

    if [ $NUM_FILES -gt 1 ]; then
        # Multiple files - concatenate them
        echo "  Concatenating $NUM_FILES files..."
        python cat_pkl.py --input-dir $SAVE_DIR --pattern 'eqdata_randstates_*.pkl' --output-filename "eqdata_randstates_haar_combined.pkl"

        # Move the concatenated file to the directory
        mv $SAVE_DIR/eqdata_randstates_haar_combined.pkl $SAVE_DIR/
        echo "  Created: eqdata_randstates_haar_combined.pkl"
    elif [ $NUM_FILES -eq 1 ]; then
        # Single file - optionally rename it
        FILE=$(ls $SAVE_DIR/eqdata_randstates_*.pkl)
        BASENAME=$(basename "$FILE")
        echo "  Single file found: $BASENAME"
        echo "  No concatenation needed."
    fi
else
    echo "WARNING: No results found in $SAVE_DIR"
    exit 1
fi

echo ""
echo "========================================================================"
echo "All done!"
echo "Results saved to: $SAVE_DIR"
if [ $NUM_FILES -gt 1 ]; then
    echo "  - eqdata_randstates_haar_combined.pkl (concatenated from $NUM_FILES files)"
else
    echo "  - $(basename $(ls $SAVE_DIR/eqdata_randstates_*.pkl))"
fi
echo "========================================================================"
