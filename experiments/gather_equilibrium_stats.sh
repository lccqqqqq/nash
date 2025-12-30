#!/bin/bash
# Template script for running equilibrium statistics gathering
#
# This script demonstrates how to run multiple independent copies of the
# gather_equilibrium_stats.py script with different seeds. The parallelization
# is handled by running multiple processes, not through MPI.
#
# Usage:
#   bash experiments/gather_equilibrium_stats.sh [NUM_CORES] [NUM_PLAYERS] [NUM_SAMPLES_PER_CORE] [ALPHA] [TIMEOUT_MINUTES] [RAND_MEASURE]
#
# Examples:
#   # Use defaults (10 cores, 3 players, 50 samples per core, alpha=0.06, 180 min timeout, haar measure)
#   bash experiments/gather_equilibrium_stats.sh
#
#   # Custom parameters
#   bash experiments/gather_equilibrium_stats.sh 20 4 100 0.1 240 rand_mps
#
# For cluster execution, replace the for loop with your cluster's job array
# or parallel submission commands (e.g., addqueue, sbatch, qsub).

# Parse command-line arguments with defaults
NUM_CORES=${1:-10}
NUM_PLAYERS=${2:-3}
NUM_SAMPLES_PER_CORE=${3:-1}
ALPHA=${4:-0.06}
RAND_MEASURE=${6:-rand_mps}


echo "========================================================================"
echo "Equilibrium Statistics Gathering - Parallel Execution"
echo "========================================================================"
echo "Configuration:"
echo "  Number of cores: $NUM_CORES"
echo "  Number of players: $NUM_PLAYERS"
echo "  Bond dimensions: ${BOND_DIMS[@]}"
echo "  Samples per core: $NUM_SAMPLES_PER_CORE"
echo "  Alpha: $ALPHA"
echo "  Random measure: $RAND_MEASURE"
echo "  Timeout: ${5:-1800} minutes"
echo "========================================================================"
echo ""

# Example: Run 5 independent copies with different seeds
# Each run will save its own eqdata_randstates_*.pkl file
# These can later be concatenated using cat_pkl.py

BOND_DIMS=(3 4 6 8)
JOB_IDS=()

for CHI in "${BOND_DIMS[@]}"; do
    echo "Submitting job for chi=$CHI..."

    # Capture addqueue output and extract job ID
    JOB_OUTPUT=$(addqueue -q long -n $NUM_CORES -m 1 -c "chi$CHI 5hrs" /usr/bin/python3 src/gather_equilibrium_stats.py \
        --num-players $NUM_PLAYERS \
        --chi $CHI \
        --num-samples $NUM_SAMPLES_PER_CORE \
        --save-dir data/equilibrium_stats/chi$CHI \
        --verbose \
        --max-iter 1000 \
        --alpha $ALPHA \
        --convergence-threshold 1e-6 \
        --expl-threshold 1e-3 \
        --rand-measure $RAND_MEASURE \
        --non-commutative-norm $NON_COMMUTATIVE_NORM 2>&1)

    # Extract job ID from output like "python3-2571502.out"
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)

    if [ -n "$JOB_ID" ]; then
        JOB_IDS+=("$JOB_ID")
        echo "  Job ID: $JOB_ID (chi=$CHI)"
    else
        echo "  WARNING: Could not extract job ID from output"
        echo "$JOB_OUTPUT"
    fi
    echo ""
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs total."

echo "========================================================================"
echo "Monitoring job progress..."
echo "========================================================================"

# Monitor jobs with timeout
TIMEOUT_MINUTES=${5:-1800}  # Default 30 hours, can override as 5th argument
CHECK_INTERVAL=60  # Check every 60 seconds
ELAPSED=0

while [ $ELAPSED -lt $((TIMEOUT_MINUTES * 60)) ]; do
    # Count how many of our jobs are still running/pending
    RUNNING=0
    for JOB_ID in "${JOB_IDS[@]}"; do
        if squeue -j $JOB_ID &>/dev/null; then
            ((RUNNING++))
        fi
    done

    if [ $RUNNING -eq 0 ]; then
        echo ""
        echo "All jobs completed after $ELAPSED seconds!"
        break
    fi

    # Show status every minute
    echo "[$((ELAPSED / 60)) min] $RUNNING/${#JOB_IDS[@]} jobs still running..."
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

if [ $RUNNING -gt 0 ]; then
    echo ""
    echo "WARNING: Timeout reached after $TIMEOUT_MINUTES minutes."
    echo "$RUNNING jobs may still be running. Check manually with: squeue -u $(whoami)"
    exit 1
fi

# Concatenate within each chi directory, then flatten structure
echo ""
echo "========================================================================"
echo "Consolidating results..."
echo "========================================================================"

MAIN_DIR="data/equilibrium_stats"

for CHI in "${BOND_DIMS[@]}"; do
    CHI_DIR="$MAIN_DIR/chi$CHI"

    # Check if chi subdirectory exists and has files
    if [ -d "$CHI_DIR" ] && ls $CHI_DIR/eqdata_randstates_*.pkl 1> /dev/null 2>&1; then
        echo "Processing chi=$CHI..."

        # Count files to determine if concatenation is needed
        NUM_FILES=$(ls -1 $CHI_DIR/eqdata_randstates_*.pkl 2>/dev/null | wc -l)

        if [ $NUM_FILES -gt 1 ]; then
            # Multiple files - concatenate them
            echo "  Concatenating $NUM_FILES files for chi=$CHI..."
            python cat_pkl.py --input-dir $CHI_DIR --pattern 'eqdata_randstates_*.pkl' --output-filename "eqdata_randstates_chi${CHI}_combined.pkl"

            # Move the concatenated file to main directory
            mv $CHI_DIR/eqdata_randstates_chi${CHI}_combined.pkl $MAIN_DIR/
            echo "  Created: eqdata_randstates_chi${CHI}_combined.pkl"
        elif [ $NUM_FILES -eq 1 ]; then
            # Single file - just rename and move it
            FILE=$(ls $CHI_DIR/eqdata_randstates_*.pkl)
            mv "$FILE" "$MAIN_DIR/eqdata_randstates_chi${CHI}_combined.pkl"
            echo "  Moved single file: eqdata_randstates_chi${CHI}_combined.pkl"
        fi

        # Remove the chi subdirectory
        rm -rf "$CHI_DIR"
        echo "  Removed directory: chi$CHI/"
    else
        echo "WARNING: No results found in $CHI_DIR"
    fi
done

echo ""
echo "========================================================================"
echo "All done!"
echo "Results per chi value saved to: $MAIN_DIR"
echo "  - eqdata_randstates_chi3_combined.pkl"
echo "  - eqdata_randstates_chi4_combined.pkl"
echo "  - eqdata_randstates_chi6_combined.pkl"
echo "  - eqdata_randstates_chi8_combined.pkl"
echo "========================================================================"


