echo "Testing for the optimal learning rate scaling for 6-player clean QPD"

NUM_PLAYERS=6
NON_COMMUTATIVE_NORM=0 # Important!
ALPHA=0.01
MAX_ALPHA=0.8
NUM_JOBS_PER_EPS=30
CHI=8
INIT_SEED=42

QUEUE="long"
NCORES=1
MEM_PER_CORE=1

BASE_SAVE_DIR="data/qpd6_lrtest"
EPS_SET=(0.001 0.002 0.004 0.006 0.008 0.01)

for EPS in ${EPS_SET[@]}; do
    echo "Submitting jobs with learning rate $EPS..."
    # We are using the same seed for all jobs because we have set the 
    SAVE_DIR="${BASE_SAVE_DIR}/eps_${EPS}"
    mkdir -p $SAVE_DIR

    for i in $(seq 1 $NUM_JOBS_PER_EPS); do
        SEED=$((INIT_SEED + i))

        JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE \
            -o output/qpd6_lrtest_eps${EPS}_job${i}_%j.out \
            /usr/bin/python3 src/solver.py \
            --non-commutative-norm $NON_COMMUTATIVE_NORM \
            --seed $SEED \
            --chi $CHI \
            --num-players $NUM_PLAYERS \
            --dtype complex \
            --max-num-steps 900 \
            --eps $EPS \
            --num-perturbations 20 \
            --perturbation-method unitary \
            --subroutine-max-iter 1000 \
            --subroutine-lr $ALPHA \
            --max-subroutine-lr $MAX_ALPHA \
            --expl-check-interval 50 \
            --expl-maxiter 100 \
            --wandb-project quantum-nash-1 \
            --save-dir $SAVE_DIR \
            --no-real-strategies \
            --wandb-experiment lrtesting
            2>&1)
        
        # Extract job ID
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)

        if [ -n "$JOB_ID" ]; then
            # Print progress every 50 jobs
            if [ $((i % 50)) -eq 0 ]; then
                echo "  EPS=$EPS: Submitted $i/$NUM_JOBS_PER_EPS jobs (latest: $JOB_ID)"
            fi
        else
            echo "  WARNING: Job $i for EPS=$EPS failed to submit"
            echo "$JOB_OUTPUT"
        fi
    done

    echo "  EPS=$EPS: Completed - submitted $NUM_JOBS_PER_EPS jobs"
    echo ""
done

echo "======================================================================"
echo "All jobs submitted!"
echo "Total jobs: $((${#EPS_SET[@]} * NUM_JOBS_PER_EPS))"
echo ""
echo "Results will be saved to: $BASE_SAVE_DIR/eps_*/"
echo "Use cat_pkl.py to concatenate results after jobs complete"
echo "======================================================================"
