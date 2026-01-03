QUEUE=${1:-long}
NCORES=${2:-1}
MEM_PER_CORE_GB=${3:-1}

SAVE_DIR=data/tests
SEED=42

JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE_GB \
    -o output/opt-fid-state-trail-testfile_%j.out \
    -c eps0.05 \
    /usr/bin/python3 src/solver.py \
    --non-commutative-norm 0 \
    --seed $SEED \
    --chi 8 \
    --num-players 6 \
    --dtype complex \
    --max-num-steps 1000 \
    --eps 0.05 \
    --num-perturbations 20 \
    --perturbation-method unitary \
    --subroutine-max-iter 1000 \
    --subroutine-lr 0.01 \
    --max-subroutine-lr 0.8 \
    --expl-check-interval 60 \
    --expl-maxiter 50 \
    --wandb-project quantum-nash-1 \
    --save-dir $SAVE_DIR \
    --wandb-experiment opt-fid-state-trail \
    --no-real-strategies
    2>&1)

JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)
echo "Job submitted with ID: $JOB_ID"