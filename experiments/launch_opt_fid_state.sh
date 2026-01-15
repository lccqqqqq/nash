QUEUE=${1:-long}
NCORES=${2:-1}
MEM_PER_CORE_GB=${3:-1}

SAVE_DIR=data/tests
SEED=42

JOB_OUTPUT=$(addqueue -q $QUEUE -n $NCORES -m $MEM_PER_CORE_GB \
    -o output/misc/opt-fid-state-product-init_%j.out \
    -c prod-init \
    /usr/bin/python3 src/solver.py \
    --non-commutative-norm 0.2 \
    --seed $SEED \
    --hamiltonian-seed 4002 \
    --chi 8 \
    --num-players 3 \
    --init-state product \
    --dtype complex \
    --max-num-steps 1000 \
    --eps 0.01 \
    --eps-schedule cosine \
    --num-perturbations 20 \
    --perturbation-method unitary \
    --subroutine-max-iter 1000 \
    --subroutine-lr 0.015 \
    --max-subroutine-lr 0.6 \
    --min-subroutine-lr 0.009 \
    --expl-check-interval 60 \
    --expl-maxiter 50 \
    --expl-threshold 1e-6 \
    --wandb-project quantum-nash-1 \
    --save-dir $SAVE_DIR \
    --wandb-experiment opt-fid-state-prod-init \
    --no-real-strategies \
    --compute-distance-to-ghz
    2>&1)

JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'python3-\K\d+' | head -1)
echo "Job submitted with ID: $JOB_ID"