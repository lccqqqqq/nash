#!/bin/bash -l
echo =========================================================   
echo Job submitted  date = Sun 11 Jan 21:44:18 GMT 2026      
date_start=`date +%s`
echo $SLURM_JOB_NUM_NODES nodes \( $SLURM_CPUS_ON_NODE processes per node \)        
echo $SLURM_JOB_NUM_NODES hosts used: $SLURM_JOB_NODELIST      
# Set this otherwise a different transport gets selected on some nodes and things break in strange ways
export OMPI_MCA_pml=^cm
echo Job output begins                                           
echo -----------------                                           
echo   
#hostname

# Need to set the max locked memory very high otherwise IB can't allocate enough and fails with "UCX  ERROR Failed to allocate memory pool chunk: Input/output error"
ulimit -l unlimited

export OMP_NUM_THEADS=1
 /usr/local/shared/slurm/bin/srun -u -n 1 --mpi=pmix --mem-per-cpu=1024 nice -n 10 /usr/bin/python3 src/solver.py --non-commutative-norm 0 --seed 16030036 --chi 16 --num-players 6 --dtype real --max-num-steps 1000 --eps 0.02 --eps-schedule cosine --num-perturbations 20 --perturbation-method unitary --gradient-method ols --ridge-lam 0.01 --subroutine-max-iter 1000 --subroutine-lr 0.02 --max-subroutine-lr 0.8 --min-subroutine-lr 0.01 --expl-check-interval 50 --expl-maxiter 80 --expl-threshold 1e-7 --wandb-project quantum-nash-new-solver --wandb-experiment adaptiveLR --save-dir data/qpd6_new_solver_test_adaptiveLR/chi_16/ols_ridge0.01 --real-strategies
  echo ---------------                                           
  echo Job output ends                                           

  date_end=`date +%s`
  seconds=$((date_end-date_start))
  minutes=$((seconds/60))
  seconds=$((seconds-60*minutes))
  hours=$((minutes/60))
  minutes=$((minutes-60*hours))
  echo =========================================================   
  echo PBS job: finished   date = `date`   
  echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
  echo =========================================================
