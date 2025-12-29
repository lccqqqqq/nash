# Create sweep from config file and launch the sweep on the hydra cluster
CONFIG_FILE=${1:-"configs/sweep_configs/sweep_config.yaml"}  # First argument or default
NUM_WORKERS=${2:-10}  # Second argument or default to 4
NUM_CORES=${3:-5}    # Third argument or default to 4

echo "Initializing sweep from configuration $CONFIG_FILE"
python3 experiments/setup_cluster_sweep.py --config $CONFIG_FILE --num-workers $NUM_WORKERS --count-per-worker 1
addqueue -q long -n $NUM_CORES -m 1 /usr/local/shared/bin/multirun commands.txt

echo "Job submitted"