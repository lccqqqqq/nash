NUM_QUBITS=${1:-6}
N_SAMPLES=${2:-10}

# Define range of eps values to test
EPS_VALUES=(0.1 0.2 0.3 0.5 0.7 1.0)


# Loop over eps values
for eps in "${EPS_VALUES[@]}"; do
    echo "Testing eps=${eps}"
    addqueue -q long -n 10 -m 1 -c "orbscan" -o output/orbit_scan_L${NUM_QUBITS}_eps${eps}_%j.out \
        /usr/bin/python3 ghz_local_optimality.py \
        --eps ${eps} --n-samples ${N_SAMPLES}
done