#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <job_id_start> <job_id_end>"
    exit 1
fi

a=$1
b=$2

for ((job=$a; job<=$b; job++)); do
    echo "Cancelling job $job"
    scancel "$job"
done
