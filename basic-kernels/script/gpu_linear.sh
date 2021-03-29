#!/bin/bash

#SBATCH --job-name=nnbench
#SBATCH --nodes=1
#SBATCH --partition=tesla
#SBATCH --gpus=1

# set up environment
source activate torch1.5

cd ../pytorch

dtype="float16"
compute_type="forward"
platform="gpu"
mkdir -p ../output/${platform}_linear

for batch_size in 256 512 1024
do
    for input_size in 1024 2048 4096 8192 32768 65536
    do
        for output_size in 1024 2048 4096 8192 32768 65536
        do
            filename=${platform}-${compute_type}-${dtype}-bs_${batch_size}_-input-${input_size}-output-${output_size}
            echo $filename

            python3 linear.py --platform ${platform} \
                --compute_type ${compute_type} \
                --dtype ${dtype} \
                --input_tensor_shape ${batch_size} ${input_size} \
                --kernel_shape ${input_size} ${output_size} \
                1>../output/${platform}_linear/$filename 2>&1 &
            wait
        done
    done
done