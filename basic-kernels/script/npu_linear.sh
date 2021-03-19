#!/bin/bash

# set npu environments
source npu_env.sh

cd ../pytorch

platform="npu"
dtype=float16
mkdir -p ../output/linear

for compute_type in forward # backward
do
    for batch_size in 1024 #64 128 256 512
    do
        for input_size in 32768 #512 2048 16384 32768 #128 256 512 1024
        do
            for output_size in 32768 #512 2048 16384 32768 #128 256 512 1024
            do
                filename=${platform}-${compute_type}-${dtype}-bs_${batch_size}_-input-${input_size}-output-${output_size}
                echo $filename

                python3 linear.py --platform $platform \
                    --compute_type $compute_type \
                    --dtype $dtype \
                    --input_tensor_shape $batch_size $input_size \
                    --kernel_shape $input_size $output_size \
                    1>../output/linear/$filename 2>&1 &
                wait
            done
        done
    done
done