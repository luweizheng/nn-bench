#!/bin/bash

# set up environment
source activate torch16

cd ../pytorch

compute_type="forward"
platform="gpu"
mkdir -p ../output/linear

for dtype in float16
do
    for batch_size in 512 1024 #32 64 128 256 512
    do
        for input_size in 10240 32768 #128 256 512 1024
        do
            for output_size in 10240 32768 #128 256 512 1024
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