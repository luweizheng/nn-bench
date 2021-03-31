#!/bin/bash

source npu_env.sh

cd ../pytorch

dtype="float16"
compute_type="forward"
platform="npu"
mkdir -p ../output/${platform}_conv2d

for batch_size in 128 512 1024
do
    for input_size in 224 448
    do
        for in_channels in 32 64 128
        do
            for out_channels in 64 128 256
            do
                for filter_size in 1 3 5 7
                do  
                    filename=bs_${batch_size}-input_${input_size}-inchannel_${in_channels}-outchannel_${out_channels}-filtersize_${filter_size}
                    echo $filename
                    
                    if [ -e ../output/${platform}_conv2d/${filename} ]
                    then
                        echo "exist"
                    else
                        python3 conv2d.py --platform=${platform} \
                            --dtype ${dtype} \
                            --compute_type ${compute_type} \
                            --num_iterations 100 \
                            --input_tensor_shape ${batch_size} ${in_channels} ${input_size} ${input_size} \
                            --kernel_shape ${filter_size} ${filter_size} ${in_channels} ${out_channels} \
                            1>../output/${platform}_conv2d/$filename 2>&1 &
                    fi
                    wait
                done
            done
        done
    done
done