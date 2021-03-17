#!/bin/bash

source activate torch16

platform="gpu"
compute_type="forward"
dtype="float16"

cd ../pytorch

for arch in resnet50 resnet101 densenet121 densenet201
do
    for batch_size in 64 128 256
    do
        filename=${platform}-arch_${arch}-${compute_type}-${dtype}-bs_${batch_size}
        echo $filename
        python3 vision.py --platform ${platform} \
                    --compute_type ${compute_type} \
                    --arch ${arch} \
                    --dtype ${dtype} \
                    --num_iterations 1 \
                    --input_tensor_shape ${batch_size} 3 224 224 \
                    1>../output/${platform}/$filename 2>&1 &
        wait
    done
done