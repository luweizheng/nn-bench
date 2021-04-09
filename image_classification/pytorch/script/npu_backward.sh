#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
compute_type="backward"
dtype="float32"

cd ../
mkdir -p output/${platform}

for arch in resnet50 resnet101 densenet121 densenet201
do
    for batch_size in 64 128 256
    do
        filename=${compute_type}-${arch}-bs_${batch_size}
        echo $filename
        python3 train.py --platform ${platform} \
                    --compute_type ${compute_type} \
                    --arch ${arch} \
                    --dtype ${dtype} \
                    --num_iterations 100 \
                    --amp \
                    --opt-level "O2" \
                    --input_tensor_shape ${batch_size} 3 224 224 \
                    1>./output/${platform}/$filename 2>&1 &
        wait
    done
done