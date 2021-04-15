#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
amp_level="O2"

cd ..
mkdir -p output/${platform}/train_1p/

for batch_size in 512 # 64 128 256 512
do
    filename=bs_${batch_size}
    echo $filename
    python3 train_1p.py \
        --platform ${platform} \
        --device-id 0 \
        --workers 40 \
        --data "/disk/Datasets/multi30k" \
        --batch-size ${batch_size} \
        --epochs 50 \
        --amp  \
        --amp-level ${amp_level} \
        > output/${platform}/train_1p/${filename} 2>&1 &
    wait
done