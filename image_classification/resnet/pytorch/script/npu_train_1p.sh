#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
amp_level="O2"
arch="resnet50"
cd ..

for batch_size in 512 # 64 128 256 512
do
    filename=bs_${batch_size}
    echo ${filename}
    mkdir -p output/${platform}/train_1p/${filename}
    python3 train_1p.py \
        --platform ${platform} \
        --device-id 0 \
        --arch ${arch} \
        --workers 32 \
        --data "/home/luweizheng/Datasets/ImageNet/ILSVRC2012" \
        --save-dir output/${platform}/train_1p/${filename}/ \
        --batch-size ${batch_size} \
        --epochs 80 \
        --amp  \
        --amp-level ${amp_level} \
        --loss-scale 1024 \
        > output/${platform}/train_1p/${filename}/log 2>&1 &
    wait
done