#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
amp_level="O2"
arch="resnet50"
cd ..

for batch_size in 128 256 512
do
    filename=bs_${batch_size}
    echo ${filename}
    mkdir -p output/${platform}/cifar10_224_train_1p/${filename}
    python3 train_1p_cifar10.py \
        --platform ${platform} \
        --device-id 0 \
        --arch ${arch} \
        --workers 32 \
        --data "~/Datasets/CIFAR10/" \
        --save-dir output/${platform}/cifar10_224_train_1p/${filename}/ \
        --batch-size ${batch_size} \
        --epochs 80 \
        --print-freq 50 \
        --amp  \
        --amp-level ${amp_level} \
        > output/${platform}/cifar10_224_train_1p/${filename}/${filename}.log 2>&1 &
    wait
done
