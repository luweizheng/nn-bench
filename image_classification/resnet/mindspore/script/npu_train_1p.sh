#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
arch="resnet50"
dataset="imagenet2012"
dataset_path="/home/luweizheng/Datasets/ImageNet/ILSVRC2012"
batch_size=256

cd ..
mkdir -p ./output/${platform}/${arch}_${dataset}/

python3 train.py \
    --net=${arch} \
    --device_target="Ascend" \
    --dataset=${dataset} \
    --dataset_path=${dataset_path} \
    --batch_size=${batch_size} \
    &> ./output/${platform}/${arch}_${dataset}/bs_${batch_size}.log &