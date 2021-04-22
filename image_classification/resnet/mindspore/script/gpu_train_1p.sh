#!/bin/bash

# set up environment
source activate ms1.1

platform="gpu"
arch="resnet50"
dataset="imagenet2012"
dataset_path="/disk/Datasets/ImageNet/ILSVRC2012"
batch_size=256

cd ..
mkdir -p ./output/${platform}/${arch}_${dataset}/

python3 train.py \
    --net=${arch} \
    --device_target="GPU" \
    --dataset=${dataset} \
    --dataset_path=${dataset_path} \
    --batch_size=${batch_size} \
    &> ./output/${platform}/${arch}_${dataset}/bs_${batch_size}.log &