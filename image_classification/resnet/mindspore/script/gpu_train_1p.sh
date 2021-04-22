#!/bin/bash

#SBATCH --job-name=image_ms
#SBATCH --nodes=1
#SBATCH --partition=tesla
#SBATCH --gpus=1

# set up environment
source activate torch1.5
export export LD_LIBRARY_PATH=~/.conda/envs/torch1.5/lib:lib64:$LD_LIBRARY_PATH

platform="gpu"
arch="resnet50"
dataset="cifar10"
# dataset="imagenet2012"
dataset_path="/ssd/CIFAR10"
# dataset_path="/disk/Datasets/ImageNet/ILSVRC2012"
batch_size=256

cd ..
mkdir -p ./output/${platform}/${arch}_${dataset}/

python3 train.py \
    --net=${arch} \
    --device_target="GPU" \
    --dataset=${dataset} \
    --dataset_path=${dataset_path} \
    --batch_size=${batch_size} \
    # &> ./output/${platform}/${arch}_${dataset}/bs_${batch_size}.log &