#!/bin/bash

#SBATCH --job-name=gru_nnbench
#SBATCH --nodes=1
#SBATCH --partition=tesla
#SBATCH --gpus=1

# set up environment
source activate torch1.5

platform="gpu"
amp_level="O2"

cd ..

for batch_size in 256 512
do
    filename=bs_${batch_size}
    echo $filename
    mkdir -p output/${platform}/train_1p/${filename}
    python3 train_1p.py \
        --platform ${platform} \
        --device-id 0 \
        --workers 32 \
        --data "~/Datasets/multi30k/" \
        --save-dir output/${platform}/train_1p/${filename}/ \
        --batch-size ${batch_size} \
        --epochs 50 \
        --amp  \
        --amp-level ${amp_level} \
        > output/${platform}/train_1p/${filename}/${filename}.log 2>&1 &
    wait
done