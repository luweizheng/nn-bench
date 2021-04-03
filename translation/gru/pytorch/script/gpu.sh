#!/bin/bash

#SBATCH --job-name=gru_nnbench
#SBATCH --nodes=1
#SBATCH --partition=tesla
#SBATCH --gpus=1

# set up environment
source activate torch1.5

platform="gpu"
compute_type="backward"
dtype="float32"
amp_level="O2"

cd ..
mkdir -p output/${platform}

for batch_size in 64 128 256
do
    filename=bs_${batch_size}
    echo $filename
    python3 train.py --platform ${platform} \
                --amp \
                --opt-level ${amp_level} \
                1>output/${platform}/$filename 2>&1 &
    wait
done