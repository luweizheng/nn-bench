#!/bin/bash

# set environment variable
source set_env.sh

platform="npu"
compute_type="backward"
dtype="float32"
amp_level="O2"

for batch_size in 64 128 256
do
    filename=bs_${batch_size}
    echo $filename
    python3 ../train.py --platform ${platform} \
                --amp \
                --opt-level ${amp_level} \
                # 1>../output/${platform}/$filename 2>&1 &
    wait
done