#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
compute_type="backward"
dtype="float32"
amp_level="O2"

cd ..
mkdir -p output/${platform}/bench/

for batch_size in 64 128 256
do
    filename=bs_${batch_size}
    echo $filename
    python3 bench.py --platform ${platform} \
                --amp \
                --opt-level ${amp_level} \
                1>output/${platform}/bench/$filename 2>&1 &
    wait
done