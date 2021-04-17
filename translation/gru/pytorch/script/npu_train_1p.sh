#!/bin/bash

# set up environment
source set_env.sh

platform="npu"
amp_level="O2"

cd ..

for batch_size in 256 512
do
    filename=bs_${batch_size}
    echo ${filename}
    mkdir -p output/${platform}/train_1p/${filename}
    python3 train_1p.py \
        --platform ${platform} \
        --device-id 0 \
        --workers 40 \
        --data "/home/luweizheng/modelzoo/built-in/PyTorch/Official/nlp/GRU_for_PyTorch/.data/multi30k" \
        --save-dir output/${platform}/train_1p/${filename}/ \
        --batch-size ${batch_size} \
        --epochs 30 \
        --amp  \
        --amp-level ${amp_level} \
        > output/${platform}/train_1p/${filename}/log 2>&1 &
    wait
done