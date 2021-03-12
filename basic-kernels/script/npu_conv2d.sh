#!/bin/bash

source npu_env.sh

cd ../pytorch

python3 conv2d.py --platform="npu" \
                --dtype float16 \
                --num_iterations 100