#!/bin/bash

# set npu environments
source npu_env.sh

cd ../pytorch

python3 linear.py --platform="npu"