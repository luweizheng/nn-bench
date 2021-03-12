#!/bin/bash

source activate torch16

cd ../pytorch

python3 conv2d.py --platform="gpu"