#!/bin/bash

cd ../pytorch

source activate torch16
python3 conv2d.py --platform="gpu"