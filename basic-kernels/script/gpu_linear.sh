#!/bin/bash

source activate torch16

cd ../pytorch

python3 linear.py --platform="gpu"