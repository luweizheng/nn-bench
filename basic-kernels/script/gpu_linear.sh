#!/bin/bash

cd ../pytorch

source activate torch16
python3 linear.py --platform="gpu"