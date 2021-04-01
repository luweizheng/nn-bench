#!/bin/bash

#SBATCH --job-name=nnbench
#SBATCH --nodes=1
#SBATCH --partition=tesla
#SBATCH --gpus=1

# set up environment
source activate torch1.5

cd ../pytorch

dtype="float16"
compute_type="forward"
platform="gpu"
mkdir -p ../output/${platform}_rnn

for batch_size in 64 128 256 512
do
    for seq_len in 128 256 512
    do
        for embeding_size in 512 1024 2048 4096 8192 16384
        do
            for hidden_size in 512 1024 2048 4096 8192 16384
            do
                filename=bs_${batch_size}-seqlen_${seq_len}-embedingsize_${embeding_size}-hiddensize_${hidden_size}
                echo $filename

                if [ -e ../output/${platform}_rnn/${filename} ]
                then
                    echo "exist"
                else
                    python3 -W ignore rnn.py --platform ${platform} \
                        --compute_type ${compute_type} \
                        --dtype ${dtype} \
                        --input_tensor_shape ${seq_len} ${batch_size} ${embeding_size} \
                        --rnn_type rnn \
                        --hidden_size ${hidden_size} \
                        1>../output/${platform}_rnn/${filename} 2>&1 &
                fi
                wait
            done
        done
    done
done