#!/bin/bash

# set npu environments
source npu_env.sh

cd ../pytorch

platform="npu"
dtype="float16"
compute_type="forward"
mkdir -p ../output/${platform}_rnn


for batch_size in 64 #64 128 256 512
do
    for seq_len in 128 #512 2048 16384 32768 #128 256 512 1024 32768
    do
        for embeding_size in 1024 2048 4096 #512 2048 16384 32768 #128 256 512 1024 32768
        do
            for hidden_size in 4096 #512 2048 16384 32768 #128 256 512 1024
            do
                filename=${platform}-${compute_type}-${dtype}-bs_${batch_size}_-seq_len-${seq_len}-embeding_size-${embeding_size}-hidden_size_${hidden_size}
                echo $filename

                python3 -W ignore rnn.py --platform ${platform} \
                    --compute_type ${compute_type} \
                    --dtype ${dtype} \
                    --input_tensor_shape ${seq_len} ${batch_size} ${embeding_size} \
                    --rnn_type rnn \
                    --hidden_size ${hidden_size} \
                    1>../output/${platform}_rnn/${filename} 2>&1 &
                wait
            done
        done
    done
done