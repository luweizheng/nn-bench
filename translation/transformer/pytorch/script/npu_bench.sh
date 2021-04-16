#! /bin/bash

source set_env.sh
cd ..

DATA_DIR=./data/wmt14_en_de_joined_dict/
platform="npu"
batch_size=128
AMP_LEVEL="O2"
TRAIN_LOG=output/${platform}/
mkdir -p ${TRAIN_LOG}

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"

for arch in transformer_wmt_en_de transformer_vaswani_wmt_en_de_big
do
    for batch_size in 64 128
    do
        filename=${arch}-bs_${batch_size}
        echo $filename
        python3 bench.py \
            $DATA_DIR \
            --platform ${platform} \
            --device-id 0 \
            --arch ${arch} \
            --source-lang en \
            --target-lang de \
            --share-all-embeddings \
            --dropout 0.1 \
            --max-sentences ${batch_size}\
            --max-tokens 102400\
            --seed 1 \
            --distributed-world-size 1\
            --amp\
            --amp-level $AMP_LEVEL \
            --clip-norm 0.0 \
            --lr-scheduler inverse_sqrt \
            > ${TRAIN_LOG}/${filename} 2>&1 &
            # --warmup-init-lr 0.0 \
            # --warmup-updates 4000 \
            # --lr 0.0006 \
            # --min-lr 0.0 \
            # --weight-decay 0.0 \
            # --criterion label_smoothed_cross_entropy \
            # --label-smoothing 0.1 \
            # --save-dir $MODELDIR \
            # --save-interval 1\
            # --update-freq 8\
            # --log-interval 1\
            # --stat-file $STAT_FILE\
            # --adam-beta1 0.9 \
            # --adam-beta2 0.997 \
            # --adam-eps "1e-9" \
            # --optimizer adam \
        wait
    done
done