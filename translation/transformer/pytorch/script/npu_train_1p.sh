#!/bin/bash

source set_env.sh

cd ..

DATA_DIR=./data/wmt14_en_de_joined_dict/
MODELDIR="./checkpoints/"
mkdir -p "$MODELDIR"
LOGFILE="$MODELDIR/log"
STAT_FILE="log.txt"
BATCH_SIZE=128
AMP_LEVEL="O2"
TRAIN_LOG=test/1p_${AMP_LEVEL}_b${BATCH_SIZE}
mkdir -p ${TRAIN_LOG}


export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export PTCOPY_ENABLE=1
export TASK_QUEUE_ENABLE=1
export DYNAMIC_OP="ADD#MUL"

python3 -u train_1p.py \
  --data $DATA_DIR \
  --platform "npu" \
  --device-id 0\
  --arch transformer_wmt_en_de \
  --source-lang en \
  --target-lang de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.997 \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences ${BATCH_SIZE}\
  --max-tokens 102400\
  --seed 1 \
  --save-dir $MODELDIR \
  --save-interval 1\
  --update-freq 8\
  --log-interval 1\
  --stat-file $STAT_FILE\
  --distributed-world-size 1\
  --amp\
  --amp-level $AMP_LEVEL \
  # > ${TRAIN_LOG}/train.log 2>&1 &