#!/bin/bash


DATASET_DIR=../data/wmt14_en_de_joined_dict
TEXT=../examples/translation/wmt14_en_de


(
  cd ../examples/translation
  bash prepare-wmt14en2de.sh --scaling18
)


# python3 ../examples/translation/preprocess.py \
#   --source-lang en \
#   --target-lang de \
#   --trainpref $TEXT/train \
#   --validpref $TEXT/valid \
#   --testpref $TEXT/test \
#   --destdir ${DATASET_DIR} \
#   --nwordssrc 33712 \
#   --nwordstgt 33712 \
#   --joined-dictionary


# cp $TEXT/code $DATASET_DIR/code
# cp $TEXT/tmp/valid.raw.de $DATASET_DIR/valid.raw.de
# sacrebleu -t wmt14/full -l en-de --echo ref > $DATASET_DIR/test.raw.de