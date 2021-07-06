#!/bin/bash

cd ..

DATA=~/kaiyang/data
DASSL=~/kaiyang/code/Dassl.pytorch

SEED=1
DATASET=visda17
TRAINER=SemiMixStyle
NET=resnet101_ms_l12
S=synthetic
T=real

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--source-domains ${S} \
--target-domains ${T} \
--dataset-config-file ${DASSL}/configs/datasets/da/${DATASET}.yaml \
--config-file configs/trainers/semimixstyle/${DATASET}.yaml \
--output-dir output/${DATASET}/${TRAINER}/${NET} \
MODEL.BACKBONE.NAME ${NET}