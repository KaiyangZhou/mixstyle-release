#!/bin/bash

DATA=~/kaiyang/data

S1=$1
S2=$2
S3=$3
T=$4
NET=$5
SEED=$6
DATASET=$7
TRAINER=$8

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--source-domains ${S1} ${S2} ${S3} \
--target-domains ${T} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
--output-dir output/${DATASET}/${TRAINER}_${NET}/${T}/seed${SEED} \
MODEL.BACKBONE.NAME ${NET}