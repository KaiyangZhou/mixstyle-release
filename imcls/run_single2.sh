#!/bin/bash

DATA=~/kaiyang/data

S1=$1
S2=$2
S3=$3
T=$4
SEED=$5
DATASET=$6
ND=$7
BATCH=$8

NET=resnet18_mixstyle_diffdom_L234_p0d5_a0d1
TRAINER=Vanilla

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--source-domains ${S1} ${S2} ${S3} \
--target-domains ${T} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/mixstyle/${DATASET}.yaml \
--output-dir output/${DATASET}/${TRAINER}_${NET}_ndomain${ND}_batch${BATCH}/${T}/seed${SEED} \
MODEL.BACKBONE.NAME ${NET} \
DATALOADER.TRAIN_X.SAMPLER RandomDomainSampler \
DATALOADER.TRAIN_X.N_DOMAIN ${ND} \
DATALOADER.TRAIN_X.BATCH_SIZE ${BATCH} \
OPTIM.MAX_EPOCH 150