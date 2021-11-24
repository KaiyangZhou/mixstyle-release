#!/bin/bash

cd ..

DATA=~/kaiyang/data
SSDG=~/kaiyang/code/ssdg-benchmark

DATASET=$1
TRAINER=Vanilla2 # use labeled source data only
NET=$2 # e.g. resnet18_ms_l123, resnet18_ms_l12

if [ ${DATASET} == ssdg_pacs ]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == ssdg_officehome ]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
fi

for SEED in $(seq 1 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/mixstyle/${DATASET}.yaml \
        --output-dir output/${DATASET}/${TRAINER}/${NET}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}
    done
done
