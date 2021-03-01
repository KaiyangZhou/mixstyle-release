#!/bin/bash

START=$1
END=$2

DATASET=pacs
D1=art_painting
D2=cartoon
D3=photo
D4=sketch

ND=2
BATCH=128

for SEED in $(seq ${START} ${END})
do
    bash run_single2.sh ${D2} ${D3} ${D4} ${D1} ${SEED} ${DATASET} ${ND} ${BATCH}
    bash run_single2.sh ${D1} ${D3} ${D4} ${D2} ${SEED} ${DATASET} ${ND} ${BATCH}
    bash run_single2.sh ${D1} ${D2} ${D4} ${D3} ${SEED} ${DATASET} ${ND} ${BATCH}
    bash run_single2.sh ${D1} ${D2} ${D3} ${D4} ${SEED} ${DATASET} ${ND} ${BATCH}
done