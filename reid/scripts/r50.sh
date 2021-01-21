#!/bin/bash

cd ..

DATA=$1
S=$2
T=$3
M=$4

python main.py \
--config-file cfgs/cfg_r50.yaml \
-s ${S} \
-t ${T} \
--root ${DATA} \
model.name ${M} \
data.save_dir output/${M}/${S}_to_${T}
