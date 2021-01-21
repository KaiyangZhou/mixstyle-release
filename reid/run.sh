#!/bin/bash

DATA=$1

###########
# resnet50
###########
python main.py \
--config-file cfgs/cfg_r50.yaml \
-s market1501 \
-t dukemtmcreid \
--root ${DATA} \
model.name resnet50_fc512_ms12_a0d1 \
data.save_dir output/resnet50_fc512_ms12_a0d1/market2duke

python main.py \
--config-file cfgs/cfg_r50.yaml \
-s dukemtmcreid \
-t market1501 \
--root ${DATA} \
model.name resnet50_fc512_ms12_a0d1 \
data.save_dir output/resnet50_fc512_ms12_a0d1/duke2market

###########
# osnet
###########
python main.py \
--config-file cfgs/cfg_osnet.yaml \
-s market1501 \
-t dukemtmcreid \
--root ${DATA} \
model.name osnet_x1_0_ms23_a0d1 \
data.save_dir output/osnet_x1_0_ms23_a0d1/market2duke

python main.py \
--config-file cfgs/cfg_osnet.yaml \
-s dukemtmcreid \
-t market1501 \
--root ${DATA} \
model.name osnet_x1_0_ms23_a0d1 \
data.save_dir output/osnet_x1_0_ms23_a0d1/duke2market