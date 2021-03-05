#!/bin/bash

# MixStyle w/ random shuffle
bash run_batch.sh 1 5 resnet18_mixstyle_L234_p0d5_a0d1 Vanilla

# MixStyle w/ domain label
bash run_batch2.sh 1 5 resnet18_mixstyle2_L234_p0d5_a0d1 Vanilla