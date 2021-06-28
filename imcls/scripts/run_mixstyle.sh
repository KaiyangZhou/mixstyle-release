#!/bin/bash

# PACS | MixStyle w/ random mixing
bash dg.sh pacs resnet18_ms_l123 random

# PACS | MixStyle w/ cross-domain mixing
bash dg.sh pacs resnet18_ms_l123 crossdomain

# OfficeHome | MixStyle w/ random mixing
bash dg.sh office_home_dg resnet18_ms_l12 random

# OfficeHome | MixStyle w/ cross-domain mixing
bash dg.sh office_home_dg resnet18_ms_l12 crossdomain