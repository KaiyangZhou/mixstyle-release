# MixStyle on image classification across domains

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `*.sh` based on the paths on your computer
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`


### Domain Generalization
```bash
# PACS | MixStyle w/ random mixing
bash dg.sh pacs resnet18_ms_l123 random

# PACS | MixStyle w/ cross-domain mixing
bash dg.sh pacs resnet18_ms_l123 crossdomain

# OfficeHome | MixStyle w/ random mixing
bash dg.sh office_home_dg resnet18_ms_l12 random

# OfficeHome | MixStyle w/ cross-domain mixing
bash dg.sh office_home_dg resnet18_ms_l12 crossdomain
```

To extract features (or feature statistics) for analysis, you can add `--vis` to the input arguments (also specify `--model-dir` and `--load-epoch`), which will run `trainer.vis()` (implemented in `trainers/vanilla2.py`). However, you need to make changes in several places to make this work, e.g., you need to modify the model's code such that the model directly outputs features. `trainer.vis()` will save the extracted features to `embed.pt`. To visualize the features, you can use `vis.py` (please see the code for more details).

### Semi-Supervised Domain Generalization
```bash
# PACS | MixStyle w/ labeled source data only
bash ssdg1.sh ssdg_pacs resnet18_ms_l123

# PACS | MixStyle w/ labeled + unlabeled source data
bash ssdg2.sh ssdg_pacs resnet18_ms_l123

# OfficeHome | MixStyle w/ labeled source data only
bash ssdg1.sh ssdg_officehome resnet18_ms_l12

# OfficeHome | MixStyle w/ labeled + unlabeled source data
bash ssdg2.sh ssdg_officehome resnet18_ms_l123
```

### Unsupervised Domain Adaptation
```bash
# Single-source UDA on VisDA-17
bash da.sh

# Multi-source UDA on PACS
bash ssdg2.sh msda_pacs resnet18_ms_l123
```


## How to cite

If you find this code useful to your research, please cite the following papers.

```
@article{zhou2020domain,
  title={Domain Adaptive Ensemble Learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv preprint arXiv:2003.07325},
  year={2020}
}

@inproceedings{zhou2021mixstyle,
  title={Domain Generalization with MixStyle},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  booktitle={ICLR},
  year={2021}
}

@article{zhou2021mixstylenn,
  title={MixStyle Neural Networks for Domain Generalization and Adaptation},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv:2107.02053},
  year={2021}
}
```