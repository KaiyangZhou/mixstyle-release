# MixStyle on image classification across domains

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `dg.sh` based on the paths on your computer
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`


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
```