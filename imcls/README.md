# MixStyle on image classification across domains

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Running scripts are provided in `scripts/`.

Please see `run_mixstyle.sh`.

Do the following steps before running the code:

- modify the `DATA` variable in `run_single.sh` and `run_single2.sh`
- activate the `dassl` environment via `conda activate dassl`

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