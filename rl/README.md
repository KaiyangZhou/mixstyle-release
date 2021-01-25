# MixStyle on reinforcement learning

## How to install

Please follow the installation instructions taken from the original repo (https://github.com/openai/coinrun) to install the requirements:

```bash
# Linux
apt-get install mpich build-essential qt5-default pkg-config
# Mac
brew install qt open-mpi pkg-config

conda create --name mixstyle-coinrun python=3.7
conda activate mixstyle-coinrun
pip install tensorflow==1.12.0  # or tensorflow-gpu
pip install -r requirements.txt
pip install -e .
```

By default, the models and the log files will be saved to `self.WORKDIR` and `self.TB_DIR`, respectively (defined in `coinrun/config.py`).

## How to run

Baseline (L2 weight decay + data augmentation)
```bash
RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id l2wd_da_run1 --num-levels 500 --test --long --l2 0.0001 -uda 1

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id l2wd_da_run2 --num-levels 500 --test --long --l2 0.0001 -uda 1

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id l2wd_da_run3 --num-levels 500 --test --long --l2 0.0001 -uda 1
```

Baseline + MixStyle
```bash
RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id l2wd_da_ms_run1 --num-levels 500 --test --long --l2 0.0001 -uda 1 --mixstyle

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id l2wd_da_ms_run2 --num-levels 500 --test --long --l2 0.0001 -uda 1 --mixstyle

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id l2wd_da_ms_run3 --num-levels 500 --test --long --l2 0.0001 -uda 1 --mixstyle
```

IBAC-SNI
```bash
RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id ibac_sni_lmda0.5_run1 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id ibac_sni_lmda0.5_run2 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id ibac_sni_lmda0.5_run3 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni
```

IBAC-SNI + MixStyle
```bash
RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id ibac_sni_lmda0.5_ms_run1 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni --mixstyle

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id ibac_sni_lmda0.5_ms_run2 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni --mixstyle

RCALL_NUM_GPU=4 mpiexec -n 4 python -m coinrun.train_agent --run-id ibac_sni_lmda0.5_ms_run3 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni --mixstyle
```

Note that

- you need 4 gpus
- each experiment is repeated for three times (i.e. run1, run2 & run3)
- by default, devices=0,1,2,3 are used
- if you wanna use devices=4,5,6,7, please append `-gpu-offset 4` to the commands

You can download the trained models and the corresponding log files used in our paper from this [google drive link](https://drive.google.com/drive/folders/1NeoGgLMtU_a3sflKqYI00bT81473BXTP?usp=sharing).

If you are looking for the tensorflow implementation of MixStyle, please refer to [this code](https://github.com/KaiyangZhou/mixstyle-release/blob/master/rl/coinrun/policies.py#L11).

## How to plot

The plotting function is implemented in `plots.py`.

A note on the tensorboard plots: For each run, you will see 4 different folders 'name_0', 'name_1', etc. The 'name_0' is the performance on the training set. The 'name_1' version is the performance on the test set. Furthermore, to compare to the paper you'll need to multiply the number of frames by 3, as tensorboard reports the frames _per worker_, whereas the paper reports the total number of frames used for training.

To use `plots.py`, fill in the `path` variable, as well as `plotname`, `plotname_kl` and the `experiments` dictionary where each entry corresponds to one line which will be the average over all run-ids listed in the corresponding list (see `plots.py` for examples).

## How to cite

If you find this code useful to your research, please cite the following papers.

```
@inproceedings{cobbe2019quantifying,
  title={Quantifying generalization in reinforcement learning},
  author={Cobbe, Karl and Klimov, Oleg and Hesse, Chris and Kim, Taehoon and Schulman, John},
  booktitle={ICML},
  year={2019}
}

@inproceedings{igl2019generalization,
  title={Generalization in reinforcement learning with selective noise injection and information bottleneck},
  author={Igl, Maximilian and Ciosek, Kamil and Li, Yingzhen and Tschiatschek, Sebastian and Zhang, Cheng and Devlin, Sam and Hofmann, Katja},
  booktitle={NeurIPS},
  year={2019}
}

@inproceedings{zhou2021mixstyle,
  title={Domain Generalization with MixStyle},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  booktitle={ICLR},
  year={2021}
}
```