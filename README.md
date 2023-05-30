# Introduction 

This repository includes python scripts for running the Tr-LinUCB algorithm. The code has been used to generate the result in the paper:

[**Truncated LinUCB for Stochastic Linear Bandits**](https://arxiv.org/abs/2202.11735), Yanglei Song and Meng Zhou

# Files

The algorithm for Tr-LinUCB and LinUCB (resp. to `S = T`) is in the file tr_linucb.py. For Linear Thompson sampling, refer to the file linTS_torch.py. For Greedy First and OLS bandit algorithm in the paper, please refer to the code of the paper: [Mostly Exploration-Free Algorithms for Contextual Bandits, Hamsa Bastani, Mohsen Bayati, and Khashayar Khosravi, Forthcoming in Management Science](https://github.com/khashayarkhv/contextual-bandits)

# Usage

## Using your local terminal/environment:

### Synthetic Data
Python version: >= Python 3.6

First, you have to install pandas, numpy as matplotlib:

```python
pip install pandas
pip install numpy
pip install matplotlib
```
For torch, please visit [the official website](https://pytorch.org/get-started/locally/) to download the latest version.

cd to your environment/folder first.

You have to first comment out line 231 and modify the variable "ncpus" based on your environment in line 234. After that, you can simple run the following script in the command line:

```shell
python3 tr_linucb.py --T X --kappa X --k X --d X --lmd X --m2 X --sigma_e X --arm_para_noise X
```
where the default setting is `T=10000, kappa=2.0, k=2, d=4, lmd=0.1, m2=1, sigma_e=0.5 and arm_para_noise=0`. Note that `arm_para_noise` only takes the binary value either 0 or 1. Use the following code to show the parameters' description:

```shell
python3 tr_linucb.py -h
```

You may define your own value for the parameters and replace `X`.

Similarly, to run linear thompson Sampling:

```shell
python3 linTS_torch.py --T X --k X --d X --R X --delta X --sigma_e X --xmax X --arm_para_noise X
```
where the default setting is `T=10000, k=2, d=4, R=0.1, delta=0.99, sigma_e=0.5 and arm_para_noise=0`. The parameters' description can be found using ```python3 linTS_torch.py -h```

### Real Data

If you want to modify the value of parameters, edit dataset.config.py in the configs folder first, then run the following:

```shell
python3 ./run_realsim.py
```

# Citation

If you are using our code, please cite the following paper:

[**Truncated LinUCB for Stochastic Linear Bandits**](https://arxiv.org/abs/2202.11735), Yanglei Song and Meng Zhou

```bibtex
@article{song2022truncated,
  title={Truncated LinUCB for Stochastic Linear Bandits},
  author={Song, Yanglei and Meng Zhou},
  journal={arXiv preprint arXiv:2202.11735},
  year={2022}
}
```
