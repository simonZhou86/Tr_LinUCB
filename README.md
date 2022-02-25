# Introduction 

This repository includes python scripts for running the Tr-LinUCB algorithm. The code has been used to generate the result in the paper:

[**Truncated LinUCB for Stochastic Linear Bandits**](https://arxiv.org/abs/2202.11735), Yanglei Song and Meng Zhou

# Files

The algorithm for Tr-LinUCB and LinUCB (resp. to $S = T$) is in the file tr_linucb.py. For Greedy First and OLS bandit algorithm in the paper, please refer to https://github.com/khashayarkhv/contextual-bandits.

# Usage

## Using your local terminal/environment:

Python version: >= Python 3.6

First, you have to install pandas, numpy as matplotlib:

```python
pip install pandas
pip install numpy
pip install matplotlib
```

You have to first comment out line 231 and modify the variable "ncpus" based on your environment in line 234. After that, you can simple run the following script in the command line:

```python
python3 tr_linucb.py --T X --kappa X --k X --d X --lmd X --m2 X --sigma_e X --arm_para_noise X
```
where the default setting is `T=10000, kappa=2.0, k=2, d=4, lmd=0.1, m2=1, sigma_e=0.5 and arm_para_noise=0`. Note that `arm_para_noise` only takes the binary value either 0 or 1. Use the following code to show the arguments' description:

```python
python3 tr_linucb.py -h
```

You may define your own value for the parameters and replace `X`.


# Citation

If you are using our code, please cite the following paper:

[**Truncated LinUCB for Stochastic Linear Bandits**](https://arxiv.org/abs/2202.11735), Yanglei Song and Meng Zhou

```bibtex
@misc{song2022truncated,
      title={Truncated LinUCB for Stochastic Linear Bandits}, 
      author={Yanglei Song and Meng zhou},
      year={2022},
      eprint={2202.11735},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
