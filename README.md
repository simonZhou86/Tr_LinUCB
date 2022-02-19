# Introduction 

This repository includes python scripts for running the Tr-LinUCB algorithm. The code has been used to generate the result in the paper:
Truncated LinUCB for Stochastic Linear Bandits, Yanglei Song and Meng Zhou

# Files

The algorithm for Tr-LinUCB is in the file tr_linucb.py. For Greedy First and OLS bandit algorithm in the paper, please refer to https://github.com/khashayarkhv/contextual-bandits.

# Usage

## if you are using local terminal/environment:

Python version: >= Python 3.6

First, you have to install pandas, numpy as matplotlib:

```python
pip install pandas
pip install numpy
pip install matplotlib
```

You have to first comment out line 231 and modify the variable "ncpus" based on your environment in line 234. After that, you can simple run the following script in the command line:

```python
python3 tr_linucb.py --T X --kappa X --k X --d X --lmd X --m2 X --sigma_e X --case X
```
where the default setting is $`T=10000`$, kappa=2.0, k=2, d=4, lmd=0.1, m2=1, sigma_e=0.5 and case=1

Users can define their own value for the parameters and replace X.

## if you are using Compute Canada:

Copy the below lines to a .sh file and submit through compute canada by sbatch your_file.sh

or you could refer to the cc_scripts folder for other useful scripts

```shell
#!/bin/bash
#
#SBATCH --account=YOUR ACCOUNT NAME
#SBATCH --time=RUNNING TIME
##
## Request nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=YOUR JOB NAME
## Declare an output log for all jobs to use:
#SBATCH --output=LOG FILE OUTPUT DIRECTORY
#SBATCH --error=ERROR FILE OUTPUT DIRECTORY
#SBATCH --verbose

cd YOUR PATH FOR THE CODE

module load python
module load scipy-stack

python tr_linucb.py --T X --kappa X --k X --d X --lmd X --m2 X --sigma_e X --case X

exit 0;

```

# Citation

If you are using our code, please cite our paper:

**"Truncated LinUCB for Stochastic Linear Bandits"**, Yanglei Song and Meng Zhou
