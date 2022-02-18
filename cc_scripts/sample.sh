#!/bin/bash
#
#SBATCH --account=YOUR ACCOUNT
#SBATCH --time=RUNNING TIME
##
## Request nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=JOB NAME
## Declare an output log for all jobs to use:
#SBATCH --output=LOG FILE OUTPUT DIRECTORY
#SBATCH --error=ERROR FILE OUTPUT DIRECTORY
#SBATCH --verbose

cd CODE PATH

module load python
module load scipy-stack

python tr_linucb.py --T 10000 --case 1

exit 0;
