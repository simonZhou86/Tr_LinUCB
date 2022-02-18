#!/bin/bash
# Submit Sequential job
### Builds the job index Create a sequential range
array_values=`seq 1 40 `


### Generate a sh file for each Job ID
# Modify the contents of the pbs file to be relevant to your simulation

WALLTIME=RUNNING TIME
SIM_NAME=SOME NAME #simulation name
#MATLAB_VERSION=2019a
SCRIPT_PATH=PATH OF THE SCRIPT
SCRIPT_FILE=SCRIPT FILE NAME
SH_FILE_NAME=.sh file name
OUT_FOLDER=OUTPUT DIRECTORY
LOG_OUT=OutfileLog #output folder
ERR_OUT=OutfileErr

for i in $array_values
do
		
	cat > ${SH_FILE_NAME}${i}.sh << EOF
#!/bin/bash
#
#SBATCH --account=YOUR ACCOUNT
#SBATCH --time=$WALLTIME
##
## Request nodes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=$SIM_NAME$i
## Declare an output log for all jobs to use:
#SBATCH --output=$OUT_FOLDER/$LOG_OUT/$SIM_NAME$i.log
#SBATCH --error=$OUT_FOLDER/$ERR_OUT/$SIM_NAME$i.err
#SBATCH --verbose

cd CODE PATH

module load python
module load scipy-stack

python $SCRIPT_FILE $i

exit 0;
				
EOF
done




mkdir $OUT_FOLDER/$LOG_OUT
mkdir $OUT_FOLDER/$ERR_OUT

# Launch the job and then remove the temporarily created qsub file.
for i in $array_values
do 
	# This submits the single job to the resource manager

echo $i
sbatch ${SH_FILE_NAME}${i}.sh
#rm -rf ${SH_FILE_NAME}${i}.sh

sleep 1
	# This removes the job file as torque reads the script at
	# submission time
done
