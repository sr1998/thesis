#!/bin/sh
#SBATCH --job-name="STNT_data_prep"
#SBATCH --partition=general,insy # Request partition.
#SBATCH --qos=medium                # This is how you specify QoS
#SBATCH --time=10:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=1         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=2GB                  # Request ... GB of RAM in total
#SBATCH --gres=gpu:a40:1        # Request 1 GPU (A40) per node



mkdir "slurm_logs/${SLURM_JOB_NAME}"

LOG_FILE="slurm_logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.out"
ERR_FILE="slurm_logs/${SLURM_JOB_NAME}/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.err"

# Redirect stdout and stderr to these files
exec > "$LOG_FILE" 2> "$ERR_FILE"

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/tudelft.net/staff-umbrella/abeellabstudents/sramezani"
export APPTAINER_NAME="apptainer-for-thesis.sif"

# for WANDB to work
curl https://curl.se/ca/cacert.pem -o ./cacert.pem
export SSL_CERT_FILE=./cacert.pem

# Setup environment
module use /opt/insy/modulefiles  # (on DAIC)
module load cuda/12.1  # If you want to use CUDA, it has to be loaded on the host

ls -l /tudelft.net/staff-umbrella/abeellabstudents/sramezani/apptainer-for-thesis.sif

## Use this simple command to check that your sbatch 
## settings are working (it should show the GPU that you requested)
# nvidia-smi

# Run script
# Note: There cannot be any characters incuding space behind the `\` symbol.

# MAML
srun apptainer exec \
    -B $HOME:$HOME \
    -B /tudelft.net/staff-umbrella/abeellabstudents/sramezani:/tudelft.net/staff-umbrella/abeellabstudents/sramezani \
    --env-file /tudelft.net/staff-umbrella/abeellabstudents/sramezani/.env \
    --nv \
    $APPTAINER_ROOT/$APPTAINER_NAME \
    python -m src.data.stunt

#   -B /projects/:/projects/ \
#   -B /scratch/$USER:/scratch/$USER \


# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /home/$USER:/home/$USER \ mounts host file-sytem inside container 
# -B can be used several times, change this to match your cluster file-system
# $APPTAINER_ROOT/$APPTAINER_NAME is the full path to the container.sif file
# python script.py is the command that you want to use inside the container