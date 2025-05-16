#!/bin/sh
#SBATCH --job-name="metalearning"
#SBATCH --partition=general,insy # Request partition.
#SBATCH --qos=medium                # This is how you specify QoS
#SBATCH --time=10:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=1         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=2GB                  # Request ... GB of RAM in total
#SBATCH --gres=gpu:a40:1        # Request 1 GPU (A40) per node

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/tudelft.net/staff-umbrella/abeellabstudents/sramezani"
export APPTAINER_NAME="apptainer-for-thesis.sif"

# for WANDB to work
curl https://curl.se/ca/cacert.pem -o ./cacert.pem
export SSL_CERT_FILE=./cacert.pem

# Setup environment
module use /opt/insy/modulefiles  # (on DAIC)
module load cuda/12.1  # If you want to use CUDA, it has to be loaded on the host

## Use this simple command to check that your sbatch 
## settings are working (it should show the GPU that you requested)w
nvidia-smi

srun apptainer exec \
    -B $HOME:$HOME \
    -B /tudelft.net/staff-umbrella/abeellabstudents/sramezani:/tudelft.net/staff-umbrella/abeellabstudents/sramezani \
    --env-file $HOME/.env \
    --nv \
    $APPTAINER_ROOT/$APPTAINER_NAME \
    jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
