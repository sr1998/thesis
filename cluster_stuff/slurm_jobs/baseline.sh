#!/bin/sh
#SBATCH --job-name="data_getter"
#SBATCH --partition="insy" # Request partition.
#SBATCH --qos=medium                # This is how you specify QoS
#SBATCH --time=12:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=1         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=1GB                  # Request ... GB of RAM in total
#SBATCH --gpus-per-task=0
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

# If you use DATASETS_ROOT inside your script otherwise remove
# export DATASETS_ROOT="/scratch/$USER/datasets"

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/home/nfs/sramezani/thesis/cluster_stuff"
export APPTAINER_NAME="apptainer-for-thesis.sif"

# for WANDB to work
curl https://curl.se/ca/cacert.pem -o ./cacert.pem
export SSL_CERT_FILE=./cacert.pem

# Setup environment
# module use /opt/insy/modulefiles  # (on DAIC)
# module load cuda/12.1  # If you want to use CUDA, it has to be loaded on the host

## Use this simple command to check that your sbatch 
## settings are working (it should show the GPU that you requested)
# nvidia-smi

# Run script
# Note: There cannot be any characters incuding space behind the `\` symbol.
srun apptainer exec \
    -B $HOME:$HOME \
    --env-file $HOME/.env \
    $APPTAINER_ROOT/$APPTAINER_NAME \
    python -m src.main \
    --config_script "run_configs.simple_lightgbm_baseline_for_optuna" \
    --study_accessions ["MGYS00005285"] \
    --summary_type "GO_abundances" \
    --pipeline_version "v4.1" \
    --label_col "phenotype" \
    --positive_class_label "Diabetes Mellitus, Type 2"
#   -B /projects/:/projects/ \
#   -B /scratch/$USER:/scratch/$USER \

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /home/$USER:/home/$USER \ mounts host file-sytem inside container 
# -B can be used several times, change this to match your cluster file-system
# $APPTAINER_ROOT/$APPTAINER_NAME is the full path to the container.sif file
# python script.py is the command that you want to use inside the container