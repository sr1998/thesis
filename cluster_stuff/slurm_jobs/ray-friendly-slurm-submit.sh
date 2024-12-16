#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name="rf-baseline"
#SBATCH --partition="insy" # Request partition.
#SBATCH --qos=short                # This is how you specify QoS
#SBATCH --time=00:00:20            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
### Always run exclusive to avoid issues with conflicting ray servers (needs fixing in the future)
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=16         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=16GB                  # Request 4 GB of RAM in total
#SBATCH --gpus-per-task=0
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

# If you use DATASETS_ROOT inside your script otherwise remove
# export DATASETS_ROOT="/scratch/$USER/datasets"

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/home/nfs/sramezani/thesis/cluster_stuff"
export APPTAINER_NAME="apptainer-for-thesis.sif"

# Setup environment
# module use /opt/insy/modulefiles  # (on DAIC)
# module load cuda/12.1  # If you want to use CUDA, it has to be loaded on the host

## Use this simple command to check that your sbatch 
## settings are working (it should show the GPU that you requested)
# nvidia-smi

# Option: save timeline info for debugging (yes="1", no="0"); see https://docs.ray.io/en/latest/ray-core/troubleshooting.html
declare -x RAY_TIMELINE="0"

# OLD: Run script
# Note: There cannot be any characters incuding space behind the `\` symbol.
# srun apptainer exec -B $HOME:$HOME $APPTAINER_ROOT/$APPTAINER_NAME python cluster_stuff/test_setup.py
#   --nv \
#   --env-file ~/.env \
#   -B /projects/:/projects/ \
#   -B /scratch/$USER:/scratch/$USER \

echo "start"

# Adapt this to your job requirements
. cluster_stuff/slurm_jobs/ray-sklearn-inside-apptainer-on-slurm.sh \
    --python_runfile="src/main.py" \
    --python_arguments="--config_script \"run_configs/simple_rf_baseline.py\" --study_accessions '[\"MGYS00003677\"]' --pipeline_version \"v4.1\" --label_col \"disease status__biosamples\" --positive_class_label \"Sick\" --run_name \"test slurm\""

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /home/$USER:/home/$USER \ mounts host file-sytem inside container 
# -B can be used several times, change this to match your cluster file-system
# $APPTAINER_ROOT/$APPTAINER_NAME is the full path to the container.sif file
# python script.py is the command that you want to use inside the container
