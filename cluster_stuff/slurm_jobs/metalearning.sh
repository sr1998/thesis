#!/bin/sh
#SBATCH --job-name="metalearning_base"
#SBATCH --partition=general,insy # Request partition.
#SBATCH --qos=short                # This is how you specify QoS
#SBATCH --time=01:30:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=1         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=4GB                  # Request ... GB of RAM in total
#SBATCH --gpus-per-task=1

# If you use DATASETS_ROOT inside your script otherwise remove
# export DATASETS_ROOT="/scratch/$USER/datasets"

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/tudelft.net/staff-umbrella/abeellabstudents/sramezani"
export APPTAINER_NAME="apptainer-for-thesis.sif"

ALGORITHM="MAML"
mkdir "slurm_logs/${SLURM_JOB_NAME}"
mkdir "slurm_logs/${SLURM_JOB_NAME}/${ALGORITHM}"

LOG_FILE="slurm_logs/${SLURM_JOB_NAME}/${ALGORITHM}/${SLURM_JOB_ID}.out"
ERR_FILE="slurm_logs/${SLURM_JOB_NAME}/${ALGORITHM}/${SLURM_JOB_ID}.err"

# Redirect stdout and stderr to these files
exec > "$LOG_FILE" 2> "$ERR_FILE"

# for WANDB to work
curl https://curl.se/ca/cacert.pem -o ./cacert.pem
export SSL_CERT_FILE=./cacert.pem

# Setup environment
module use /opt/insy/modulefiles  # (on DAIC)
module load cuda/12.1  # If you want to use CUDA, it has to be loaded on the host

ls -l /tudelft.net/staff-umbrella/abeellabstudents/sramezani/apptainer-for-thesis.sif

## Use this simple command to check that your sbatch 
## settings are working (it should show the GPU that you requested)
nvidia-smi

# Run script
# Note: There cannot be any characters incuding space behind the `\` symbol.
srun apptainer exec \
    -B $HOME:$HOME \
    --env-file $HOME/.env \
    --nv \
    $APPTAINER_ROOT/$APPTAINER_NAME \
    python -m src.main_meta_learning \
    --model_script="src.models.models" \
    --model_name="model2" \
    --abundance_file="mpa4_species_profile_preprocessed.csv" \
    --metadata_file="sample_group_species_preprocessed.csv" \
    --test_study="" \
    --val_study="JieZ_2017" \
    --outer_lr_range="(1, 1)" \
    --inner_lr_range="(0.5, 0.5)" \
    --inner_rl_reduction_factor=2 \
    --n_epochs=150 \
    --train_k_shot=10 \
    --n_gradient_steps=5 \
    --n_parallel_tasks=5 \
    --n_components_reduction_factor=0 \
    --use_cached_pca=False \
    --do_normalization_before_scaling=True \
    --scale_factor_before_training=100 \
    --loss_fn="BCELog" \
    --algorithm="${ALGORITHM}" \
    # --what "sun et al" \
    # --config_script "run_configs.rf_baseline_for_sun_et_al" \
    # --study "JieZ_2017"\
    # --positive_class_label "Disease" \
    # --summary_type "taxonomy_abundances_SSU" \
    # --pipeline_version "v4.1" \
    # --label_col "disease status__biosamples" \
#   -B /projects/:/projects/ \
#   -B /scratch/$USER:/scratch/$USER \


# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /home/$USER:/home/$USER \ mounts host file-sytem inside container 
# -B can be used several times, change this to match your cluster file-system
# $APPTAINER_ROOT/$APPTAINER_NAME is the full path to the container.sif file
# python script.py is the command that you want to use inside the container