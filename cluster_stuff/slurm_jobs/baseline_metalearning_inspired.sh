#!/bin/sh
#SBATCH --job-name="sun_et_al_metalearning_inspired_baseline"
#SBATCH --partition=general,insy # Request partition.
#SBATCH --qos=short                # This is how you specify QoS
#SBATCH --time=4:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=1         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=1GB                  # Request ... GB of RAM in total
#SBATCH --gpus-per-task=0


STUDIES=(
    'ChenB_2020' 'ChuY_2021' 'HeQ_2017' 'HuY_2019'
    'HuangR_2020' 'LiJ_2017' 'LiR_2021'
    'LiuP_2021' 'LiuR_2017' 'LuW_2018' 'MaoL_2021'
    'QiX_2019' 'QianY_2020' 'QinN_2014' 'WanY_2021'
    'WangM_2019' 'WangX_2020' 'WengY_2019' 'YanQ_2017'
    'YangY_2021' 'YeZ_2018' 'YeZ_2020' 'YeohYK_2021' 'YuJ_2017'
    'ZhangX_2015' 'ZhongH_2019' 'ZhouC_2020' 'ZhuF_2020'
    'ZhuJ_2018' 'ZhuQ_2021' 'ZuoK_2019'
)
# 31

# STUDIES=('JieZ_2017' 'WangQ_2021' 'ZengQ_2021' 'HanL_2021')
# STUDIES=('QinJ_2012') 3:30
TEST_STUDY="${STUDIES[$SLURM_ARRAY_TASK_ID]}"

mkdir "slurm_logs/${SLURM_JOB_NAME}"
mkdir "slurm_logs/${SLURM_JOB_NAME}/${STUDY}"

LOG_FILE="slurm_logs/${SLURM_JOB_NAME}/${TEST_STUDY}/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${TEST_STUDY}.out"
ERR_FILE="slurm_logs/${SLURM_JOB_NAME}/${TEST_STUDY}/${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${TEST_STUDY}.err"

# Redirect stdout and stderr to these files
exec > "$LOG_FILE" 2> "$ERR_FILE"

# If you use DATASETS_ROOT inside your script otherwise remove
# export DATASETS_ROOT="/scratch/$USER/datasets"

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/tudelft.net/staff-umbrella/abeellabstudents/sramezani"
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
    python -m src.main_baseline \
    --what "sun et al" \
    --config_script "run_configs.rf_metalearning_baseline_for_sun_et_al" \
    --test_study "$TEST_STUDY" \
    --abundance_file "mpa4_species_profile_preprocessed.csv"
    --metadata_file "sample_group_species_preprocessed.csv"
    --eval_k_shot 10 \
    --positive_class_label "Disease" \

# srun apptainer exec \
#     -B $HOME:$HOME \
#     --env-file $HOME/.env \
#     $APPTAINER_ROOT/$APPTAINER_NAME \
#     python -m src.main \
#     --config_script "run_configs.simple_lightgbm_baseline_for_optuna" \
#     --study_accessions ["MGYS00005285"] \
#     --summary_type "GO_abundances" \
#     --pipeline_version "v4.1" \
#     --label_col "phenotype" \
#     --positive_class_label "Diabetes Mellitus, Type 2"


#   -B /projects/:/projects/ \
#   -B /scratch/$USER:/scratch/$USER \

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /home/$USER:/home/$USER \ mounts host file-sytem inside container 
# -B can be used several times, change this to match your cluster file-system
# $APPTAINER_ROOT/$APPTAINER_NAME is the full path to the container.sif file
# python script.py is the command that you want to use inside the container