#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name="rf-baseline"
#SBATCH --partition="insy" # Request partition.
#SBATCH --qos=short                # This is how you specify QoS
#SBATCH --time=00:05:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                 # Request 1 node
### Always run exclusive to avoid issues with conflicting ray servers (needs fixing in the future)
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # Set one task per node
#SBATCH --cpus-per-task=16         # Request number of CPUs (threads) per task. Be mindful of #CV splits and max_concurrent argument value given to ray in code
#SBATCH --mem=16GB                  # Request ... GB of RAM in total
#SBATCH --gpus-per-task=0
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

# Assuming you have a dedicated directory for *.sif files
export APPTAINER_ROOT="/home/nfs/sramezani/thesis/cluster_stuff"
export APPTAINER_NAME="apptainer-for-thesis.sif"

# Option: save timeline info for debugging (yes="1", no="0"); see https://docs.ray.io/en/latest/ray-core/troubleshooting.html
declare -x RAY_TIMELINE="0"

# This script is meant to be called from a SLURM job file. 
# It launches a ray server on each compute node and starts a python script on the head node.
# It is assumed apptainer is used for environment creation

# The script has four command line options:
# --python_runfile="..." name of the python file to run
# --python_arguments="..." optional arguments to the python script

# --apptainer_root="..." Assuming you have a dedicated directory for *.sif files
# --apptainer_name="..." Name of the apptainer container
## $APPTAINER_ROOT/$APPTAINER_NAME is the full path to the container.sif file

# --rundir="..." name of the working directory (default: current)
# --temp_dir="..." location to store ray temporary files (default: /tmp/ray)

# set defaults
python_runfile="src/main.py"
python_arguments="--config_script 'run_configs/simple_rf_baseline.py' --study_accessions ['MGYS00003677'] --pipeline_version 'v4.1' --label_col 'disease status__biosamples' --positive_class_label 'Sick' --run_name 'test slurm'"
apptainer_root="/home/nfs/sramezani/thesis/cluster_stuff"
apptainer_name="apptainer-for-thesis.sif"
rundir="/home/nfs/sramezani/thesis"
temp_dir="/tmp/ray"

echo "start"

echo "arguments: $python_arguments"

# Abort if no main python file name is given
if [[ ${python_runfile} == "MISSING" ]]
then
  echo "Missing python_runfile option. Aborting."
  exit  
fi

if [[ $RAY_TIMELINE == "1" ]]
then
echo "RAY PROFILING MODE ENABLED"
fi

# generate password for node-to-node communication
redis_password=$(uuidgen)
export redis_password

# get node names and identify IP addresses
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

# set up head node
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

# launch head node, leaving one core unused for the main python script
echo "STARTING HEAD at $node_1"
srun --job-name="ray-head" --nodes=1 --ntasks=1 --cpus-per-task=$((SLURM_CPUS_PER_TASK-1)) -w "$node_1" \
    apptainer exec -B $HOME:$HOME $apptainer_root/$apptainer_name \
    ray start --head --temp-dir="${temp_dir}" --include-dashboard=false --num-cpus=$((SLURM_CPUS_PER_TASK-1)) --node-ip-address=$ip --port=$port --redis-password=$redis_password --block  &
sleep 10

# launch worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --job-name="ray-worker" --nodes=1 --ntasks=1 -w "$node_i" \
    apptainer exec -B $HOME:$HOME $apptainer_root/$apptainer_name \
    ray start --temp-dir="${temp_dir}" --num-cpus=$SLURM_CPUS_PER_TASK --address=$ip_head --redis-password=$redis_password --block &
  sleep 5
done

# export RAY_ADDRESS, so that ray can be initialised using ray.init(), without address
RAY_ADDRESS=$ip_head
export RAY_ADDRESS

# launch main program file on a single core. Wait for it to exit
srun --job-name="main" --unbuffered --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$node_1" \
    apptainer exec -B $HOME:$HOME $apptainer_root/$apptainer_name \
    ray status; cd ${rundir}; python -u ${python_runfile} ${python_arguments}

# if RAY_TIMELINE == 1, save the timeline
if [[ $RAY_TIMELINE == "1" ]]; then
srun --job-name="timeline" --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$node_1" \
      apptainer exec -B $HOME:$HOME $apptainer_root/$apptainer_name \
      ray timeline; cp -n /tmp/ray-timeline-* ${temp_dir}
fi

# stop the ray cluster
srun --job-name="ray-stop" --nodes=1 --ntasks=1 --cpus-per-task=1 -w "$node_1" \
    apptainer exec -B $HOME:$HOME $apptainer_root/$apptainer_name \
    ray stop

# wait for everything so we don't cancel head/worker jobs that have not had time to clean up
wait