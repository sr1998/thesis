import os

from ray.util.multiprocessing import Pool


def worker_function(task_id):
    """A simple worker function to simulate computation."""
    print(f"Task {task_id} is running on CPU {os.getpid()}")
    return f"Result from task {task_id}"


def main():
    num_tasks = 15  # Match `--ntasks-per-node` in SLURM script
    print(f"SLURM_JOB_ID: {os.getenv('SLURM_JOB_ID')}")
    print(f"SLURM_NTASKS: {os.getenv('SLURM_NTASKS')}")
    print(f"SLURM_CPUS_PER_TASK: {os.getenv('SLURM_CPUS_PER_TASK')}")
    print(f"Available CPUs: {cpu_count()}")

    # Use a pool of processes
    with Pool(num_tasks) as pool:
        results = pool.map(worker_function, range(num_tasks))

    # Print results
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
