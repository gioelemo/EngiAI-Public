# HPC Integration

Engineer Assistant integrates with high-performance computing (HPC) clusters for running computationally intensive tasks.

## Prerequisites

- Access to an HPC cluster with SLURM
- SSH key authentication configured
- Proper environment modules on the cluster

## Configuration

Set up your HPC credentials in the configuration:

```bash
export HPC_HOST_ALIAS="euler"
export HPC_HOSTNAME="cluster.university.edu"
export HPC_USERNAME="your_username"
```

Or in your `.env` file:

```bash
HPC_HOST_ALIAS=euler
HPC_HOSTNAME=cluster.university.edu
HPC_USERNAME=your_username
```

## Submitting Jobs

### Basic Job Submission

```python
from src.agents import HPCAgent

agent = HPCAgent()

job_id = agent.submit_job(
    script="simulation.sh",
    nodes=4,
    cpus_per_node=48,
    time="24:00:00",
    memory="128GB"
)

print(f"Job submitted with ID: {job_id}")
```

### Job Script Template

Create a SLURM job script (`simulation.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=optimization
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --mem=128GB

module load python/3.11
module load openmpi

python simulation.py
```

## Monitoring Jobs

### Check Job Status

```python
status = agent.check_status(job_id)
print(f"Job status: {status}")
```

### View Job Queue

```python
queue = agent.get_queue()
for job in queue:
    print(f"{job['id']}: {job['status']} - {job['name']}")
```

### Retrieve Output

```python
# Get standard output
output = agent.get_output(job_id)

# Get standard error
errors = agent.get_errors(job_id)
```

## Advanced Usage

### Interactive Job Monitoring

```python
import time

while True:
    status = agent.check_status(job_id)
    if status in ["COMPLETED", "FAILED", "CANCELLED"]:
        break
    print(f"Job {job_id}: {status}")
    time.sleep(60)  # Check every minute

print(f"Final status: {status}")
```

### Batch Job Submission

```python
job_ids = []
configs = [
    {"volfrac": 0.3, "forcedist": 0.0},
    {"volfrac": 0.4, "forcedist": 0.5},
    {"volfrac": 0.5, "forcedist": 1.0},
]

for config in configs:
    job_id = agent.submit_optimization_job(config)
    job_ids.append(job_id)

print(f"Submitted {len(job_ids)} jobs")
```

## File Transfer

### Upload Files

```python
agent.upload_file(
    local_path="local_data.npy",
    remote_path="/scratch/username/data.npy"
)
```

### Download Results

```python
agent.download_file(
    remote_path="/scratch/username/results.npy",
    local_path="results.npy"
)
```

## Job Templates

The HPC agent provides pre-configured templates for common tasks:

```python
# Optimization job
job_id = agent.submit_optimization_job(
    problem="beams2d",
    config={"volfrac": 0.3}
)

# Simulation job
job_id = agent.submit_simulation_job(
    design="design.npy",
    simulator="fenics"
)
```

## Troubleshooting

### Connection Issues

If you encounter connection problems:

1. Verify SSH access: `ssh your_username@cluster.university.edu`
2. Check SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
3. Test SLURM commands: `ssh cluster.university.edu "squeue -u $USER"`

### Job Failures

Common issues and solutions:

- **Out of Memory**: Increase `--mem` in job script
- **Time Limit**: Increase `--time` or optimize your code
- **Module Not Found**: Check available modules with `module avail`

## Best Practices

1. **Use appropriate resources**: Don't request more than needed
2. **Set realistic time limits**: Add buffer time for queue delays
3. **Monitor disk space**: Clean up old results regularly
4. **Use checkpoints**: Save intermediate results for long jobs
5. **Test locally first**: Debug on small problems before scaling up

## Next Steps

- [Learn about agents](agents.md)
- [Tool reference](tools.md)
- [Configuration guide](../configuration.md)
