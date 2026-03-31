# HPC Integration

EngiAI integrates with high-performance computing (HPC) clusters for running computationally intensive tasks via SLURM.

## Prerequisites

- Access to an HPC cluster with SLURM (e.g., ETH Euler)
- SSH key authentication configured
- Proper environment modules on the cluster

## Configuration

Set up your HPC credentials in `.env`:

```bash
HPC_HOST_ALIAS=euler
HPC_HOSTNAME=euler.ethz.ch
HPC_USERNAME=your_username
```

And configure your SSH access:

```bash
# In ~/.ssh/config
Host euler
    HostName euler.ethz.ch
    User your-username
    IdentityFile ~/.ssh/id_ed25519
```

Test the connection:
```bash
ssh euler
```

## Using the HPC Agent

The HPC agent is accessed through natural language via the Streamlit UI or Supervisor Agent. Simply describe what you want to do:

### Submitting Jobs

```
"Submit a training job to Euler with seed 1 and 100 epochs"
"Run the cGAN training on HPC for beams2d"
```

The agent will:
1. Generate a SLURM script with appropriate resources
2. Submit the job via SSH
3. Report the job ID

### Monitoring Jobs

```
"Check the status of my running jobs"
"Monitor job 12345 until it completes"
```

### Retrieving Results

```
"Download the output from job 12345"
"Get the training results from HPC"
```

## Available HPC Tools

The HPC agent has access to these tools:

| Tool | Description |
|------|-------------|
| `test_hpc_connection` | Verify SSH connectivity to the cluster |
| `submit_slurm_job` | Submit a SLURM job script |
| `get_slurm_job_status` | Check job status |
| `monitor_job_until_complete` | Poll until job finishes |
| `cancel_slurm_job` | Cancel a running job |
| `download_job_outputs` | Retrieve output files via SFTP |
| `get_active_jobs_summary` | List all active jobs |

## SLURM Configuration

Default SLURM parameters can be set in `.env`:

```bash
SLURM_TIME=00:45:00
SLURM_NTASKS=1
SLURM_CPUS_PER_TASK=4
SLURM_MEM_PER_CPU=7GB
SLURM_GPUS=rtx_4090:1
```

These can also be configured via the Settings UI.

## ML Training Pipeline

The HPC agent supports end-to-end ML training workflows:

1. **Generate training script** — Creates a SLURM script for EngiOpt model training
2. **Submit to cluster** — Uploads and submits the job
3. **Monitor execution** — Tracks job progress until completion
4. **Evaluate results** — Downloads the trained model and computes metrics

Example:
```
"Train a cGAN model for beams2d on Euler with seed 1 and 100 epochs,
then evaluate it against the dataset baseline"
```

## SSH in Docker

When running in Docker, SSH keys are automatically mounted from `~/.ssh`. See the [Docker Deployment Guide](../docker_deployment.md) for details on SSH agent forwarding with passphrase-protected keys.

## Troubleshooting

### Connection Issues

1. Verify SSH access: `ssh euler`
2. Check SSH key permissions: `chmod 600 ~/.ssh/id_ed25519`
3. For Docker: ensure SSH agent is running with `ssh-add -l`

### Job Failures

- Check job output: `cat slurm-<jobid>.out`
- Verify modules are available: `module avail`
- Check resource requests match cluster limits

## Next Steps

- [Learn about agents](agents.md)
- [Tool reference](tools.md)
- [Configuration guide](../configuration.md)
