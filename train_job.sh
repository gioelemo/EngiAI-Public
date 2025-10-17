#!/bin/bash
#SBATCH --job-name=diffusion_2d_cond_beams2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load required modules (adjust for your HPC cluster)
module load python/3.9
module load cuda/11.3

# Activate virtual environment (adjust path)
source /path/to/venv/bin/activate

# Set environment variable for WandB tracking
export USE_WANDB=True

# Optional: Log in to WandB (if not already logged in)
# wandb login

# Navigate to project directory (adjust path)
cd /path/to/engiopt

# Run training command
python engiopt/diffusion_2d_cond/diffusion_2d_cond.py --problem-id "beams2d" --track --wandb-entity myusername --save-model --n-epochs 500 --seed 42

echo "Training complete!"
