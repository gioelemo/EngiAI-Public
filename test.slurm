#!/bin/bash
#SBATCH --job-name=cgan_cnn_2d_beams2d
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7GB
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=engiopt_cgan_cnn_2d_beams2d_%j.out
#SBATCH --error=engiopt_cgan_cnn_2d_beams2d_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gioelemo@ethz.ch

# Load required modules
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

# Activate virtual environment
source ~/venvs/engineer_assistant/bin/activate

# Navigate to project directory
cd $HOME/EngiOpt

# Run training command
python engiopt/cgan_cnn_2d/cgan_cnn_2d.py --problem-id "beams2d" --no-track --wandb-entity None --save-model --n-epochs 50 --seed 1

echo "Training complete!"
