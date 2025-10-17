#!/usr/bin/env python3
"""
Training script for EngiOpt diffusion_2d_cond model on beams2d problem.
Generated automatically - can be customized as needed.
"""

import os
import subprocess

# Set USE_WANDB environment variable
os.environ["USE_WANDB"] = "True"

# Training configuration
use_wandb = os.getenv("USE_WANDB") == "True"
track_flag = "--track" if use_wandb else "--no-track"

# Build and run the command
command = [
    "python",
    "engiopt/diffusion_2d_cond/diffusion_2d_cond.py",
    "--problem-id",
    "beams2d",
    track_flag,
    "--wandb-entity",
    "myusername",
    "--save-model",
    "--n-epochs",
    "500",
    "--seed",
    "42",
]

print(f"Running command: {' '.join(command)}")
subprocess.run(command, check=True)
