#!/bin/bash
# Launch script for the Streamlit UI

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate python-ruff-template

# Launch Streamlit
streamlit run src/ui/streamlit_app.py
