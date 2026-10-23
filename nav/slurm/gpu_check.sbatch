#!/bin/bash
#SBATCH --job-name=lewm-gpucheck
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/yf2685/lewm-nav/logs/%x-%j.out
# Stage 1 check: CUDA works with the venv's torch build, and a LeWM forward/backward pass runs.

source /scratch/yf2685/lewm-nav/venv/bin/activate
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python nav/gpu_check.py
