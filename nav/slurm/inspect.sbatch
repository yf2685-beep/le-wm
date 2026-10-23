#!/bin/bash
#SBATCH --job-name=lewm-inspect
#SBATCH --account=torch_pr_926_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yf2685/lewm-nav/logs/%x-%j.out
# Stage 1 check: dataset statistics straight from the scene tarballs.

source /scratch/yf2685/lewm-nav/venv/bin/activate
cd "${SLURM_SUBMIT_DIR}"
python nav/inspect_navdp.py \
    --raw /scratch/yf2685/lewm-nav/raw/matterport3d_d435i \
    --out /scratch/yf2685/lewm-nav/reports/inspect \
    --workers 16
