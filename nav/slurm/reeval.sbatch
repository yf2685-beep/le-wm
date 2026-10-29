#!/bin/bash
#SBATCH --job-name=lewm-reeval
#SBATCH --account=torch_pr_926_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yf2685/lewm-nav/logs/%x-%j.out
# Re-run nav/eval_latent.py on finished runs (e.g. after adding a metric); overwrites eval_val.json.
# Usage:  sbatch nav/slurm/reeval.sbatch <run_name> [<run_name> ...]

set -u
source /scratch/yf2685/lewm-nav/venv/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export NAVDP_H5_DIR=/scratch/yf2685/lewm-nav/h5 LEWM_RUN_ROOT=/scratch/yf2685/lewm-nav/runs HDF5_USE_FILE_LOCKING=FALSE
for RUN in "$@"; do
  python nav/eval_latent.py --run "$LEWM_RUN_ROOT/$RUN" --split val --max-k 8 --max-clips 20000 || exit 1
done
