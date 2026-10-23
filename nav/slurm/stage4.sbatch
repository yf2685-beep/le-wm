#!/bin/bash
#SBATCH --job-name=lewm-s4
#SBATCH --account=torch_pr_926_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --array=1-28
#SBATCH --output=/scratch/yf2685/lewm-nav/logs/%x-%A_%a.out
# Stage 4: one line of nav/slurm/stage4_sweep.txt per array task: train (resumes from last.ckpt), then eval on val.
# Submit from the repo root:  sbatch nav/slurm/stage4.sbatch      (subset: sbatch --array=1,5 ...)

set -u
source /scratch/yf2685/lewm-nav/venv/bin/activate
cd "${SLURM_SUBMIT_DIR}"
export NAVDP_H5_DIR=/scratch/yf2685/lewm-nav/h5 LEWM_RUN_ROOT=/scratch/yf2685/lewm-nav/runs HDF5_USE_FILE_LOCKING=FALSE

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" nav/slurm/stage4_sweep.txt)
RUN=$(echo "$ARGS" | sed -E 's/.*run_name=([^ ]+).*/\1/')
echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname): $ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$LEWM_RUN_ROOT/$RUN/DONE" ]; then
  python nav/train_nav.py $ARGS nav.clip_stride=4 trainer.max_epochs=10 num_workers=10 || exit 1
fi
if [ ! -f "$LEWM_RUN_ROOT/$RUN/eval_val.json" ]; then
  python nav/eval_latent.py --run "$LEWM_RUN_ROOT/$RUN" --split val --max-k 8 --max-clips 20000
fi
