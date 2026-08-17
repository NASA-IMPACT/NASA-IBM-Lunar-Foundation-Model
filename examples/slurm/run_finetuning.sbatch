#!/bin/bash
# ============================================================================
# Example SLURM wrapper for a single-GPU fine-tuning run on an HPC cluster.
#
# Edit for your site: SLURM account, partition, walltime, mem/gpu resources,
# and the conda / venv activation. The paths and the `CFG_PATH` variable below
# are placeholders — set them to your bundle layout and target config.
# ============================================================================

# --- SLURM directives -------------------------------------------------------
#SBATCH --job-name=ni-fm-finetune
#SBATCH --account=YOUR_ACCOUNT          # <-- edit
#SBATCH --partition=gpu_normal          # <-- edit for your partition
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --output=logs/finetune.out
#SBATCH --error=logs/finetune.err

# --- Config to run ----------------------------------------------------------
CFG_PATH="terratorch_integration/configs/ice_prosp/ni_lfm_ps8_all_modalities.yaml"

# --- Prep -------------------------------------------------------------------
set +e
cd "$SLURM_SUBMIT_DIR"
JOB_ID=$SLURM_JOB_ID
echo "JOB_ID: ${JOB_ID}"

# --- Environment ------------------------------------------------------------
# Load your site's conda / miniconda / pyenv here. Example (conda):
#   module load miniconda3
#   source activate ni_lfm
# Example (pyenv):
#   eval "$(pyenv init -)"
#   pyenv activate ni_lfm

# Repo root must be importable.
export PYTHONPATH="$PWD"

# --- Rename outputs to include job id --------------------------------------
FINAL_OUT="logs/${JOB_ID}_finetune.out"
FINAL_ERR="logs/${JOB_ID}_finetune.err"
[ -f "logs/finetune.out" ] && mv "logs/finetune.out" "$FINAL_OUT"
[ -f "logs/finetune.err" ] && mv "logs/finetune.err" "$FINAL_ERR"

echo "================================"
echo "NASA-IBM Lunar Foundation Model fine-tuning"
echo "  JobID:  ${JOB_ID}"
echo "  Config: ${CFG_PATH}"
echo "  Out:    ${FINAL_OUT}"
echo "  Err:    ${FINAL_ERR}"
echo "================================"

# --- Run --------------------------------------------------------------------
srun terratorch fit -c "${CFG_PATH}"

EXIT_CODE=$?
set -e
echo ""
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Fine-tuning completed successfully."
else
    echo "Fine-tuning failed with exit code ${EXIT_CODE}. See ${FINAL_ERR}."
fi
echo "================================"
exit $EXIT_CODE
