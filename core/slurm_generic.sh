#!/bin/bash
# =============================================================================
# slurm_generic.sh — Generic SLURM array template for Hercules (CICA)
#
# This script is the execution engine of the framework.
# It reads one command per line from CMD_FILE and runs the command
# corresponding to the current SLURM array task index.
#
# Usage (never call directly — submitted by manager.py):
#   sbatch --array=1-N --export=CMD_FILE=cmds_phase.txt slurm_generic.sh
#
# Required --export variables:
#   CMD_FILE : path to the command list file (one command per line)
#
# Optional environment variables:
#   CONDA_ENV  : conda environment name (default: experiment)
#   CONDA_BASE : path to Miniconda installation (auto-detected if unset)
# =============================================================================

# ── Resource defaults (override with sbatch flags or manager.py) ─────────────
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dmarper2@upo.es

# ── 1. Conda activation ───────────────────────────────────────────────────────
# Hercules stores Miniconda at a fixed path. Other clusters fall back to
# conda info --base or $HOME/miniconda3.

HERCULES_CONDA="/lustre/software/easybuild/common/software/Miniconda3/4.9.2"
CONDA_ENV="${CONDA_ENV:-experiment}"

if [ -f "$HERCULES_CONDA/etc/profile.d/conda.sh" ]; then
    source "$HERCULES_CONDA/etc/profile.d/conda.sh"
else
    _BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
    if [ -f "$_BASE/etc/profile.d/conda.sh" ]; then
        source "$_BASE/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Cannot find conda. Set CONDA_BASE or install Miniconda."
        exit 1
    fi
fi

# Hercules requires 'source activate' instead of 'conda activate'
source activate "$CONDA_ENV" || {
    echo "[ERROR] Cannot activate conda environment: $CONDA_ENV"
    exit 1
}

# ── 2. Move to submission directory ───────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || { echo "[ERROR] SLURM_SUBMIT_DIR not set."; exit 1; }

# ── 3. Select the command for this array task ──────────────────────────────────
if [ -z "$CMD_FILE" ]; then
    echo "[ERROR] CMD_FILE not set. Pass it with --export=CMD_FILE=path."
    exit 1
fi

if [ ! -f "$CMD_FILE" ]; then
    echo "[ERROR] CMD_FILE not found: $CMD_FILE"
    exit 1
fi

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")

if [ -z "$CMD" ]; then
    echo "[ERROR] No command at line $SLURM_ARRAY_TASK_ID in $CMD_FILE"
    exit 1
fi

# ── 4. Run ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job     : $SLURM_ARRAY_JOB_ID  |  Task: $SLURM_ARRAY_TASK_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Env     : $CONDA_ENV"
echo "  Start   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Command : $CMD"
echo "============================================================"

eval "$CMD"

EXIT=$?
echo "------------------------------------------------------------"
echo "  Finish  : $(date '+%Y-%m-%d %H:%M:%S')  |  Exit: $EXIT"
echo "============================================================"
exit $EXIT
