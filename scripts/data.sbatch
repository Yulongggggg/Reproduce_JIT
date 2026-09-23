#!/usr/bin/env bash
#SBATCH --job-name=jit_data
#SBATCH --partition=alpha
#SBATCH --account=co_carson_aiaided
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec /mnt/home/yliu5/miniforge3/bin/python -u scripts/prepare_data.py
