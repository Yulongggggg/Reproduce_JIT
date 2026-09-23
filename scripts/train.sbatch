#!/usr/bin/env bash
#SBATCH --job-name=jit_matrix_200ep
#SBATCH --partition=alpha
#SBATCH --account=co_carson_aiaided
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:8
#SBATCH --constraint=h100|h200
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PWD/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export PATH="$PWD/.env/bin:$PATH"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
test -f data/imagenet/manifest.json
test -f reports/smoke.json
configs=(
  configs/b16_200ep.json
  configs/l16_200ep.json
)
for config in "${configs[@]}"; do
  echo "Starting official JiT 200-epoch configuration: ${config}"
  torchrun --standalone --nproc_per_node=8 scripts/run.py --mode train --config "${config}"
done
echo 'JiT-B/16 and JiT-L/16 200-epoch configurations and fixed-CFG FID-50K evaluations completed.'
