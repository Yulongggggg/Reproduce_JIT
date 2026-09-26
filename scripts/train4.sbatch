#!/usr/bin/env bash
#SBATCH --job-name=jit_b16_l16_4gpu
#SBATCH --partition=alpha
#SBATCH --account=co_carson_aiaided
#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100|h200
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train4-%j.log
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
for config in configs/b16_4gpu_200ep.json configs/l16_4gpu_200ep.json; do
  out=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$config")
  if python3 -c "import json,sys; sys.exit(json.load(open(sys.argv[1])).get('status') != 'complete')" "$out/summary.json" 2>/dev/null; then
    echo "Already complete, skipping: ${config}"
    continue
  fi
  echo "Starting 4-GPU, global-batch-1024 JiT training: ${config}"
  torchrun --standalone --nproc_per_node=4 scripts/run.py --mode train --config "$config"
done
