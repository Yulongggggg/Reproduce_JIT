#!/usr/bin/env bash
#SBATCH --job-name=jit_smoke
#SBATCH --partition=alpha
#SBATCH --account=co_carson_aiaided
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100|h200
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PWD/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export PATH="$PWD/.env/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
torchrun --standalone --nproc_per_node=2 scripts/run.py --mode smoke
# Confirm that the pinned FID implementation loads its Inception model and reference statistics.
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048', 'logits_unbiased']).cuda()
stats = np.load('vendor/JiT/fid_stats/jit_in256_stats.npz')
print('FID extractor ready; reference keys:', stats.files)
with torch.no_grad():
    features = extractor(torch.zeros(2,3,256,256,dtype=torch.uint8,device='cuda'))
assert all(torch.isfinite(x).all() for x in features)
path = Path('reports/smoke.json')
result = json.loads(path.read_text())
result['fid_extractor'] = 'loaded and finite forward pass; reference mu/sigma available'
path.write_text(json.dumps(result,indent=2)+'\n')
PY
