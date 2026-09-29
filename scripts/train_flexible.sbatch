#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PWD/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export PATH="$PWD/.env/bin:/cm/shared/apps/slurm/current/bin:$PATH"
test -f data/imagenet/manifest.json
test -f reports/smoke.json
python scripts/resource_queue.py worker "$1" "$2"
