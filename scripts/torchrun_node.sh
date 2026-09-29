#!/usr/bin/env bash
set -euo pipefail
exec torchrun --nnodes=2 --nproc_per_node=2 \
  --node_rank="${SLURM_NODEID}" \
  --master_addr="${JIT_MASTER_ADDR}" --master_port="${JIT_MASTER_PORT}" \
  scripts/run.py --mode train --config "$1"
