#!/usr/bin/env bash
set -euo pipefail
# The first split allocations failed in ibv_modify_qp before training. Use NCCL's
# TCP transport for these candidates; leave intra-node P2P/NVLink unchanged.
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_DEBUG=INFO
export JIT_NCCL_PREFLIGHT=1
exec torchrun --nnodes=2 --nproc_per_node=2 \
  --node_rank="${SLURM_NODEID}" \
  --master_addr="${JIT_MASTER_ADDR}" --master_port="${JIT_MASTER_PORT}" \
  scripts/run.py --mode train --config "$1"
