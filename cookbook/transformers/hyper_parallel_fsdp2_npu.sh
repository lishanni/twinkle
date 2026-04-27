#!/usr/bin/env bash
set -euo pipefail

# N-card FSDP2 example (auto-detects card count from ASCEND_RT_VISIBLE_DEVICES):
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash cookbook/transformers/hyper_parallel_fsdp2_npu.sh

export HYPER_PARALLEL_PLATFORM=torch

# Deterministic computation for reproducible loss comparison
export HCCL_DETERMINISTIC=true

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NPROC=$(npu-smi info | grep -c "910B")
fi

LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hyper_parallel_fsdp2_npu_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo "Using $NPROC NPUs"
echo "HCCL_DETERMINISTIC=$HCCL_DETERMINISTIC"

torchrun \
  --nproc_per_node="$NPROC" \
  cookbook/transformers/hyper_parallel_fsdp2_npu.py
