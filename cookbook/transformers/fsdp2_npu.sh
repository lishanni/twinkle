#!/usr/bin/env bash
set -euo pipefail

# Twinkle native FSDP2 baseline on NPU (no hyper_parallel):
#   ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
#   bash cookbook/transformers/fsdp2_npu.sh

# Deterministic computation for reproducible loss comparison
export HCCL_DETERMINISTIC=true

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NPROC=$(npu-smi info | grep -c "910B")
fi

LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fsdp2_npu_baseline_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo "Using $NPROC NPUs (native FSDP2 baseline)"
echo "HCCL_DETERMINISTIC=$HCCL_DETERMINISTIC"

torchrun \
  --nproc_per_node="$NPROC" \
  cookbook/transformers/fsdp2_npu.py
