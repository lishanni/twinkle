#!/usr/bin/env bash
set -euo pipefail

# Profile Qwen3-32B: native FSDP2 vs HyperParallel FSDP2
# Runs both backends sequentially, profiling step 10 only each time.
#
# Usage:
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
#   bash cookbook/transformers/profile_fsdp2_compare_32b.sh

export HYPER_PARALLEL_PLATFORM=torch

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NPROC=$(npu-smi info | grep -c "910B")
fi

LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/log"
mkdir -p "$LOG_DIR"

SCRIPT="cookbook/transformers/profile_fsdp2_compare_32b.py"

for BACKEND in native hp; do
    LOG_FILE="$LOG_DIR/profile_32b_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"
    echo "===== Running backend: $BACKEND | $NPROC NPUs ====="
    echo "Log: $LOG_FILE"
    torchrun --nproc_per_node="$NPROC" "$SCRIPT" --backend "$BACKEND" 2>&1 | tee "$LOG_FILE"
    echo "===== Finished backend: $BACKEND ====="
    echo
done

echo "All done. Compare profiling data under log/profiling/32b_native_fsdp2 vs log/profiling/32b_hp_fsdp2"
