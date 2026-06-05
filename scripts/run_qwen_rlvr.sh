#!/bin/sh
set -e

calib_data="${CALIB_DATA:-actor_math_500_response}"
run_qwen="${RUN_QWEN:-1}"
run_rlvr="${RUN_RLVR:-1}"
qwen_script="${QWEN_SCRIPT:-scripts/qwen_0.5b.sh}"
rlvr_script="${RLVR_SCRIPT:-scripts/rlvr_ppo_qwen2.5_0.5b.sh}"

if [ "$run_qwen" = "1" ]; then
    CALIB_DATA="$calib_data" sh "$qwen_script"
fi

if [ "$run_rlvr" = "1" ]; then
    CALIB_DATA="$calib_data" sh "$rlvr_script"
fi
