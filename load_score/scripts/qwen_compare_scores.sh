#!/bin/sh
set -e

python_bin="${PYTHON_BIN:-python}"
if [ "$python_bin" = "python" ] && [ -x "/home/tans5/anaconda3/envs/prune_llm/bin/python" ]; then
    python_bin="/home/tans5/anaconda3/envs/prune_llm/bin/python"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

run_root="${RUN_ROOT:-out/qwen2.5_0.5b/20260529_154255}"
calib_data="${CALIB_DATA:-MetaMathQA-math-500}"
seq_len="${SEQ_LEN:-1024}"
ratio="${RATIO:-0.5}"
side="${SIDE:-low}"
order="${ORDER:-global}"
metric="${METRIC:-jaccard}"
output_dir="${OUTPUT_DIR:-load_score/out/qwen2.5_0.5b/$(basename "$run_root")/analysis/$calib_data/seq_len_$seq_len}"

"$python_bin" load_score/score_analysis.py \
    --score "wanda=$run_root/wanda/wanda/$calib_data/seq_len_$seq_len" \
    --score "magnitude=$run_root/magnitude/magnitude/$calib_data/seq_len_$seq_len" \
    --score "sparsegpt=$run_root/sparsegpt/sparsegpt/$calib_data/seq_len_$seq_len" \
    --output_dir "$output_dir" \
    --ratio "$ratio" \
    --side "$side" \
    --order "$order" \
    --metric "$metric" \
    --layer_heatmaps
