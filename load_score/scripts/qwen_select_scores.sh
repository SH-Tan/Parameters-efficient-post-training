#!/bin/sh
set -e

python_bin="${PYTHON_BIN:-python}"
if [ "$python_bin" = "python" ] && [ -x "/home/tans5/anaconda3/envs/prune_llm/bin/python" ]; then
    python_bin="/home/tans5/anaconda3/envs/prune_llm/bin/python"
fi

run_root="${RUN_ROOT:-out/qwen2.5_0.5b/20260529_154255}"
calib_data="${CALIB_DATA:-MetaMathQA-math-500}"
seq_len="${SEQ_LEN:-1024}"
ratio="${RATIO:-0.5}"
side="${SIDE:-low}"
order="${ORDER:-global}"
methods="${METHODS:-wanda magnitude sparsegpt}"
output_root="${OUTPUT_ROOT:-load_score/out/qwen2.5_0.5b}"

for method in $methods; do
    score_dir="$run_root/$method/$method/$calib_data/seq_len_$seq_len"
    output_dir="$output_root/$(basename "$run_root")/$method/$calib_data/seq_len_$seq_len"
    "$python_bin" load_score/load_score.py \
        --score_dir "$score_dir" \
        --output_dir "$output_dir" \
        --ratio "$ratio" \
        --side "$side" \
        --order "$order" \
        --name "${method}_${order}_${side}_${ratio}"
done
