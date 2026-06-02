#!/bin/sh
set -e

python_bin="${PYTHON_BIN:-python}"
if [ "$python_bin" = "python" ] && [ -x "/home/tans5/anaconda3/envs/prune_llm/bin/python" ]; then
    python_bin="/home/tans5/anaconda3/envs/prune_llm/bin/python"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

run_root="${RUN_ROOT:-out/llama3-8b/20260601_184254}"
c4_data="${C4_DATA:-c4}"
seq_len="${SEQ_LEN:-1024}"
output_dir="${OUTPUT_DIR:-load_score/out/llama3-8b/layer_heatmaps_and_shift/seq_len_$seq_len}"

"$python_bin" load_score/layer_heatmap_summary.py \
    --output_dir "$output_dir" \
    --score "llama3_c4_wanda=$run_root/wanda/wanda/$c4_data/seq_len_$seq_len" \
    --score "llama3_c4_sparsegpt=$run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len" \
    --score "llama3_c4_magnitude=$run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len" \
    --compare "llama3_c4_wanda_vs_sparsegpt=$run_root/wanda/wanda/$c4_data/seq_len_$seq_len,$run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len" \
    --compare "llama3_c4_wanda_vs_magnitude=$run_root/wanda/wanda/$c4_data/seq_len_$seq_len,$run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len" \
    --compare "llama3_c4_sparsegpt_vs_magnitude=$run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len,$run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len"
