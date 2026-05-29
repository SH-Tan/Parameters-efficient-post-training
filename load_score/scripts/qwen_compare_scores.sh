#!/bin/sh
set -e

python_bin="${PYTHON_BIN:-python}"
if [ "$python_bin" = "python" ] && [ -x "/home/tans5/anaconda3/envs/prune_llm/bin/python" ]; then
    python_bin="/home/tans5/anaconda3/envs/prune_llm/bin/python"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

c4_run_root="${C4_RUN_ROOT:-out/qwen2.5_0.5b/20260529_145310}"
math_run_root="${MATH_RUN_ROOT:-out/qwen2.5_0.5b/20260529_154255}"
rlvr_math_run_root="${RLVR_MATH_RUN_ROOT:-out/rlvr_ppo_qwen2.5_0.5b_metamath_global_step_800/20260529_173712}"
rlvr_c4_run_root="${RLVR_C4_RUN_ROOT:-out/rlvr_ppo_qwen2.5_0.5b_metamath_global_step_800/20260529_180622}"
c4_data="${C4_DATA:-c4}"
math_data="${MATH_DATA:-MetaMathQA-math-500}"
seq_len="${SEQ_LEN:-1024}"
output_dir="${OUTPUT_DIR:-load_score/out/qwen2.5_0.5b/layer_heatmaps_and_shift/seq_len_$seq_len}"

"$python_bin" load_score/layer_heatmap_summary.py \
    --output_dir "$output_dir" \
    --score "base_c4_wanda=$c4_run_root/wanda/wanda/$c4_data/seq_len_$seq_len" \
    --score "base_c4_sparsegpt=$c4_run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len" \
    --score "base_c4_magnitude=$c4_run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len" \
    --score "base_math_wanda=$math_run_root/wanda/wanda/$math_data/seq_len_$seq_len" \
    --score "base_math_sparsegpt=$math_run_root/sparsegpt/sparsegpt/$math_data/seq_len_$seq_len" \
    --score "rlvr_math_wanda=$rlvr_math_run_root/wanda/wanda/$math_data/seq_len_$seq_len" \
    --score "rlvr_math_sparsegpt=$rlvr_math_run_root/sparsegpt/sparsegpt/$math_data/seq_len_$seq_len" \
    --score "rlvr_c4_wanda=$rlvr_c4_run_root/wanda/wanda/$c4_data/seq_len_$seq_len" \
    --score "rlvr_c4_sparsegpt=$rlvr_c4_run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len" \
    --score "rlvr_c4_magnitude=$rlvr_c4_run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len" \
    --compare "same_base_wanda_c4_vs_math=$c4_run_root/wanda/wanda/$c4_data/seq_len_$seq_len,$math_run_root/wanda/wanda/$math_data/seq_len_$seq_len" \
    --compare "same_base_sparsegpt_c4_vs_math=$c4_run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len,$math_run_root/sparsegpt/sparsegpt/$math_data/seq_len_$seq_len" \
    --compare "same_rlvr_wanda_c4_vs_math=$rlvr_c4_run_root/wanda/wanda/$c4_data/seq_len_$seq_len,$rlvr_math_run_root/wanda/wanda/$math_data/seq_len_$seq_len" \
    --compare "same_rlvr_sparsegpt_c4_vs_math=$rlvr_c4_run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len,$rlvr_math_run_root/sparsegpt/sparsegpt/$math_data/seq_len_$seq_len" \
    --compare "two_model_c4_wanda=$c4_run_root/wanda/wanda/$c4_data/seq_len_$seq_len,$rlvr_c4_run_root/wanda/wanda/$c4_data/seq_len_$seq_len" \
    --compare "two_model_c4_sparsegpt=$c4_run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len,$rlvr_c4_run_root/sparsegpt/sparsegpt/$c4_data/seq_len_$seq_len" \
    --compare "two_model_c4_magnitude=$c4_run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len,$rlvr_c4_run_root/magnitude/magnitude/$c4_data/seq_len_$seq_len" \
    --compare "two_model_math_wanda=$math_run_root/wanda/wanda/$math_data/seq_len_$seq_len,$rlvr_math_run_root/wanda/wanda/$math_data/seq_len_$seq_len" \
    --compare "two_model_math_sparsegpt=$math_run_root/sparsegpt/sparsegpt/$math_data/seq_len_$seq_len,$rlvr_math_run_root/sparsegpt/sparsegpt/$math_data/seq_len_$seq_len"
