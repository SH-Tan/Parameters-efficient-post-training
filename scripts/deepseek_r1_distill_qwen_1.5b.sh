#!/bin/bash
#
# run_prune.sh
#
# Driver script for the pruning sweep: runs one or more pruning methods
# (magnitude / wanda / sparsegpt / random) over a grid of sparsity ratios
# and score orders, optionally evaluates perplexity + a downstream task,
# and optionally plots the results.
#
# Usage:
#   ./run_prune.sh                  # run with defaults below
#   run_wanda=1 run_magnitude=0 ./run_prune.sh
#   plot_only=1 ./run_prune.sh       # re-plot the most recent run
#   plot_only=1 run_name=20260101_120000 ./run_prune.sh   # re-plot a specific run
#
# Every variable below can be overridden by exporting it (or setting it
# inline) before calling this script — nothing needs to be edited here
# unless you want to change a *default*.
#
# Note: a couple of names stay uppercase on purpose — MPLCONFIGDIR and
# PYTORCH_CUDA_ALLOC_CONF are read directly by matplotlib/PyTorch, and
# $USER is set by the OS — so those can't be renamed without breaking them.

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Python / environment
# -----------------------------------------------------------------------------

python_bin="${python_bin:-python}"
conda_python="/home/tans5/anaconda3/envs/prune_llm/bin/python"
if [ "$python_bin" = "python" ] && [ -x "$conda_python" ]; then
    python_bin="$conda_python"
fi

main_py="${main_py:-main.py}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-$(id -un)}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p "$MPLCONFIGDIR"

log_stage() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# -----------------------------------------------------------------------------
# 2. Model / data / output locations
# -----------------------------------------------------------------------------

model="${model:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
cache_dir="${cache_dir:-llm_weights}"
calib_data="${calib_data:-dataset/deepseek1.5b/dsk_1d5_8192.parquet}"
pp_eval_data="${pp_eval_data:-wikitext2}"
downstream_task_data="${downstream_task_data:-dataset/mathqa500/test.parquet}"
downstream_prompt_key="${downstream_prompt_key:-prompt}"
downstream_response_key="${downstream_response_key:-}"
downstream_reward_score_dir="${downstream_reward_score_dir:-}"
output_root="${output_root:-out/deepseek_r1_distill_qwen_1.5b}"

# -----------------------------------------------------------------------------
# 3. Which stages to run
# -----------------------------------------------------------------------------

run_magnitude="${run_magnitude:-1}"
run_sparsegpt="${run_sparsegpt:-0}"
run_wanda="${run_wanda:-1}"
run_random="${run_random:-0}"

run_pp_eval="${run_pp_eval:-0}"
run_downstream_eval="${run_downstream_eval:-1}"
run_plots="${run_plots:-1}"

save_score_pkl="${save_score_pkl:-0}"
clear_results="${clear_results:-1}"

# plot_only=1 -> skip all pruning/eval, just regenerate plots for a run.
plot_only="${plot_only:-0}"

if [ "$plot_only" = "1" ] && [ -z "${run_name:-}" ]; then
    # No run_name given while plot-only -> use the most recently created run dir.
    latest_run_root="$(ls -1dt "$output_root"/*/ 2>/dev/null | head -n 1 || true)"
    if [ -z "$latest_run_root" ]; then
        echo "No existing runs found under $output_root" >&2
        exit 1
    fi
    run_name="$(basename "$latest_run_root")"
else
    run_name="${run_name:-$(date +%Y%m%d_%H%M%S)}"
fi

run_dir="$output_root/$run_name"

wanda_save_dir="${wanda_save_dir:-$run_dir/wanda}"
magnitude_save_dir="${magnitude_save_dir:-$run_dir/magnitude}"
sparsegpt_save_dir="${sparsegpt_save_dir:-$run_dir/sparsegpt}"
random_save_dir="${random_save_dir:-$run_dir/random}"

# -----------------------------------------------------------------------------
# 4. Pruning sweep parameters
# -----------------------------------------------------------------------------

nsamples="${nsamples:-128}"
seed="${seed:-42}"
seq_len="${seq_len:-8192}"
pp_seqlen="${pp_seqlen:-2048}"

sparsity_ratios="${sparsity_ratios:-0 0.1 0.2 0.3 0.4 0.5 0.6 0.7}"
score_orders="${score_orders:-per_op}"
prune_ops="${prune_ops:-}"          # empty = all ops

calib_forward_batch_size="${calib_forward_batch_size:-128}"
wanda_calib_forward_batch_size="${wanda_calib_forward_batch_size:-128}"
sparsegpt_calib_forward_batch_size="${sparsegpt_calib_forward_batch_size:-128}"
sparsegpt_hessian_chunk_size="${sparsegpt_hessian_chunk_size:-2048}"

model_device="${model_device:-auto_free}"

# -----------------------------------------------------------------------------
# 5. Downstream eval parameters
# -----------------------------------------------------------------------------
downstream_backend="${downstream_backend:-vllm}"
vllm_tensor_parallel_size="${vllm_tensor_parallel_size:-1}"
# if [ -z "${vllm_tensor_parallel_size:-}" ]; then
#     if command -v nvidia-smi >/dev/null 2>&1; then
#         vllm_tensor_parallel_size="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
#     else
#         vllm_tensor_parallel_size="1"
#     fi
#     [ "${vllm_tensor_parallel_size:-0}" -ge 1 ] 2>/dev/null || vllm_tensor_parallel_size="1"
# fi
vllm_gpu_memory_utilization="${vllm_gpu_memory_utilization:-0.7}"
vllm_dtype="${vllm_dtype:-auto}"
vllm_python="${vllm_python:-}"

downstream_max_examples="${downstream_max_examples:-500}"
downstream_start_index="${downstream_start_index:-0}"
downstream_shuffle="${downstream_shuffle:-0}"
downstream_batch_size="${downstream_batch_size:-64}"
downstream_generation_max_batch_tokens="${downstream_generation_max_batch_tokens:-491520}"
downstream_use_cache="${downstream_use_cache:-1}"
downstream_max_prompt_length="${downstream_max_prompt_length:-2048}"
downstream_max_new_tokens="${downstream_max_new_tokens:-8192}"
downstream_temperature="${downstream_temperature:-0.0}"
downstream_top_p="${downstream_top_p:-1.0}"
downstream_top_k="${downstream_top_k:-0}"
downstream_response_log_max="${downstream_response_log_max:-50}"
save_pruned_model="${save_pruned_model:-0}"
pruned_model_root="${pruned_model_root:-}"

# -----------------------------------------------------------------------------
# 6. Plotting parameters
# -----------------------------------------------------------------------------

plot_max_sparsity="${plot_max_sparsity:-0.5}"

# =============================================================================
# Build CLI flag arrays from the config above
# =============================================================================
#
# We use bash arrays (not string concatenation) so that arguments containing
# spaces are passed correctly and the flag logic is easy to read/extend.

build_score_pkl_flag() {
    if [ "$save_score_pkl" = "1" ]; then
        echo "--save_score_pkl"
    else
        echo "--no_save_score_pkl"
    fi
}

build_pp_eval_flags() {
    # Empty if PPL eval is enabled; --skip_pp_eval if disabled.
    if [ "$run_pp_eval" = "0" ]; then
        echo "--skip_pp_eval"
    fi
}

build_prune_ops_flags() {
    if [ -n "$prune_ops" ]; then
        echo "--prune_ops $prune_ops"
    fi
}

build_downstream_eval_flags() {
    [ "$run_downstream_eval" = "1" ] || return 0

    local flags=(
        --do_downstream_eval
        --downstream_task_data "$downstream_task_data"
        --downstream_prompt_key "$downstream_prompt_key"
        --downstream_max_examples "$downstream_max_examples"
        --downstream_start_index "$downstream_start_index"
        --downstream_batch_size "$downstream_batch_size"
        --downstream_generation_max_batch_tokens "$downstream_generation_max_batch_tokens"
        --downstream_max_prompt_length "$downstream_max_prompt_length"
        --downstream_max_new_tokens "$downstream_max_new_tokens"
        --downstream_temperature "$downstream_temperature"
        --downstream_top_p "$downstream_top_p"
        --downstream_top_k "$downstream_top_k"
        --downstream_response_log_max "$downstream_response_log_max"
        --downstream_backend "$downstream_backend"
        --vllm_tensor_parallel_size "$vllm_tensor_parallel_size"
        --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization"
        --vllm_dtype "$vllm_dtype"
    )

    [ -n "$downstream_response_key" ] && flags+=(--downstream_response_key "$downstream_response_key")
    [ -n "$downstream_reward_score_dir" ] && flags+=(--downstream_reward_score_dir "$downstream_reward_score_dir")
    [ -n "$vllm_python" ] && flags+=(--vllm_python "$vllm_python")
    [ "$save_pruned_model" = "1" ] && flags+=(--save_pruned_model)
    [ -n "$pruned_model_root" ] && flags+=(--pruned_model_root "$pruned_model_root")
    [ "$downstream_shuffle" = "1" ] && flags+=(--downstream_shuffle)
    [ "$downstream_use_cache" = "1" ] && flags+=(--downstream_use_cache)

    printf '%s\n' "${flags[@]}"
}

# -----------------------------------------------------------------------------
# Per-method calibration batch size override
# -----------------------------------------------------------------------------

calib_batch_size_for_method() {
    case "$1" in
        wanda)     echo "$wanda_calib_forward_batch_size" ;;
        sparsegpt) echo "$sparsegpt_calib_forward_batch_size" ;;
        *)         echo "$calib_forward_batch_size" ;;
    esac
}

# =============================================================================
# Core actions
# =============================================================================

run_method() {
    local method="$1"
    local method_save_dir="$2"
    local calib_result_name="${calib_data//\//__}"
    calib_result_name="${calib_result_name//:/_}"
    while [[ "$calib_result_name" == [._]* ]]; do
        calib_result_name="${calib_result_name#?}"
    done
    while [[ "$calib_result_name" == *[._] ]]; do
        calib_result_name="${calib_result_name%?}"
    done
    [ -n "$calib_result_name" ] || calib_result_name="dataset"
    local method_result_dir="$method_save_dir/results/$calib_result_name/seq_len_$seq_len"
    local method_log="$method_save_dir/run.log"
    local method_calib_batch_size
    method_calib_batch_size="$(calib_batch_size_for_method "$method")"

    mkdir -p "$method_save_dir" "$method_result_dir"

    if [ "$clear_results" = "1" ]; then
        : > "$method_log"
        [ "$run_pp_eval" = "1" ] && : > "$method_result_dir/pp_eval_results.csv"
        [ "$run_downstream_eval" = "1" ] && : > "$method_result_dir/downstream_task_results.csv"
    fi

    log_stage "Starting method=$method save_dir=$method_save_dir score_orders=$score_orders sparsity_ratios=$sparsity_ratios prune_ops=${prune_ops:-all}" | tee -a "$method_log"
    log_stage "Detailed method log: $method_log"

    # Assemble the conditional flag groups into arrays.
    local prune_ops_flags=()
    local pp_eval_flags=()
    local downstream_flags=()
    while IFS= read -r line; do [ -n "$line" ] && prune_ops_flags+=($line); done < <(build_prune_ops_flags)
    while IFS= read -r line; do [ -n "$line" ] && pp_eval_flags+=("$line"); done < <(build_pp_eval_flags)
    while IFS= read -r line; do [ -n "$line" ] && downstream_flags+=("$line"); done < <(build_downstream_eval_flags)

    "$python_bin" "$main_py" \
        --model "$model" \
        --prune_method "$method" \
        --save "$method_save_dir" \
        --cache_dir "$cache_dir" \
        --calib_data "$calib_data" \
        --pp_eval_data "$pp_eval_data" \
        --nsamples "$nsamples" \
        --seed "$seed" \
        --model_device "$model_device" \
        --calib_forward_batch_size "$method_calib_batch_size" \
        --sparsegpt_hessian_chunk_size "$sparsegpt_hessian_chunk_size" \
        --seqlen "$seq_len" \
        --pp_seqlen "$pp_seqlen" \
        --sparsity_ratio $sparsity_ratios \
        --score_order $score_orders \
        "${prune_ops_flags[@]}" \
        "$(build_score_pkl_flag)" \
        "${pp_eval_flags[@]}" \
        "${downstream_flags[@]}" \
        >> "$method_log" 2>&1

    log_stage "Finished method=$method save_dir=$method_save_dir score_orders=$score_orders prune_ops=${prune_ops:-all}" | tee -a "$method_log"
}

enabled_plot_methods() {
    local methods=()
    [ "$run_wanda" = "1" ] && methods+=(wanda)
    [ "$run_magnitude" = "1" ] && methods+=(magnitude)
    [ "$run_sparsegpt" = "1" ] && methods+=(sparsegpt)
    [ "$run_random" = "1" ] && methods+=(random)
    printf '%s\n' "${methods[@]}"
}

draw_plots() {
    local plot_methods=()
    while IFS= read -r m; do [ -n "$m" ] && plot_methods+=("$m"); done < <(enabled_plot_methods)

    if [ "${#plot_methods[@]}" -eq 0 ]; then
        echo "No methods selected for plotting." >&2
        exit 1
    fi

    "$python_bin" -m eval.plot_results \
        --run_root "$run_dir" \
        --calib_data "$calib_data" \
        --seq_len "$seq_len" \
        --pp_seq_len "$pp_seqlen" \
        --max_sparsity "$plot_max_sparsity" \
        --methods "${plot_methods[@]}"
}

# =============================================================================
# Main
# =============================================================================

[ "$plot_only" != "1" ] && mkdir -p "$run_dir"

echo "Run output root:        $run_dir"
echo "Calibration data:       $calib_data"
echo "PPL eval data:          $pp_eval_data"
echo "Prune ops:              ${prune_ops:-all}"
echo "Downstream eval enabled: $run_downstream_eval"
echo "Downstream backend:      $downstream_backend"
echo "WANDA save dir:         $wanda_save_dir"
echo "Magnitude save dir:     $magnitude_save_dir"
echo "SparseGPT save dir:     $sparsegpt_save_dir"
echo "Random save dir:        $random_save_dir"

if [ "$plot_only" = "1" ]; then
    log_stage "Plot-only mode: regenerating plots for run $run_name"
    draw_plots
else
    log_stage "Starting pruning/evaluation run $run_name"
    log_stage "Enabled methods: $(enabled_plot_methods | tr '\n' ' ')"
    [ "$run_magnitude" = "1" ] && run_method "magnitude" "$magnitude_save_dir"
    [ "$run_wanda" = "1" ]     && run_method "wanda" "$wanda_save_dir"
    [ "$run_sparsegpt" = "1" ] && run_method "sparsegpt" "$sparsegpt_save_dir"
    [ "$run_random" = "1" ]    && run_method "random" "$random_save_dir"

    if [ "$run_plots" = "1" ] && { [ "$run_pp_eval" = "1" ] || [ "$run_downstream_eval" = "1" ]; }; then
        draw_plots
    fi
fi
