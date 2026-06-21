#!/bin/bash
#SBATCH --job-name=prune_1d5
#SBATCH --account=ASC24079
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j_prune_1d5.out
#SBATCH --error=slurm-%j_prune_1d5.err

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

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "$submit_dir/main.py" ] && [ -d "$submit_dir/scripts" ]; then
    repo_root="$(cd "$submit_dir" && pwd)"
else
    script_dir_probe="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "$script_dir_probe/.." && pwd)"
fi
script_name="deepseek_r1_distill_qwen_1.5b_multi_node.sh"
script_path="${script_path:-$repo_root/scripts/$script_name}"
main_py="${main_py:-$repo_root/main.py}"
if [ ! -f "$main_py" ] || [ ! -f "$script_path" ]; then
    echo "Could not resolve repository paths." >&2
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
    echo "  PWD=$PWD" >&2
    echo "  repo_root=$repo_root" >&2
    echo "  main_py=$main_py" >&2
    echo "  script_path=$script_path" >&2
    echo "Submit this script from the repository root, or set script_path=/absolute/path/to/$script_name." >&2
    exit 1
fi

if command -v module >/dev/null 2>&1; then
    module reset || true
    module load nvidia/25.9 || true
fi

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "Missing venv activation script: ${VENV}/bin/activate" >&2
    echo "Set VENV=/absolute/path/to/venv, matching the working GRPO scripts." >&2
    exit 1
fi
source "${VENV}/bin/activate"
python_bin="${python_bin:-$(command -v python3)}"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

scratch_root="${SCRATCH:-/tmp/${USER:-$(id -un)}}"
export HF_HOME="${HF_HOME:-$scratch_root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$scratch_root/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$scratch_root/.cache/huggingface/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$scratch_root/.cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$XDG_CACHE_HOME"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-$(id -un)}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$MPLCONFIGDIR"

log_stage() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

first_executable() {
    local candidate=""
    for candidate in "$@"; do
        [ -n "$candidate" ] && [ -x "$candidate" ] && printf '%s\n' "$candidate" && return 0
    done
    return 1
}

command_path_if_executable() {
    local command_name="$1"
    local resolved=""
    resolved="$(command -v "$command_name" 2>/dev/null || true)"
    [ -n "$resolved" ] && [ -x "$resolved" ] && printf '%s\n' "$resolved"
}

# -----------------------------------------------------------------------------
# 2. Model / data / output locations
# -----------------------------------------------------------------------------

model="${model:-/work2/09576/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B}"
cache_dir="${cache_dir:-llm_weights}"
calib_data="${calib_data:-$repo_root/dataset/deepseek1.5b/dsk_1d5_8192.parquet}"
pp_eval_data="${pp_eval_data:-wikitext2}"
downstream_task_data="${downstream_task_data:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
downstream_prompt_key="${downstream_prompt_key:-prompt}"
downstream_response_key="${downstream_response_key:-}"
downstream_reward_score_dir="${downstream_reward_score_dir:-}"
output_root="${output_root:-out/deepseek_r1_distill_qwen_1.5b}"
if [[ "$output_root" != /* ]]; then
    output_root="$repo_root/$output_root"
fi

# -----------------------------------------------------------------------------
# 2b. Multi-node launch controls
# -----------------------------------------------------------------------------
# The pruning/eval Python entry point is single-process. Multi-node execution is
# therefore implemented by assigning independent pruning methods to different
# Slurm nodes. This avoids pretending the model code is distributed while still
# using the cluster correctly for the method sweep.

multi_node="${multi_node:-1}"
gpus_per_task="${gpus_per_task:-${SLURM_GPUS_PER_TASK:-1}}"
slurm_cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
system_bash_bin="${system_bash_bin:-$(first_executable /bin/bash /usr/bin/bash)}"
srun_bin="${srun_bin:-$(first_executable /usr/bin/srun /bin/srun "$(command_path_if_executable srun)" || true)}"
scontrol_bin="${scontrol_bin:-$(first_executable /usr/bin/scontrol /bin/scontrol "$(command_path_if_executable scontrol)" || true)}"
srun_gpu_args="${srun_gpu_args:-}"
if [ -z "$srun_gpu_args" ] && [ -n "${SLURM_GPUS_PER_TASK:-}" ]; then
    srun_gpu_args="--gpus-per-task=$SLURM_GPUS_PER_TASK"
fi
multi_node_worker="${multi_node_worker:-0}"
worker_method="${worker_method:-}"

# -----------------------------------------------------------------------------
# 3. Which stages to run
# -----------------------------------------------------------------------------

run_magnitude="${run_magnitude:-0}"
run_sparsegpt="${run_sparsegpt:-0}"
run_wanda="${run_wanda:-0}"
run_random="${run_random:-1}"

run_pp_eval="${run_pp_eval:-0}"
run_downstream_eval="${run_downstream_eval:-1}"
run_plots="${run_plots:-1}"

save_score_pkl="${save_score_pkl:-0}"
clear_results="${clear_results:-1}"
stream_method_log="${stream_method_log:-1}"
keep_worker_log="${keep_worker_log:-1}"

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
multi_node_log_dir="${multi_node_log_dir:-$run_dir/slurm_logs}"

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
pp_seqlen="${pp_seqlen:-8192}"

sparsity_ratios="${sparsity_ratios:-0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
score_orders="${score_orders:-per_op local global}"
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
vllm_tensor_parallel_size="${vllm_tensor_parallel_size:-$gpus_per_task}"
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
vllm_enforce_eager="${vllm_enforce_eager:-0}"

downstream_max_examples="${downstream_max_examples:-500}"
downstream_start_index="${downstream_start_index:-0}"
downstream_shuffle="${downstream_shuffle:-0}"
downstream_batch_size="${downstream_batch_size:-32}"
downstream_generation_max_batch_tokens="${downstream_generation_max_batch_tokens:-32768}"
downstream_use_cache="${downstream_use_cache:-1}"
downstream_max_prompt_length="${downstream_max_prompt_length:-2048}"
downstream_max_new_tokens="${downstream_max_new_tokens:-4096}"
downstream_temperature="${downstream_temperature:-0.0}"
downstream_top_p="${downstream_top_p:-1.0}"
downstream_top_k="${downstream_top_k:-0}"
downstream_response_log_max="${downstream_response_log_max:-50}"
save_pruned_model="${save_pruned_model:-0}"
pruned_model_root="${pruned_model_root:-$scratch_root/pruned_checkpoints/deepseek_r1_distill_qwen_1.5b/$run_name}"

# -----------------------------------------------------------------------------
# 6. Plotting parameters
# -----------------------------------------------------------------------------

plot_max_sparsity="${plot_max_sparsity:-0.7}"

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
    [ "$save_pruned_model" = "1" ] && flags+=(--save_pruned_model)
    [ -n "$pruned_model_root" ] && flags+=(--pruned_model_root "$pruned_model_root")
    [ "$downstream_shuffle" = "1" ] && flags+=(--downstream_shuffle)
    [ "$downstream_use_cache" = "1" ] && flags+=(--downstream_use_cache)
    if [ "$vllm_enforce_eager" = "1" ]; then
        flags+=(--vllm_enforce_eager)
    else
        flags+=(--no_vllm_enforce_eager)
    fi

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

    local cmd=(
        "$python_bin" "$main_py"
        --model "$model"
        --prune_method "$method"
        --save "$method_save_dir"
        --cache_dir "$cache_dir"
        --calib_data "$calib_data"
        --pp_eval_data "$pp_eval_data"
        --nsamples "$nsamples"
        --seed "$seed"
        --model_device "$model_device"
        --calib_forward_batch_size "$method_calib_batch_size"
        --sparsegpt_hessian_chunk_size "$sparsegpt_hessian_chunk_size"
        --seqlen "$seq_len"
        --pp_seqlen "$pp_seqlen"
        --sparsity_ratio $sparsity_ratios
        --score_order $score_orders
        "${prune_ops_flags[@]}"
        "$(build_score_pkl_flag)"
        "${pp_eval_flags[@]}"
        "${downstream_flags[@]}"
    )

    if [ "$stream_method_log" = "1" ]; then
        "${cmd[@]}" 2>&1 | tee -a "$method_log"
    else
        "${cmd[@]}" >> "$method_log" 2>&1
    fi

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


save_dir_for_method() {
    case "$1" in
        wanda) echo "$wanda_save_dir" ;;
        magnitude) echo "$magnitude_save_dir" ;;
        sparsegpt) echo "$sparsegpt_save_dir" ;;
        random) echo "$random_save_dir" ;;
        *) echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

run_selected_methods_sequential() {
    local method=""
    while IFS= read -r method; do
        [ -n "$method" ] || continue
        run_method "$method" "$(save_dir_for_method "$method")"
    done < <(enabled_plot_methods)
}

run_worker_method() {
    if [ -z "$worker_method" ]; then
        echo "multi_node_worker=1 requires worker_method=<magnitude|wanda|sparsegpt|random>" >&2
        exit 1
    fi
    run_method "$worker_method" "$(save_dir_for_method "$worker_method")"
}

run_methods_multi_node() {
    if [ -z "$srun_bin" ] || [ -z "$scontrol_bin" ] || [ -z "${SLURM_JOB_NODELIST:-}" ]; then
        log_stage "No Slurm allocation detected; running selected methods sequentially"
        run_selected_methods_sequential
        return 0
    fi

    local methods=()
    local method=""
    while IFS= read -r method; do [ -n "$method" ] && methods+=("$method"); done < <(enabled_plot_methods)

    if [ "${#methods[@]}" -eq 0 ]; then
        echo "No pruning methods enabled." >&2
        exit 1
    fi

    if [ ! -x "$srun_bin" ]; then
        echo "srun_bin is not executable: $srun_bin" >&2
        echo "Set srun_bin=/absolute/path/to/srun if needed." >&2
        exit 1
    fi
    if [ ! -x "$scontrol_bin" ]; then
        echo "scontrol_bin is not executable: $scontrol_bin" >&2
        echo "Set scontrol_bin=/absolute/path/to/scontrol if needed." >&2
        exit 1
    fi

    local nodes=()
    while IFS= read -r node; do [ -n "$node" ] && nodes+=("$node"); done < <("$scontrol_bin" show hostnames "$SLURM_JOB_NODELIST")

    if [ "${#nodes[@]}" -eq 0 ]; then
        echo "Could not resolve nodes from SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST" >&2
        exit 1
    fi

    if [ ! -x "$system_bash_bin" ]; then
        echo "system_bash_bin is not executable: $system_bash_bin" >&2
        echo "Set system_bash_bin=/absolute/path/to/bash if this cluster uses a nonstandard location." >&2
        exit 1
    fi

    mkdir -p "$multi_node_log_dir"
    log_stage "Launching ${#methods[@]} method worker(s) across ${#nodes[@]} Slurm node(s)"
    if [ "${#methods[@]}" -lt "${#nodes[@]}" ]; then
        echo "Note: only ${#methods[@]} method(s) are enabled, so only ${#methods[@]} node(s) will do pruning work."
    fi
    echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
    echo "SLURM nodes: ${nodes[*]}"
    echo "Worker logs: $multi_node_log_dir"

    local pids=()
    local method_index=0
    for method in "${methods[@]}"; do
        local node="${nodes[$((method_index % ${#nodes[@]}))]}"
        local worker_log="$multi_node_log_dir/${method}.log"
        log_stage "Launching method=$method on node=$node log=$worker_log"
        if [ "$keep_worker_log" = "1" ]; then
            "$srun_bin" --nodes=1 --ntasks=1 --cpus-per-task="$slurm_cpus_per_task" $srun_gpu_args -w "$node" \
                "$system_bash_bin" -c 'source "$1/bin/activate" &&
                    export PYTHONPATH="$2"
                    export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1
                    export multi_node_worker=1 worker_method="$3" multi_node=0 run_name="$4"
                    export python_bin="$5" main_py="$6" script_path="$7"
                    exec "$8" "$7"' \
                _ "$VENV" "$PYTHONPATH" "$method" "$run_name" "$python_bin" "$main_py" "$script_path" "$system_bash_bin" \
                2>&1 | tee "$worker_log" &
        else
            "$srun_bin" --nodes=1 --ntasks=1 --cpus-per-task="$slurm_cpus_per_task" $srun_gpu_args -w "$node" \
                "$system_bash_bin" -c 'source "$1/bin/activate" &&
                    export PYTHONPATH="$2"
                    export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1
                    export multi_node_worker=1 worker_method="$3" multi_node=0 run_name="$4"
                    export python_bin="$5" main_py="$6" script_path="$7"
                    exec "$8" "$7"' \
                _ "$VENV" "$PYTHONPATH" "$method" "$run_name" "$python_bin" "$main_py" "$script_path" "$system_bash_bin" &
        fi
        pids+=("$!")
        method_index=$((method_index + 1))
    done

    local status=0
    local pid=""
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    if [ "$status" -ne 0 ]; then
        echo "At least one multi-node pruning worker failed. Check $multi_node_log_dir/*.log" >&2
        exit "$status"
    fi
    log_stage "All multi-node pruning workers finished"
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

echo "Submit dir:             $submit_dir"
echo "Repo root:              $repo_root"
echo "Script path:            $script_path"
echo "Run output root:        $run_dir"
echo "Calibration data:       $calib_data"
echo "PPL eval data:          $pp_eval_data"
echo "Prune ops:              ${prune_ops:-all}"
echo "Downstream eval enabled: $run_downstream_eval"
echo "Downstream backend:      $downstream_backend"
echo "Multi-node scheduler:    $multi_node"
echo "Worker mode:             $multi_node_worker${worker_method:+ ($worker_method)}"
echo "GPUs per worker task:    $gpus_per_task"
echo "srun GPU args:           ${srun_gpu_args:-<none>}"
echo "VENV:                   $VENV"
echo "HF home:                $HF_HOME"
echo "HF datasets cache:      $HF_DATASETS_CACHE"
echo "HF hub cache:           $HF_HUB_CACHE"
echo "System bash binary:      $system_bash_bin"
echo "srun binary:             ${srun_bin:-<not found>}"
echo "scontrol binary:         ${scontrol_bin:-<not found>}"
echo "WANDA save dir:         $wanda_save_dir"
echo "Magnitude save dir:     $magnitude_save_dir"
echo "SparseGPT save dir:     $sparsegpt_save_dir"
echo "Random save dir:        $random_save_dir"
echo "Pruned checkpoint root: $pruned_model_root"
echo "Stream method log:      $stream_method_log"
echo "Keep worker log:        $keep_worker_log"

if [ "$plot_only" = "1" ]; then
    log_stage "Plot-only mode: regenerating plots for run $run_name"
    draw_plots
else
    log_stage "Starting pruning/evaluation run $run_name"
    if [ "$multi_node_worker" = "1" ]; then
        log_stage "Worker method: $worker_method"
        run_worker_method
    elif [ "$multi_node" = "1" ]; then
        run_methods_multi_node
    else
        run_selected_methods_sequential
    fi

    if [ "$run_plots" = "1" ] && { [ "$run_pp_eval" = "1" ] || [ "$run_downstream_eval" = "1" ]; }; then
        draw_plots
    fi
fi
