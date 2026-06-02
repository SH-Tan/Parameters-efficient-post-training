#!/bin/sh
set -e
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python_bin="${PYTHON_BIN:-python}"
if [ "$python_bin" = "python" ] && [ -x "/home/tans5/anaconda3/envs/prune_llm/bin/python" ]; then
    python_bin="/home/tans5/anaconda3/envs/prune_llm/bin/python"
fi
main_py="${MAIN_PY:-main.py}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

log_msg() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

model="${MODEL:-Qwen/Qwen2.5-0.5B}"
cache_dir="${CACHE_DIR:-llm_weights}"
calib_data="${CALIB_DATA:-c4}"
pp_eval_data="${PP_EVAL_DATA:-c4_test}"
output_root="${OUTPUT_ROOT:-out/qwen2.5_0.5b}"
plot_only="${PLOT_ONLY:-0}"
if [ "$plot_only" = "1" ] && [ -z "${RUN_NAME:-}" ]; then
    latest_run_root="$(ls -1dt "$output_root"/* 2>/dev/null | head -n 1 || true)"
    if [ -z "$latest_run_root" ]; then
        echo "No existing runs found under $output_root"
        exit 1
    fi
    run_name="$(basename "$latest_run_root")"
else
    run_name="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
fi
wanda_save_dir="${WANDA_SAVE_DIR:-$output_root/$run_name/wanda}"
magnitude_save_dir="${MAGNITUDE_SAVE_DIR:-$output_root/$run_name/magnitude}"
sparsegpt_save_dir="${SPARSEGPT_SAVE_DIR:-$output_root/$run_name/sparsegpt}"

nsamples="${NSAMPLES:-128}"
seed="${SEED:-13}"
seq_len="${SEQ_LEN:-1024}"
pp_seqlen="${PP_SEQLEN:-1024}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.1 0.2 0.3 0.4 0.5 0.7}"
score_orders="${SCORE_ORDERS:-global local per_op}"

run_wanda="${RUN_WANDA:-1}"
run_magnitude="${RUN_MAGNITUDE:-1}"
run_sparsegpt="${RUN_SPARSEGPT:-1}"
run_pp_eval="${RUN_PP_EVAL:-1}"
run_downstream_eval="${RUN_DOWNSTREAM_EVAL:-1}"
run_plots="${RUN_PLOTS:-1}"
save_score_pkl="${SAVE_SCORE_PKL:-0}"
clear_results="${CLEAR_RESULTS:-1}"
model_device="${MODEL_DEVICE:-auto_free}"
downstream_task_data="${DOWNSTREAM_TASK_DATA:-ShuoZheLi/MetaMathQA-math-500}"
downstream_prompt_key="${DOWNSTREAM_PROMPT_KEY:-prompt}"
downstream_response_key="${DOWNSTREAM_RESPONSE_KEY:-}"
downstream_reward_score_dir="${DOWNSTREAM_REWARD_SCORE_DIR:-}"
downstream_max_examples="${DOWNSTREAM_MAX_EXAMPLES:-500}"
downstream_start_index="${DOWNSTREAM_START_INDEX:-0}"
downstream_shuffle="${DOWNSTREAM_SHUFFLE:-0}"
downstream_max_prompt_length="${DOWNSTREAM_MAX_PROMPT_LENGTH:-1024}"
downstream_max_new_tokens="${DOWNSTREAM_MAX_NEW_TOKENS:-1024}"
downstream_temperature="${DOWNSTREAM_TEMPERATURE:-0.0}"
downstream_top_p="${DOWNSTREAM_TOP_P:-1.0}"
downstream_top_k="${DOWNSTREAM_TOP_K:-0}"

score_pkl_arg="--save_score_pkl"
if [ "$save_score_pkl" = "0" ]; then
    score_pkl_arg="--no_save_score_pkl"
fi

pp_eval_arg=""
if [ "$run_pp_eval" = "0" ]; then
    pp_eval_arg="--skip_pp_eval"
fi

downstream_eval_arg=""
if [ "$run_downstream_eval" = "1" ]; then
    downstream_eval_arg="--do_downstream_eval --downstream_task_data $downstream_task_data --downstream_prompt_key $downstream_prompt_key --downstream_max_examples $downstream_max_examples --downstream_start_index $downstream_start_index --downstream_max_prompt_length $downstream_max_prompt_length --downstream_max_new_tokens $downstream_max_new_tokens --downstream_temperature $downstream_temperature --downstream_top_p $downstream_top_p --downstream_top_k $downstream_top_k"
    if [ -n "$downstream_response_key" ]; then
        downstream_eval_arg="$downstream_eval_arg --downstream_response_key $downstream_response_key"
    fi
    if [ -n "$downstream_reward_score_dir" ]; then
        downstream_eval_arg="$downstream_eval_arg --downstream_reward_score_dir $downstream_reward_score_dir"
    fi
    if [ "$downstream_shuffle" = "1" ]; then
        downstream_eval_arg="$downstream_eval_arg --downstream_shuffle"
    fi
fi

run_method() {
    method=$1
    method_save_dir=$2
    method_result_dir="$method_save_dir/results/$calib_data/seq_len_$seq_len"
    method_log="$method_save_dir/run.log"

    mkdir -p "$method_save_dir" "$method_result_dir"
    if [ "$clear_results" = "1" ]; then
        : > "$method_log"
        if [ "$run_pp_eval" = "1" ]; then
            : > "$method_result_dir/pp_eval_results.csv"
        fi
        if [ "$run_downstream_eval" = "1" ]; then
            : > "$method_result_dir/downstream_task_results.csv"
        fi
    fi

    log_msg "Running method=$method save_dir=$method_save_dir score_orders=$score_orders" | tee -a "$method_log"
    log_msg "Live log: $method_log"
    tail -n 0 -f "$method_log" &
    tail_pid=$!

    set +e
    "$python_bin" -u "$main_py" \
        --model "$model" \
        --prune_method "$method" \
        --save "$method_save_dir" \
        --cache_dir "$cache_dir" \
        --calib_data "$calib_data" \
        --pp_eval_data "$pp_eval_data" \
        --nsamples "$nsamples" \
        --seed "$seed" \
        --model_device "$model_device" \
        --seqlen "$seq_len" \
        --pp_seqlen $pp_seqlen \
        --sparsity_ratio $sparsity_ratios \
        --score_order $score_orders \
        $score_pkl_arg \
        $pp_eval_arg \
        $downstream_eval_arg >> "$method_log" 2>&1
    status=$?
    set -e

    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    if [ "$status" -ne 0 ]; then
        log_msg "Failed method=$method status=$status. Last log lines:" | tee -a "$method_log"
        tail -n 80 "$method_log"
        exit "$status"
    fi

    log_msg "Finished method=$method save_dir=$method_save_dir score_orders=$score_orders" | tee -a "$method_log"
}

draw_plots() {
    plot_methods=""
    if [ "$run_wanda" = "1" ]; then
        plot_methods="$plot_methods wanda"
    fi
    if [ "$run_magnitude" = "1" ]; then
        plot_methods="$plot_methods magnitude"
    fi
    if [ "$run_sparsegpt" = "1" ]; then
        plot_methods="$plot_methods sparsegpt"
    fi
    if [ -z "$plot_methods" ]; then
        echo "No methods selected for plotting."
        exit 1
    fi
    log_msg "Drawing plots for run_root=$output_root/$run_name"
    "$python_bin" -m eval.plot_results \
        --run_root "$output_root/$run_name" \
        --calib_data "$calib_data" \
        --seq_len "$seq_len" \
        --pp_seq_len "$pp_seqlen" \
        --max_sparsity 0.5 \
        --methods $plot_methods
}

if [ "$plot_only" != "1" ]; then
    mkdir -p "$output_root/$run_name"
fi
log_msg "Run output root: $output_root/$run_name"
log_msg "Calibration data: $calib_data"
log_msg "PPL eval data: $pp_eval_data"
log_msg "Downstream eval enabled: $run_downstream_eval"
log_msg "Methods: wanda=$run_wanda magnitude=$run_magnitude sparsegpt=$run_sparsegpt"
log_msg "Sparsity ratios: $sparsity_ratios"
log_msg "Score orders: $score_orders"
log_msg "WANDA save dir: $wanda_save_dir"
log_msg "Magnitude save dir: $magnitude_save_dir"
log_msg "SparseGPT save dir: $sparsegpt_save_dir"

if [ "$plot_only" = "1" ]; then
    draw_plots
else
    if [ "$run_wanda" = "1" ]; then
        run_method "wanda" "$wanda_save_dir"
    fi

    if [ "$run_magnitude" = "1" ]; then
        run_method "magnitude" "$magnitude_save_dir"
    fi

    if [ "$run_sparsegpt" = "1" ]; then
        run_method "sparsegpt" "$sparsegpt_save_dir"
    fi

    if [ "$run_plots" = "1" ] && { [ "$run_pp_eval" = "1" ] || [ "$run_downstream_eval" = "1" ]; }; then
        draw_plots
    fi
fi
