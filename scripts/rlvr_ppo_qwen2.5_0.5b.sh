#!/bin/sh
set -e

python_bin="${PYTHON_BIN:-python}"
if [ "$python_bin" = "python" ] && [ -x "/home/tans5/anaconda3/envs/prune_llm/bin/python" ]; then
    python_bin="/home/tans5/anaconda3/envs/prune_llm/bin/python"
fi
main_py="${MAIN_PY:-main_rlvr.py}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

model="${MODEL:-llm_weights/rlvr_ppo_qwen2.5_0.5B_metamath_global_step_800}"
cache_dir="${CACHE_DIR:-llm_weights}"
calib_data="${CALIB_DATA:-MetaMathQA-math-500}"
output_root="${OUTPUT_ROOT:-out/rlvr_ppo_qwen2.5_0.5b_metamath_global_step_800}"
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
sparsity_ratios="${SPARSITY_RATIOS:-0 0.1 0.3 0.5 0.7 0.9 1.0}"
score_orders="${SCORE_ORDERS:-global local per_op}"

run_wanda="${RUN_WANDA:-1}"
run_magnitude="${RUN_MAGNITUDE:-1}"
run_sparsegpt="${RUN_SPARSEGPT:-1}"
run_pp_eval="${RUN_PP_EVAL:-1}"
run_plots="${RUN_PLOTS:-1}"
save_score_pkl="${SAVE_SCORE_PKL:-0}"
clear_results="${CLEAR_RESULTS:-1}"
model_device="${MODEL_DEVICE:-auto_free}"

score_pkl_arg="--save_score_pkl"
if [ "$save_score_pkl" = "0" ]; then
    score_pkl_arg="--no_save_score_pkl"
fi

pp_eval_arg=""
if [ "$run_pp_eval" = "0" ]; then
    pp_eval_arg="--skip_pp_eval"
fi

run_method() {
    method=$1
    method_save_dir=$2
    method_result_dir="$method_save_dir/results/$calib_data/seq_len_$seq_len"
    method_log="$method_save_dir/run.log"

    mkdir -p "$method_save_dir" "$method_result_dir"
    if [ "$clear_results" = "1" ]; then
        : > "$method_log"
        : > "$method_result_dir/pp_eval_results.csv"
        : > "$method_result_dir/pp_eval_${method}.txt"
    fi

    echo "Running method=$method save_dir=$method_save_dir score_orders=$score_orders" | tee -a "$method_log"
    "$python_bin" "$main_py" \
        --model "$model" \
        --prune_method "$method" \
        --save "$method_save_dir" \
        --cache_dir "$cache_dir" \
        --calib_data "$calib_data" \
        --nsamples "$nsamples" \
        --seed "$seed" \
        --model_device "$model_device" \
        --seqlen "$seq_len" \
        --pp_seqlen $pp_seqlen \
        --sparsity_ratio $sparsity_ratios \
        --score_order $score_orders \
        $score_pkl_arg \
        $pp_eval_arg >> "$method_log" 2>&1
    echo "Finished method=$method save_dir=$method_save_dir score_orders=$score_orders" | tee -a "$method_log"
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
    "$python_bin" plot_results.py \
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
echo "Run output root: $output_root/$run_name"
echo "WANDA save dir: $wanda_save_dir"
echo "Magnitude save dir: $magnitude_save_dir"
echo "SparseGPT save dir: $sparsegpt_save_dir"

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

    if [ "$run_pp_eval" = "1" ] && [ "$run_plots" = "1" ]; then
        draw_plots
    fi
fi
