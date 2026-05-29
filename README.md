# Parameters-efficient-post-training

Utilities for score-based pruning experiments on small language models.

The repo currently supports:

- Models: `Qwen/Qwen2.5-0.5B` and the local RLVR PPO Qwen checkpoint.
- Pruning scores: WANDA, magnitude, and SparseGPT.
- Score ordering: global, local per layer, and per operation.
- Evaluation: perplexity results saved as CSV plus comparison plots up to a target sparsity.
- Score analysis: load saved score PKLs, select low/high score parameters, and compare score similarity.

## Layout

```text
main.py                         # Qwen entrypoint
main_rlvr.py                    # RLVR checkpoint entrypoint
scripts/qwen_0.5b.sh            # Qwen run script
scripts/rlvr_ppo_qwen2.5_0.5b.sh # RLVR run script
prune_wanda.py                  # WANDA score collection
prune_magnitude.py              # magnitude score collection
prune_sparsegpt.py              # SparseGPT score collection
score_prune_utils.py            # prune by saved scores
result_utils.py                 # plot helpers
plot_results.py                 # plot CLI
load_score/                     # saved score selection and analysis
out/                            # run outputs
llm_weights/                    # local/downloaded model weights
```

## Run Qwen

Run all three methods with the default settings:

```sh
sh scripts/qwen_0.5b.sh
```

Common overrides:

```sh
CALIB_DATA=c4 \
SCORE_ORDERS="global local per_op" \
SPARSITY_RATIOS="0 0.1 0.3 0.5" \
SAVE_SCORE_PKL=1 \
sh scripts/qwen_0.5b.sh
```

Run only one method:

```sh
RUN_WANDA=1 RUN_MAGNITUDE=0 RUN_SPARSEGPT=0 sh scripts/qwen_0.5b.sh
```

## Run RLVR

The RLVR script uses `main_rlvr.py` and expects the local checkpoint at:

```text
llm_weights/rlvr_ppo_qwen2.5_0.5B_metamath_global_step_800
```

Run all three methods:

```sh
sh scripts/rlvr_ppo_qwen2.5_0.5b.sh
```

Override the model path if needed:

```sh
MODEL=/path/to/rlvr_checkpoint sh scripts/rlvr_ppo_qwen2.5_0.5b.sh
```

## Outputs

Each run is saved under:

```text
out/<model_name>/<run_name>/<method>/
```

Important files:

```text
run.log
results/<calib_data>/seq_len_<N>/pp_eval_results.csv
results/<calib_data>/seq_len_<N>/pp_eval_<method>.txt
plots/<calib_data>/seq_len_<N>/*.png
<method>/<calib_data>/seq_len_<N>/layer_000.pkl   # saved score PKLs
```

The score PKLs are saved only when `SAVE_SCORE_PKL=1`.

## Plot Existing Results

If a run already has CSV results, draw plots without rerunning pruning:

```sh
PLOT_ONLY=1 RUN_NAME=<run_name> sh scripts/qwen_0.5b.sh
```

Or call the plot CLI directly:

```sh
python plot_results.py \
  --run_root out/qwen2.5_0.5b/<run_name> \
  --calib_data c4 \
  --seq_len 1024 \
  --pp_seq_len 1024 \
  --max_sparsity 0.5
```

Compare calibration datasets for the same model:

```sh
python plot_dataset_compare.py \
  --dataset_run c4=out/qwen2.5_0.5b/<c4_run_name> \
  --dataset_run MetaMathQA-math-500=out/qwen2.5_0.5b/<metamath_run_name> \
  --output_dir out/qwen2.5_0.5b/dataset_compare \
  --seq_len 1024 \
  --pp_seq_len 1024 \
  --max_sparsity 0.5
```

This creates one plot for each method and score ordering, with one curve per dataset.

## Score Selection and Analysis

Use `load_score/` to load saved pruning scores, select low/high score parameters, and compare methods or models.

Select the lowest 50% globally:

```sh
RUN_ROOT=out/qwen2.5_0.5b/<run_name> \
CALIB_DATA=c4 \
ORDER=global \
SIDE=low \
RATIO=0.5 \
sh load_score/scripts/qwen_select_scores.sh
```

Compare selected score masks across methods:

```sh
RUN_ROOT=out/qwen2.5_0.5b/<run_name> \
CALIB_DATA=c4 \
ORDER=global \
SIDE=low \
RATIO=0.5 \
sh load_score/scripts/qwen_compare_scores.sh
```

See [load_score/README.md](load_score/README.md) for details.

## Main Options

- `CALIB_DATA`: `c4`, `wikitext2`, `MetaMathQA-math-500`, `metamathqa_math_500`, or `math_500`.
- `SCORE_ORDERS`: one or more of `global`, `local`, `per_op`.
- `SPARSITY_RATIOS`: sparsity levels to evaluate.
- `NSAMPLES`: calibration sample count.
- `SEQ_LEN`: calibration sequence length.
- `PP_SEQLEN`: perplexity evaluation sequence length.
- `MODEL_DEVICE`: default is `auto_free`, which chooses the GPU with the most free memory.
- `SAVE_SCORE_PKL`: set to `1` to keep score files for later analysis.
- `RUN_PLOTS`: set to `0` to skip plot generation.
- `PLOT_ONLY`: set to `1` to plot existing CSV results.

## Notes

- The scripts prefer `/home/tans5/anaconda3/envs/prune_llm/bin/python` when it exists.
- Use `SAVE_SCORE_PKL=0` when scores are not needed after evaluation to reduce disk usage.
- For larger models, global score operations can use more memory than local or per-op ordering.
