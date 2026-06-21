# Parameters-efficient-post-training

Utilities for memory-aware score-based pruning experiments on language models.

The repo currently supports:

- Models: Hugging Face causal LM checkpoints such as `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and `Qwen/Qwen2.5-0.5B`.
- Pruning methods: magnitude, WANDA, SparseGPT, and random.
- Score ordering: `per_op`, `local`, and `global`.
- Evaluation: optional perplexity evaluation plus downstream task accuracy.
- Downstream generation backends: Transformers or isolated vLLM subprocess.
- Score analysis: optional score PKL saving, selection, and comparison under `load_score/`.

## Layout

```text
main.py                              # pruning/evaluation entrypoint
scripts/deepseek_r1_distill_qwen_1.5b.sh # current DeepSeek sweep script
scripts/qwen_0.5b.sh                 # Qwen sweep script
scripts/rlvr_ppo_qwen2.5_0.5b.sh     # RLVR checkpoint sweep script
prune/                               # pruning methods and score utilities
eval/                                # pruning eval loop, PPL eval, plot CLIs
downstream_eval/                     # downstream task and vLLM runners
utils/                               # model loading and CLI args
load_score/                          # saved score selection and analysis
dataset/                             # local calibration/eval data
llm_weights/                         # local/downloaded model cache
out/                                 # run outputs
```

## Run DeepSeek

The current main script is:

```sh
bash scripts/deepseek_r1_distill_qwen_1.5b.sh
```

Default behavior:

- Uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- Uses calibration data `dataset/deepseek1.5b/dsk_1d5_8192.parquet`.
- Runs magnitude only by default.
- Runs downstream eval with `vllm`.
- Skips perplexity eval by default.
- Recomputes scores in memory by default instead of saving score PKLs.
- Uses `CUDA_VISIBLE_DEVICES=1` unless overridden.

Common overrides:

```sh
CUDA_VISIBLE_DEVICES=0 \
run_magnitude=1 \
run_wanda=1 \
run_sparsegpt=0 \
sparsity_ratios="0 0.1 0.2 0.3 0.4 0.5" \
score_orders="per_op local global" \
bash scripts/deepseek_r1_distill_qwen_1.5b.sh
```

Use a smaller downstream run:

```sh
downstream_max_examples=50 \
downstream_batch_size=8 \
downstream_generation_max_batch_tokens=65536 \
bash scripts/deepseek_r1_distill_qwen_1.5b.sh
```

Plot an existing run only:

```sh
plot_only=1 run_name=<run_name> bash scripts/deepseek_r1_distill_qwen_1.5b.sh
```

If `run_name` is omitted in plot-only mode, the script uses the latest run under `out/deepseek_r1_distill_qwen_1.5b/`.

## vLLM Notes

The main pruning script prefers `/home/tans5/anaconda3/envs/prune_llm/bin/python` for pruning when it exists. The vLLM downstream stage runs as a separate subprocess.

By default the vLLM subprocess command starts with:

```sh
python -m downstream_eval.vllm_accuracy_runner ...
```

That means it uses the active Bash environment's `python`. Activate the vLLM environment before running the script, or pass an explicit Python:

```sh
vllm_python=/home/tans5/anaconda3/envs/vllm/bin/python \
bash scripts/deepseek_r1_distill_qwen_1.5b.sh
```

For CUDA multiprocessing with vLLM:

```sh
VLLM_WORKER_MULTIPROC_METHOD=spawn bash scripts/deepseek_r1_distill_qwen_1.5b.sh
```

vLLM downstream eval saves pruned checkpoints automatically, because vLLM loads from a Hugging Face checkpoint directory.

## Outputs

Each run is saved under:

```text
out/<model_name>/<run_name>/<method>/
```

Important files:

```text
run.log
results/<calib_data>/seq_len_<N>/pp_eval_results.csv
results/<calib_data>/seq_len_<N>/downstream_task_results.csv
results/<calib_data>/seq_len_<N>/downstream_task_responses_<order>_sparsity_<ratio>.jsonl
results/<calib_data>/seq_len_<N>/downstream_task_metrics_<order>_sparsity_<ratio>.json
results/<calib_data>/seq_len_<N>/pruned_models/<method>_<order>_sparsity_<ratio>/
plots/<calib_data>/seq_len_<N>/*.png
```

Score PKLs are saved only when `save_score_pkl=1` or `--save_score_pkl` is used.

## Main Script Options

Most script variables can be overridden inline:

- `CUDA_VISIBLE_DEVICES`: GPU ids visible to pruning and vLLM. Default in the DeepSeek script is `1`.
- `model`: HF model name or local checkpoint path.
- `cache_dir`: model cache directory. Default `llm_weights`.
- `calib_data`: calibration parquet path or dataset alias.
- `downstream_task_data`: downstream parquet path or HF dataset id.
- `run_magnitude`, `run_wanda`, `run_sparsegpt`, `run_random`: enable methods with `1` or disable with `0`.
- `run_pp_eval`: set `1` to run perplexity evaluation.
- `run_downstream_eval`: set `1` to run downstream task accuracy.
- `run_plots`: set `0` to skip plot generation.
- `clear_results`: set `0` to append to existing CSV/log files.
- `nsamples`: calibration sample count.
- `seq_len`: calibration sequence length.
- `pp_seqlen`: perplexity evaluation sequence length.
- `sparsity_ratios`: space-separated sparsity ratios.
- `score_orders`: one or more of `per_op`, `local`, `global`.
- `prune_ops`: optional operation subset, such as `q k v` or `q_proj v_proj`.
- `calib_forward_batch_size`: default forward microbatch size.
- `wanda_calib_forward_batch_size`: WANDA-specific forward microbatch size.
- `sparsegpt_calib_forward_batch_size`: SparseGPT-specific forward microbatch size.
- `sparsegpt_hessian_chunk_size`: SparseGPT Hessian token chunk size.
- `model_device`: default `auto_free`, which picks the GPU with the most free memory.
- `downstream_backend`: `vllm` or `transformers`.
- `vllm_tensor_parallel_size`: vLLM tensor parallel size.
- `vllm_gpu_memory_utilization`: vLLM GPU memory fraction.
- `vllm_dtype`: vLLM dtype, default `auto`.
- `vllm_python`: optional Python executable for the vLLM subprocess.
- `downstream_max_examples`: downstream example count; use `-1` for all.
- `downstream_batch_size`: max examples per downstream generation microbatch.
- `downstream_generation_max_batch_tokens`: cap prompt plus generation tokens per microbatch.
- `downstream_max_prompt_length`: prompt truncation length.
- `downstream_max_new_tokens`: generation length.
- `downstream_response_log_max`: number of full responses to write; `-1` writes all.
- `save_pruned_model`: keep pruned Hugging Face checkpoints after eval.
- `pruned_model_root`: optional root for pruned checkpoints.
- `save_score_pkl`: set `1` to persist score PKLs.
- `plot_only`: set `1` to regenerate plots without pruning/eval.

## Direct CLI

The script wraps `main.py`. You can also call it directly:

```sh
python main.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --prune_method magnitude \
  --save out/manual/magnitude \
  --cache_dir llm_weights \
  --calib_data dataset/deepseek1.5b/dsk_1d5_8192.parquet \
  --seqlen 8192 \
  --sparsity_ratio 0 0.1 0.2 \
  --score_order per_op local global \
  --skip_pp_eval \
  --do_downstream_eval \
  --downstream_backend vllm
```

## Score Selection and Analysis

Use `load_score/` to load saved pruning scores, select low/high score parameters, and compare methods or models.

Select the lowest 50% globally:

```sh
RUN_ROOT=out/deepseek_r1_distill_qwen_1.5b/<run_name> \
CALIB_DATA=dataset__deepseek1.5b__dsk_1d5_8192.parquet \
SEQ_LEN=8192 \
ORDER=global \
SIDE=low \
RATIO=0.5 \
sh load_score/scripts/qwen_select_scores.sh
```

See [load_score/README.md](load_score/README.md) for details.

## Memory Notes

- Keep `save_score_pkl=0` when score files are not needed after evaluation.
- Prefer `per_op` or `local` ordering when memory is tight; `global` score sorting can require more memory.
- Lower `calib_forward_batch_size` or method-specific batch sizes if calibration OOMs.
- Lower `downstream_batch_size` or `downstream_generation_max_batch_tokens` if vLLM generation OOMs.
- Use `CUDA_VISIBLE_DEVICES=<ids>` to isolate the run to selected GPUs.
