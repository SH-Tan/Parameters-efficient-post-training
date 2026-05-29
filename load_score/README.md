# Load Score Utilities

Utilities for loading saved pruning score PKLs, selecting score subsets, and comparing
score agreement across methods/models.

The expected score directory format is the one produced by the pruning pipeline:

```text
out/<model>/<run>/<method>/<method>/<calib_data>/seq_len_<N>/layer_000.pkl
```

Each `layer_XXX.pkl` stores per-op score tensors.

## Select Scores

Build a reusable boolean mask for the lowest or highest scored parameters.

```sh
python load_score/load_score.py \
  --score_dir out/qwen2.5_0.5b/20260529_154255/wanda/wanda/MetaMathQA-math-500/seq_len_1024 \
  --output_dir load_score/out/example/wanda \
  --ratio 0.5 \
  --side low \
  --order global
```

Arguments:

- `--ratio 0.5`: select 50% of parameters.
- `--side low`: select low-score parameters, usually treated as unimportant.
- `--side high`: select high-score parameters.
- `--order global`: rank scores across all layers/ops.
- `--order local`: rank scores within each layer.
- `--order per_op`: rank scores within each operation.

The saved mask can later be used for training control. For example, if you want to
freeze the early/lowest 50% unimportant parameters, use:

```sh
--ratio 0.5 --side low --order global
```

The output contains:

- `*_mask.pkl`: boolean masks keyed by `layer:op`.
- `*_summary.csv`: selected count and selected ratio per layer/op.

## Compare Scores

Compare selections across methods/models using Jaccard or overlap similarity.

```sh
python load_score/score_analysis.py \
  --score wanda=out/qwen2.5_0.5b/20260529_154255/wanda/wanda/MetaMathQA-math-500/seq_len_1024 \
  --score magnitude=out/qwen2.5_0.5b/20260529_154255/magnitude/magnitude/MetaMathQA-math-500/seq_len_1024 \
  --score sparsegpt=out/qwen2.5_0.5b/20260529_154255/sparsegpt/sparsegpt/MetaMathQA-math-500/seq_len_1024 \
  --output_dir load_score/out/qwen_analysis \
  --ratio 0.5 \
  --side low \
  --order global \
  --metric jaccard \
  --layer_heatmaps
```

Outputs:

- similarity matrix CSV
- similarity heatmap PNG
- optional per-method layer/op heatmaps showing selected ratios

## Scripts

Select masks for each method:

```sh
sh load_score/scripts/qwen_select_scores.sh
```

Compare methods:

```sh
sh load_score/scripts/qwen_compare_scores.sh
```

Common overrides:

```sh
RUN_ROOT=out/qwen2.5_0.5b/20260529_154255 \
CALIB_DATA=MetaMathQA-math-500 \
RATIO=0.5 \
SIDE=low \
ORDER=global \
sh load_score/scripts/qwen_compare_scores.sh
```
