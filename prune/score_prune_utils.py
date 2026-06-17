import math

import torch

from prune import filter_prune_ops, find_layers
from utils.score_io_utils import load_layer_scores


def _lowest_mask(metric, prune_count):
    prune_count = int(prune_count)
    if prune_count <= 0:
        return torch.zeros_like(metric, dtype=torch.bool)
    if prune_count >= metric.numel():
        return torch.ones_like(metric, dtype=torch.bool)

    flat_metric = metric.reshape(-1).float()
    threshold = torch.kthvalue(flat_metric, k=prune_count).values
    mask = flat_metric <= threshold
    extra = int(mask.sum().item()) - prune_count
    if extra > 0:
        tied = (flat_metric == threshold).nonzero(as_tuple=False).flatten()
        mask[tied[:extra]] = False
    return mask.reshape(metric.shape)


def _validate_score_shape(score, weight, layer_idx, name):
    if tuple(score.shape) != tuple(weight.shape):
        raise ValueError(
            f"Score shape mismatch for layer {layer_idx} {name}: "
            f"{tuple(score.shape)} vs {tuple(weight.shape)}"
        )


def _get_layer_scores(score_source, layer_idx):
    if isinstance(score_source, (str, bytes)):
        return load_layer_scores(score_source, layer_idx)
    return score_source[layer_idx]


def _iter_layer_modules_with_scores(model, score_source, prune_ops=None):
    for layer_idx, layer in enumerate(model.model.layers):
        layer_scores = _get_layer_scores(score_source, layer_idx)
        subset = filter_prune_ops(find_layers(layer), prune_ops)
        for name, module in subset.items():
            if name not in layer_scores:
                raise KeyError(f"Missing scores for layer {layer_idx} {name}")
            score = layer_scores[name]
            _validate_score_shape(score, module.weight.data, layer_idx, name)
            yield layer_idx, name, module, score


def _score_min_max_count(model, score_source, prune_ops=None):
    min_score = None
    max_score = None
    total_count = 0

    for _, _, _, score in _iter_layer_modules_with_scores(model, score_source, prune_ops):
        score = score.detach().float()
        score_min = float(score.min().item())
        score_max = float(score.max().item())
        min_score = score_min if min_score is None else min(min_score, score_min)
        max_score = score_max if max_score is None else max(max_score, score_max)
        total_count += score.numel()

    return min_score, max_score, total_count


def _count_scores_leq(model, score_source, threshold, prune_ops=None):
    count = 0
    for _, _, _, score in _iter_layer_modules_with_scores(model, score_source, prune_ops):
        count += int((score.float() <= threshold).sum().item())
    return count


def _count_scores_lt(model, score_source, threshold, prune_ops=None):
    count = 0
    for _, _, _, score in _iter_layer_modules_with_scores(model, score_source, prune_ops):
        count += int((score.float() < threshold).sum().item())
    return count


def _next_score_above(model, score_source, threshold, prune_ops=None, chunk_size=1000000):
    next_score = None
    for _, _, _, score in _iter_layer_modules_with_scores(model, score_source, prune_ops):
        flat_score = score.reshape(-1)
        for start in range(0, flat_score.numel(), chunk_size):
            chunk = flat_score[start:start + chunk_size].float()
            mask = chunk > threshold
            if not mask.any().item():
                continue
            candidate = float(chunk[mask].min().item())
            next_score = candidate if next_score is None else min(next_score, candidate)
    return next_score


def _global_low_score_threshold(model, score_source, sparsity_ratio, prune_ops=None, steps=24):
    min_score, max_score, total_count = _score_min_max_count(model, score_source, prune_ops)
    prune_count = int(total_count * sparsity_ratio)
    if prune_count <= 0:
        return None, 0
    if prune_count >= total_count:
        return max_score, total_count

    if min_score >= 0 and max_score > 0:
        zero_count = _count_scores_leq(model, score_source, 0.0, prune_ops)
        if prune_count <= zero_count:
            return 0.0, prune_count

        min_positive = _next_score_above(model, score_source, 0.0, prune_ops)
        if min_positive is not None:
            low = math.log(min_positive)
            high = math.log(max_score)
            for _ in range(steps):
                mid = (low + high) / 2.0
                if _count_scores_leq(model, score_source, math.exp(mid), prune_ops) >= prune_count:
                    high = mid
                else:
                    low = mid
            return math.exp(high), prune_count

    low = min_score
    high = max_score
    for _ in range(steps):
        mid = (low + high) / 2.0
        if _count_scores_leq(model, score_source, mid, prune_ops) >= prune_count:
            high = mid
        else:
            low = mid

    return high, prune_count


def _apply_global_pruning(model, score_source, threshold, target_count, prune_ops=None):
    lower_count = _count_scores_lt(model, score_source, threshold, prune_ops)
    tie_budget = max(0, int(target_count) - lower_count)
    pruned = 0
    total = 0

    for layer_idx, name, module, score in _iter_layer_modules_with_scores(model, score_source, prune_ops):
        score = score.float()
        mask = score < threshold
        if tie_budget > 0:
            tied = (score == threshold).reshape(-1).nonzero(as_tuple=False).flatten()
            if tied.numel() > 0:
                selected = tied[:tie_budget]
                flat_mask = mask.reshape(-1)
                flat_mask[selected] = True
                tie_budget -= int(selected.numel())

        selected_count = int(mask.sum().item())
        overflow = pruned + selected_count - int(target_count)
        if overflow > 0:
            flat_mask = mask.reshape(-1)
            selected = flat_mask.nonzero(as_tuple=False).flatten()
            flat_mask[selected[-overflow:]] = False

        weight = module.weight.data
        weight[mask.to(device=weight.device)] = 0
        pruned += int(mask.sum().item())
        total += mask.numel()
        print(f"pruned layer {layer_idx} {name}: {int(mask.sum().item())}/{mask.numel()}")
    return pruned, total


def _apply_local_pruning(model, score_source, sparsity_ratio, prune_ops=None):
    pruned = 0
    total = 0
    for layer_idx, layer in enumerate(model.model.layers):
        layer_scores = _get_layer_scores(score_source, layer_idx)
        subset = filter_prune_ops(find_layers(layer), prune_ops)
        layer_count = sum(layer_scores[name].numel() for name in subset)
        total += layer_count
        layer_prune_count = int(layer_count * sparsity_ratio)
        if layer_prune_count <= 0:
            continue

        flat_scores = torch.cat([
            layer_scores[name].reshape(-1).float()
            for name in subset
        ])
        flat_mask = _lowest_mask(flat_scores, layer_prune_count)
        offset = 0

        for name, module in subset.items():
            score = layer_scores[name]
            _validate_score_shape(score, module.weight.data, layer_idx, name)
            next_offset = offset + score.numel()
            mask = flat_mask[offset:next_offset].reshape(score.shape)
            offset = next_offset
            weight = module.weight.data
            weight[mask.to(device=weight.device)] = 0
            pruned += int(mask.sum().item())
            print(f"pruned layer {layer_idx} {name}: {int(mask.sum().item())}/{mask.numel()}")
        del flat_scores, flat_mask
    return pruned, total


def _apply_per_op_pruning(model, score_source, sparsity_ratio, prune_ops=None):
    pruned = 0
    total = 0
    for layer_idx, name, module, score in _iter_layer_modules_with_scores(model, score_source, prune_ops):
        mask = _lowest_mask(score, int(score.numel() * sparsity_ratio))
        weight = module.weight.data
        weight[mask.to(device=weight.device)] = 0
        pruned += int(mask.sum().item())
        total += mask.numel()
        print(f"pruned layer {layer_idx} {name}: {int(mask.sum().item())}/{mask.numel()}")
    return pruned, total


def prune_by_scores(model, score_source, sparsity_ratio, score_order, prune_ops=None):
    if sparsity_ratio <= 0:
        _, _, total_count = _score_min_max_count(model, score_source, prune_ops)
        return {"pruned": 0, "total": total_count, "actual_sparsity": 0.0}
    if sparsity_ratio > 1:
        raise ValueError(f"sparsity_ratio must be in [0, 1], got {sparsity_ratio}")

    if score_order == "global":
        threshold, target_count = _global_low_score_threshold(model, score_source, sparsity_ratio, prune_ops)
        pruned, total = _apply_global_pruning(model, score_source, threshold, target_count, prune_ops)
    elif score_order == "local":
        pruned, total = _apply_local_pruning(model, score_source, sparsity_ratio, prune_ops)
    elif score_order == "per_op":
        pruned, total = _apply_per_op_pruning(model, score_source, sparsity_ratio, prune_ops)
    else:
        raise ValueError(f"Unsupported score_order: {score_order}")

    return {
        "pruned": pruned,
        "total": total,
        "actual_sparsity": 0.0 if total == 0 else pruned / total,
    }
