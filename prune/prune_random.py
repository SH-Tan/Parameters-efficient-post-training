import numpy as np
import torch

from prune import find_layers


def _iter_prunable_modules(model):
    for layer_idx, layer in enumerate(model.model.layers):
        for name, module in find_layers(layer).items():
            yield layer_idx, name, module


def _module_prune_counts(modules, target_count, rng):
    remaining_total = sum(module.weight.numel() for _, _, module in modules)
    remaining_prune = int(target_count)
    counts = []

    for _, _, module in modules:
        numel = module.weight.numel()
        if remaining_prune <= 0:
            prune_count = 0
        elif remaining_prune >= remaining_total:
            prune_count = numel
        else:
            prune_count = int(rng.hypergeometric(numel, remaining_total - numel, remaining_prune))

        counts.append(prune_count)
        remaining_total -= numel
        remaining_prune -= prune_count

    return counts


def _random_zero_module(module, prune_count, generator):
    weight = module.weight.data
    numel = weight.numel()
    prune_count = int(prune_count)
    if prune_count <= 0:
        return 0
    if prune_count >= numel:
        weight.zero_()
        return numel

    perm = torch.randperm(numel, generator=generator)
    idx = perm[:prune_count].to(device=weight.device)
    weight.reshape(-1)[idx] = 0
    return prune_count


def _apply_random_module_counts(modules, counts, generator):
    pruned = 0
    total = 0
    for (layer_idx, name, module), prune_count in zip(modules, counts):
        numel = module.weight.numel()
        selected = _random_zero_module(module, prune_count, generator)
        pruned += selected
        total += numel
        print(f"random pruned layer {layer_idx} {name}: {selected}/{numel}")
    return pruned, total


def _apply_global_random_pruning(model, sparsity_ratio, rng, generator):
    modules = list(_iter_prunable_modules(model))
    total = sum(module.weight.numel() for _, _, module in modules)
    counts = _module_prune_counts(modules, int(total * sparsity_ratio), rng)
    return _apply_random_module_counts(modules, counts, generator)


def _apply_local_random_pruning(model, sparsity_ratio, rng, generator):
    pruned = 0
    total = 0
    for layer_idx, layer in enumerate(model.model.layers):
        modules = [(layer_idx, name, module) for name, module in find_layers(layer).items()]
        layer_total = sum(module.weight.numel() for _, _, module in modules)
        counts = _module_prune_counts(modules, int(layer_total * sparsity_ratio), rng)
        layer_pruned, _ = _apply_random_module_counts(modules, counts, generator)
        pruned += layer_pruned
        total += layer_total
    return pruned, total


def _apply_per_op_random_pruning(model, sparsity_ratio, generator):
    pruned = 0
    total = 0
    for layer_idx, name, module in _iter_prunable_modules(model):
        numel = module.weight.numel()
        selected = _random_zero_module(module, int(numel * sparsity_ratio), generator)
        pruned += selected
        total += numel
        print(f"random pruned layer {layer_idx} {name}: {selected}/{numel}")
    return pruned, total


def prune_random(model, sparsity_ratio, score_order, seed):
    if sparsity_ratio <= 0:
        total = sum(module.weight.numel() for _, _, module in _iter_prunable_modules(model))
        return {"pruned": 0, "total": total, "actual_sparsity": 0.0}
    if sparsity_ratio > 1:
        raise ValueError(f"sparsity_ratio must be in [0, 1], got {sparsity_ratio}")

    score_orders = {"global", "local", "per_op"}
    if score_order not in score_orders:
        raise ValueError(f"Unsupported score_order: {score_order}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    if score_order == "global":
        pruned, total = _apply_global_random_pruning(model, sparsity_ratio, rng, generator)
    elif score_order == "local":
        pruned, total = _apply_local_random_pruning(model, sparsity_ratio, rng, generator)
    else:
        pruned, total = _apply_per_op_random_pruning(model, sparsity_ratio, generator)

    return {
        "pruned": pruned,
        "total": total,
        "actual_sparsity": 0.0 if total == 0 else pruned / total,
    }
