import csv
import os
import pickle

import torch


def layer_paths(score_dir):
    names = [
        name for name in os.listdir(score_dir)
        if name.startswith("layer_") and name.endswith(".pkl")
    ]
    return [os.path.join(score_dir, name) for name in sorted(names)]


def load_score_payload(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def iter_score_tensors(score_dir):
    for path in layer_paths(score_dir):
        payload = load_score_payload(path)
        layer_idx = int(payload["layer_idx"])
        method = payload.get("method", "")
        metadata = payload.get("metadata", {})
        for op_name, score in payload.get("scores", {}).items():
            yield layer_idx, op_name, method, metadata, score.detach().cpu()


def score_count(score_dir):
    total = 0
    for _, _, _, _, score in iter_score_tensors(score_dir):
        total += score.numel()
    return total


def _lowest_mask(score, keep_ratio):
    keep_count = int(score.numel() * keep_ratio)
    if keep_count <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    if keep_count >= score.numel():
        return torch.ones_like(score, dtype=torch.bool)

    flat = score.reshape(-1).float()
    threshold = torch.kthvalue(flat, keep_count).values
    mask = flat <= threshold
    extra = int(mask.sum().item()) - keep_count
    if extra > 0:
        tied = (flat == threshold).nonzero(as_tuple=False).flatten()
        mask[tied[:extra]] = False
    return mask.reshape(score.shape)


def _highest_mask(score, keep_ratio):
    return _lowest_mask(-score.float(), keep_ratio)


def _select_mask(score, ratio, side):
    if side == "low":
        return _lowest_mask(score, ratio)
    if side == "high":
        return _highest_mask(score, ratio)
    raise ValueError(f"Unsupported side: {side}")


def build_selection(score_dir, ratio=0.5, side="low", order="per_op"):
    if order == "global":
        flat_scores = [
            score.reshape(-1).float()
            for _, _, _, _, score in iter_score_tensors(score_dir)
        ]
        if not flat_scores:
            raise ValueError(f"No score tensors found in {score_dir}")
        flat_mask = _select_mask(torch.cat(flat_scores), ratio, side)
        selection = {}
        offset = 0
        for layer_idx, op_name, _, _, score in iter_score_tensors(score_dir):
            next_offset = offset + score.numel()
            selection[(layer_idx, op_name)] = flat_mask[offset:next_offset].reshape(score.shape)
            offset = next_offset
        return selection

    if order == "local":
        selection = {}
        by_layer = {}
        for layer_idx, op_name, _, _, score in iter_score_tensors(score_dir):
            by_layer.setdefault(layer_idx, []).append((op_name, score))
        for layer_idx, items in by_layer.items():
            flat_scores = [score.reshape(-1).float() for _, score in items]
            flat_mask = _select_mask(torch.cat(flat_scores), ratio, side)
            offset = 0
            for op_name, score in items:
                next_offset = offset + score.numel()
                selection[(layer_idx, op_name)] = flat_mask[offset:next_offset].reshape(score.shape)
                offset = next_offset
        return selection

    if order == "per_op":
        selection = {}
        for layer_idx, op_name, _, _, score in iter_score_tensors(score_dir):
            selection[(layer_idx, op_name)] = _select_mask(score, ratio, side)
        return selection

    raise ValueError(f"Unsupported order: {order}")


def save_selection(selection, output_path, score_dir, ratio, side, order):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "score_dir": score_dir,
        "ratio": float(ratio),
        "side": side,
        "order": order,
        "selection": {
            f"{layer_idx}:{op_name}": mask
            for (layer_idx, op_name), mask in selection.items()
        },
    }
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def summarize_selection(selection):
    rows = []
    total_selected = 0
    total_params = 0
    for (layer_idx, op_name), mask in sorted(selection.items()):
        selected = int(mask.sum().item())
        count = mask.numel()
        total_selected += selected
        total_params += count
        rows.append({
            "layer": layer_idx,
            "op": op_name,
            "selected": selected,
            "total": count,
            "ratio": 0.0 if count == 0 else selected / count,
        })
    return rows, total_selected, total_params


def write_summary_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "op", "selected", "total", "ratio"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def selection_flat_bool(selection):
    tensors = [mask.reshape(-1).bool() for _, mask in sorted(selection.items())]
    if not tensors:
        return torch.empty(0, dtype=torch.bool)
    return torch.cat(tensors)


def jaccard(mask_a, mask_b):
    if mask_a.numel() != mask_b.numel():
        raise ValueError(f"Mask size mismatch: {mask_a.numel()} vs {mask_b.numel()}")
    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    return 0.0 if union == 0 else intersection / union


def overlap(mask_a, mask_b):
    if mask_a.numel() != mask_b.numel():
        raise ValueError(f"Mask size mismatch: {mask_a.numel()} vs {mask_b.numel()}")
    selected = mask_a.sum().item()
    if selected == 0:
        return 0.0
    return torch.logical_and(mask_a, mask_b).sum().item() / selected
