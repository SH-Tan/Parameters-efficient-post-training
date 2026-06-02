import os
import pickle

import torch


def clone_pickleable_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: clone_pickleable_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_pickleable_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_pickleable_to_cpu(v) for v in value)
    return value


def save_layer_scores_pkl(layer_idx, scores, save_dir, method, metadata=None):
    if save_dir is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer_idx:03d}.pkl")
    payload = {
        "layer_idx": int(layer_idx),
        "method": method,
        "scores": {
            op_name: clone_pickleable_to_cpu(score)
            for op_name, score in scores.items()
        },
    }
    if metadata is not None:
        payload["metadata"] = clone_pickleable_to_cpu(metadata)

    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_path


def load_layer_scores_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    return (
        int(payload["layer_idx"]),
        payload.get("scores", {}),
        payload.get("metadata", None),
    )


def load_layer_scores(score_dir, layer_idx):
    pkl_path = os.path.join(score_dir, f"layer_{layer_idx:03d}.pkl")
    loaded_layer_idx, scores, _ = load_layer_scores_pkl(pkl_path)
    if loaded_layer_idx != layer_idx:
        raise ValueError(f"Expected layer {layer_idx} scores, found layer {loaded_layer_idx}")
    return scores
