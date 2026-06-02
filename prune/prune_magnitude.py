import torch

from prune import find_layers
from utils.score_io_utils import save_layer_scores_pkl


def compute_magnitude_scores(model, save_dir=None):
    layers = model.model.layers
    all_scores = [] if save_dir is None else None

    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer)
        layer_scores = {}
        for name, module in subset.items():
            print(f"collecting magnitude scores layer {layer_idx} name {name}")
            layer_scores[name] = module.weight.detach().abs().cpu()

        save_path = save_layer_scores_pkl(
            layer_idx,
            layer_scores,
            save_dir,
            method="magnitude",
            metadata={"score_layout": "weight_out_in"},
        )
        if save_path is not None:
            print(f"saved layer {layer_idx} magnitude scores to {save_path}")
        if all_scores is not None:
            all_scores.append(layer_scores)
        else:
            del layer_scores

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_scores
