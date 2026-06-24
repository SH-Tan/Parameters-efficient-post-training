import json
from pathlib import Path

import numpy as np
import torch


class WrappedGPT:
    def __init__(self, layer, activation_chunk_size=2048):
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=layer.weight.device, dtype=torch.float32)
        self.nsamples = 0
        self.activation_chunk_size = int(activation_chunk_size)

    def add_batch(self, inp, _):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        inp = inp.reshape((-1, inp.shape[-1]))
        batch_size = inp.shape[0]
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        chunk_size = self.activation_chunk_size
        if chunk_size <= 0:
            chunk_size = inp.shape[0]
        for start in range(0, inp.shape[0], chunk_size):
            inp_chunk = inp[start:start + chunk_size].float()
            self.scaler_row += inp_chunk.pow(2).sum(dim=0) / self.nsamples
            del inp_chunk


def wanda_activation_norm(scaler_row):
    return scaler_row.detach().float().clamp_min(0).sqrt()


def save_wanda_activation_norm_stats(layer_idx, norm_by_name, output_dir, bins=256, metadata=None):
    output_dir = Path(output_dir) / "wanda_activation_norms"
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(norm_by_name)
    values = [norm_by_name[name].detach().float().cpu().numpy().reshape(-1) for name in names]
    merged = np.concatenate(values)
    max_value = float(merged.max(initial=0.0))
    bin_edges = np.linspace(0.0, max_value if max_value > 0 else 1.0, int(bins) + 1, dtype=np.float32)
    hist_counts = np.stack([np.histogram(value, bins=bin_edges)[0] for value in values]).astype(np.int64)

    quantile_probs = np.array([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0], dtype=np.float32)
    quantiles = np.stack([np.quantile(value, quantile_probs) for value in values]).astype(np.float32)
    summary_names = np.array(["mean", "std", "min", "max"])
    summary = np.array(
        [[value.mean(), value.std(), value.min(), value.max()] for value in values],
        dtype=np.float32,
    )

    npz_path = output_dir / f"layer_{layer_idx:03d}.npz"
    np.savez_compressed(
        npz_path,
        op_names=np.array(names),
        bin_edges=bin_edges,
        hist_counts=hist_counts,
        quantile_probs=quantile_probs,
        quantiles=quantiles,
        summary_names=summary_names,
        summary=summary,
        metadata=json.dumps(metadata or {}),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = np.diff(bin_edges)
    centers = bin_edges[:-1] + widths / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, counts in zip(names, hist_counts):
        total = counts.sum()
        density = counts / (total * widths) if total > 0 else counts
        ax.step(centers, density, where="mid", label=name)
    ax.set_title(f"WANDA activation norm distribution layer {layer_idx}")
    ax.set_xlabel("sqrt(mean(input^2)) per hidden feature")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()

    png_path = output_dir / f"layer_{layer_idx:03d}.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return npz_path, png_path


def layer_forward(model, layer, hidden_states, attention_mask, position_ids):
    squeeze_output = hidden_states.dim() == 2
    if squeeze_output:
        hidden_states = hidden_states.unsqueeze(0)

    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    if position_ids is not None and hasattr(model.model, "rotary_emb"):
        cos, sin = model.model.rotary_emb(hidden_states, position_ids)
        kwargs["position_embeddings"] = (cos, sin)
    output = layer(hidden_states, **kwargs)[0]
    return output.squeeze(0) if squeeze_output else output


def calibration_batch_tensor(tensor, start, end, batch_size, device):
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor.to(device)
    if tensor.shape[0] == 1 and batch_size > 1:
        return tensor.expand(batch_size, *tensor.shape[1:]).to(device)
    return tensor[start:end].to(device)
