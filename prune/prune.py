import torch
import torch.nn as nn

PRUNABLE_OP_ALIASES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
    "down": "mlp.down_proj",
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "up_proj": "mlp.up_proj",
    "gate_proj": "mlp.gate_proj",
    "down_proj": "mlp.down_proj",
}
PRUNABLE_OPS = tuple(dict.fromkeys(PRUNABLE_OP_ALIASES.values()))


def as_device(device):
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device) if isinstance(device, str) else device


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def normalize_prune_ops(prune_ops):
    if prune_ops is None:
        return None
    normalized = []
    for raw_op in prune_ops:
        for op in raw_op.split(","):
            op = op.strip()
            if not op:
                continue
            if op not in PRUNABLE_OP_ALIASES:
                choices = ", ".join(sorted(PRUNABLE_OP_ALIASES))
                raise ValueError(f"Unsupported prune op: {op}. Choices: {choices}")
            canonical_op = PRUNABLE_OP_ALIASES[op]
            if canonical_op not in normalized:
                normalized.append(canonical_op)
    return tuple(normalized) if normalized else None


def filter_prune_ops(subset, prune_ops):
    if prune_ops is None:
        return subset
    prune_ops = set(prune_ops)
    return {name: module for name, module in subset.items() if name in prune_ops}


def check_sparsity(model, prune_ops=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = filter_prune_ops(find_layers(layer), prune_ops)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            zero_count = int((W == 0).sum().item())
            param_count = W.numel()
            count += zero_count
            total_params += param_count

            sub_count += zero_count
            sub_params += param_count

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # ===== device handling =====
    if isinstance(device, str):
        device = torch.device(device)

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
        device = torch.device(f"cuda:{dev}") if isinstance(dev, int) else dev

    # Keep calibration activations on CPU. WANDA/SparseGPT move one sample at a
    # time to the active layer device to avoid holding model + full activation
    # buffers on GPU.
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device="cpu"
    )

    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None
    }

    class CatchInput(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            i = cache["i"]
            # ✅ stop if enough samples collected
            if i >= nsamples:
                raise StopIteration

            if inp.dim() == 3:
                if inp.shape[0] != 1:
                    raise ValueError(
                        f"Expected calibration batch size 1, got input shape {tuple(inp.shape)}"
                    )
                inp = inp.squeeze(0)
            elif inp.dim() != 2:
                raise ValueError(f"Unexpected calibration input shape {tuple(inp.shape)}")

            inps[i].copy_(inp.detach().cpu())
            cache["i"] += 1

            # only save once (they're usually same shape)
            if cache["attention_mask"] is None:
                attention_mask = kwargs.get("attention_mask", None)
                cache["attention_mask"] = None if attention_mask is None else attention_mask.detach().cpu()

            if cache["position_ids"] is None:
                position_ids = kwargs.get("position_ids", None)
                cache["position_ids"] = None if position_ids is None else position_ids.detach().cpu()

            raise CatchInput

    try:
        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(device))
            except CatchInput:
                pass
            except StopIteration:
                break
    finally:
        if isinstance(layers[0], Catcher):
            layers[0] = layers[0].module
        model.config.use_cache = use_cache

    outs = torch.zeros_like(inps)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    return inps, outs, attention_mask, position_ids
