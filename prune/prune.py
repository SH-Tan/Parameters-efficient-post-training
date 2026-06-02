import torch
import torch.nn as nn


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

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

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

    # ===== allocate =====
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device
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

            inps[i].copy_(inp)   # faster + safer than assignment
            cache["i"] += 1

            # only save once (they're usually same shape)
            if cache["attention_mask"] is None:
                cache["attention_mask"] = kwargs.get("attention_mask", None)

            if cache["position_ids"] is None:
                cache["position_ids"] = kwargs.get("position_ids", None)

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
