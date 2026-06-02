import torch

from eval.data import get_loaders
from prune import as_device, find_layers, prepare_calibration_input
from prune.wanda_utils import WrappedGPT, layer_forward
from utils.score_io_utils import save_layer_scores_pkl


def compute_wanda_scores(args, model, tokenizer, device=torch.device("cuda:0"), save_dir=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print(f"loading calibration data for WANDA scores with seqlen={model.seqlen}")
    dataloader, _ = get_loaders(
        args.calib_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    del dataloader

    layers = model.model.layers
    wanda_scores = [] if save_dir is None else None

    for layer_idx, layer in enumerate(layers):
        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and (f"model.layers.{layer_idx}" in model.hf_device_map):
            dev = as_device(model.hf_device_map[f"model.layers.{layer_idx}"])
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        wrapped_layers = {name: WrappedGPT(module) for name, module in subset.items()}

        def add_batch(name):
            def hook(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return hook

        handles = [
            subset[name].register_forward_hook(add_batch(name))
            for name in wrapped_layers
        ]

        for sample_idx in range(args.nsamples):
            with torch.no_grad():
                outs[sample_idx] = layer_forward(
                    model,
                    layer,
                    inps[sample_idx],
                    attention_mask,
                    position_ids,
                )

        for handle in handles:
            handle.remove()

        layer_scores = {}
        for name, module in subset.items():
            print(f"collecting WANDA scores layer {layer_idx} name {name}")
            weight_cpu = module.weight.detach().cpu()
            scaler_cpu = wrapped_layers[name].scaler_row.detach().cpu()
            layer_scores[name] = torch.abs(weight_cpu) * torch.sqrt(scaler_cpu.reshape((1, -1)))
            del weight_cpu, scaler_cpu

        save_path = save_layer_scores_pkl(
            layer_idx,
            layer_scores,
            save_dir,
            method="wanda",
            metadata={
                "score_layout": "weight_out_in",
                "calib_data": args.calib_data,
                "nsamples": args.nsamples,
                "seqlen": model.seqlen,
            },
        )
        if save_path is not None:
            print(f"saved layer {layer_idx} WANDA scores to {save_path}")

        if wanda_scores is not None:
            wanda_scores.append(layer_scores)
        else:
            del layer_scores
        inps, outs = outs, inps
        del wrapped_layers, handles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    del inps, outs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wanda_scores
