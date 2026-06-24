import torch
from tqdm.auto import tqdm

from eval.data import get_loaders
from prune import as_device, filter_prune_ops, find_layers, prepare_calibration_input
from prune.wanda_utils import (
    WrappedGPT,
    calibration_batch_tensor,
    layer_forward,
    save_wanda_activation_norm_artifacts,
    wanda_activation_norm,
)
from utils.score_io_utils import save_layer_scores_pkl


def compute_wanda_scores(args, model, tokenizer, device=torch.device("cuda:0"), save_dir=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps = outs = attention_mask = position_ids = None
    try:
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
        need_scores = save_dir is not None or not (
            getattr(args, "skip_pp_eval", False) and not getattr(args, "do_downstream_eval", False)
        )
        wanda_scores = [] if save_dir is None and need_scores else None
        batch_size = max(1, int(getattr(args, "calib_forward_batch_size", 1)))

        for layer_idx, layer in enumerate(tqdm(layers, desc="WANDA score layers", unit="layer")):
            subset = filter_prune_ops(find_layers(layer), args.prune_ops)

            dev = as_device(device)
            if hasattr(model, "hf_device_map") and (f"model.layers.{layer_idx}" in model.hf_device_map):
                dev = as_device(model.hf_device_map[f"model.layers.{layer_idx}"])

            wrapped_layers = {
                name: WrappedGPT(
                    module,
                    activation_chunk_size=getattr(args, "wanda_activation_chunk_size", 2048),
                )
                for name, module in subset.items()
            }

            def add_batch(name):
                def hook(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return hook

            handles = [
                subset[name].register_forward_hook(add_batch(name))
                for name in wrapped_layers
            ]

            try:
                for batch_start in tqdm(range(0, args.nsamples, batch_size), desc=f"WANDA layer {layer_idx} calibration", unit="batch", leave=False):
                    batch_end = min(batch_start + batch_size, args.nsamples)
                    current_batch_size = batch_end - batch_start
                    with torch.no_grad():
                        batch_input = inps[batch_start:batch_end].to(dev)
                        batch_attention_mask = calibration_batch_tensor(
                            attention_mask,
                            batch_start,
                            batch_end,
                            current_batch_size,
                            dev,
                        )
                        batch_position_ids = calibration_batch_tensor(
                            position_ids,
                            batch_start,
                            batch_end,
                            current_batch_size,
                            dev,
                        )
                        batch_output = layer_forward(
                            model,
                            layer,
                            batch_input,
                            batch_attention_mask,
                            batch_position_ids,
                        )
                        batch_output_cpu = batch_output.detach().cpu()
                        del batch_output
                        outs[batch_start:batch_end].copy_(batch_output_cpu)
                        del batch_input, batch_attention_mask, batch_position_ids, batch_output_cpu
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            finally:
                for handle in handles:
                    handle.remove()

            activation_norms = {}
            layer_scores = {} if need_scores else None
            for name, module in subset.items():
                scaler_cpu = wrapped_layers[name].scaler_row.detach().float().cpu()
                activation_norm = wanda_activation_norm(scaler_cpu)
                if getattr(args, "wanda_save_activation_stats", False):
                    activation_norms[name] = activation_norm
                if need_scores:
                    print(f"collecting WANDA scores layer {layer_idx} name {name}")
                    weight_cpu = module.weight.detach().float().cpu()
                    layer_scores[name] = weight_cpu.abs() * activation_norm.reshape((1, -1))
                    del weight_cpu
                del scaler_cpu, activation_norm

            if getattr(args, "wanda_save_activation_stats", False):
                stats_dir = getattr(args, "wanda_activation_stats_dir", None) or save_dir or args.save
                input_norm_path, hist_plot_path = save_wanda_activation_norm_artifacts(
                    layer_idx,
                    activation_norms,
                    stats_dir,
                    bins=getattr(args, "wanda_activation_stats_bins", 256),
                    metadata={
                        "calib_data": args.calib_data,
                        "nsamples": args.nsamples,
                        "seqlen": model.seqlen,
                        "score_layout": "input_hidden",
                    },
                )
                print(
                    f"saved WANDA input norm artifacts for layer {layer_idx} to "
                    f"{input_norm_path} and {hist_plot_path}"
                )

            if need_scores:
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
            elif layer_scores is not None:
                del layer_scores
            inps, outs = outs, inps
            del wrapped_layers, handles, activation_norms
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return wanda_scores
    finally:
        model.config.use_cache = use_cache
        del inps, outs, attention_mask, position_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
