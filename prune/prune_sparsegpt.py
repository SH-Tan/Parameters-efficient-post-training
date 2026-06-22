import torch
from tqdm.auto import tqdm

from eval.data import get_loaders
from prune import as_device, filter_prune_ops, find_layers, prepare_calibration_input
from prune.sparsegpt_utils import SparseGPT
from prune.wanda_utils import calibration_batch_tensor, layer_forward
from utils.score_io_utils import save_layer_scores_pkl


def compute_sparsegpt_scores(args, model, tokenizer, device=torch.device("cuda:0"), save_dir=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps = outs = attention_mask = position_ids = None
    try:
        print(f"loading calibration data for SparseGPT scores with seqlen={model.seqlen}")
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
        sparsegpt_scores = [] if save_dir is None else None
        batch_size = max(1, int(getattr(args, "calib_forward_batch_size", 1)))

        for layer_idx, layer in enumerate(tqdm(layers, desc="SparseGPT score layers", unit="layer")):
            subset = filter_prune_ops(find_layers(layer), args.prune_ops)

            dev = as_device(device)
            if hasattr(model, "hf_device_map") and (f"model.layers.{layer_idx}" in model.hf_device_map):
                dev = as_device(model.hf_device_map[f"model.layers.{layer_idx}"])

            sparsegpt_layers = {
                name: SparseGPT(
                    module,
                    hessian_chunk_size=getattr(args, "sparsegpt_hessian_chunk_size", 8192),
                )
                for name, module in subset.items()
            }

            def add_batch(name):
                def hook(_, inp, out):
                    sparsegpt_layers[name].add_batch(inp[0].data, out.data)
                return hook

            handles = [
                subset[name].register_forward_hook(add_batch(name))
                for name in sparsegpt_layers
            ]

            try:
                for batch_start in tqdm(range(0, args.nsamples, batch_size), desc=f"SparseGPT layer {layer_idx} calibration", unit="batch", leave=False):
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
                        outs[batch_start:batch_end].copy_(batch_output.detach().cpu())
                        del batch_input, batch_attention_mask, batch_position_ids, batch_output
            finally:
                for handle in handles:
                    handle.remove()

            layer_scores = {}
            for name in subset:
                print(f"collecting SparseGPT scores layer {layer_idx} name {name}")
                layer_scores[name] = sparsegpt_layers[name].score(
                    percdamp=getattr(args, "sparsegpt_percdamp", 0.01),
                )
                sparsegpt_layers[name].free()

            save_path = save_layer_scores_pkl(
                layer_idx,
                layer_scores,
                save_dir,
                method="sparsegpt",
                metadata={
                    "score_layout": "weight_out_in",
                    "calib_data": args.calib_data,
                    "nsamples": args.nsamples,
                    "seqlen": model.seqlen,
                    "percdamp": getattr(args, "sparsegpt_percdamp", 0.01),
                },
            )
            if save_path is not None:
                print(f"saved layer {layer_idx} SparseGPT scores to {save_path}")

            if sparsegpt_scores is not None:
                sparsegpt_scores.append(layer_scores)
            else:
                del layer_scores
            inps, outs = outs, inps
            del sparsegpt_layers, handles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return sparsegpt_scores
    finally:
        model.config.use_cache = use_cache
        del inps, outs, attention_mask, position_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
