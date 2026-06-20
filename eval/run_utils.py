import gc
import json
import os
import subprocess
import sys

import torch
from tqdm.auto import tqdm

from downstream_eval.model_accuracy_test import DownstreamEvalConfig, evaluate_downstream_task_accuracy, load_examples
from eval.eval import eval_ppl_with_loader, load_ppl_eval_data
from eval.result_utils import (
    append_downstream_result_csv,
    append_result_csv,
    draw_accuracy_vs_sparsity,
    draw_pp_vs_sparsity,
    result_dir,
    safe_result_component,
)
from prune import check_sparsity, normalize_prune_ops
from prune.prune_magnitude import compute_magnitude_scores
from prune.prune_random import prune_random
from prune.prune_sparsegpt import compute_sparsegpt_scores
from prune.prune_wanda import compute_wanda_scores
from prune.score_prune_utils import prune_by_scores
from utils.model_utils import get_llm, get_tokenizer, resolve_model_device


def log_stage(message):
    print(f"\n[stage] {message}", flush=True)


def score_save_dir(args):
    path = os.path.join(
        args.save,
        safe_result_component(args.calib_data),
        f"seq_len_{args.seqlen}",
    )
    os.makedirs(path, exist_ok=True)
    return path


def resolve_sparsity_ratios(args):
    ratios = []
    seen = set()
    for ratio in args.sparsity_ratio:
        ratio = float(ratio)
        if ratio < 0 or ratio > 1:
            raise ValueError(f"sparsity_ratio must be in [0, 1], got {ratio}")
        if ratio not in seen:
            ratios.append(ratio)
            seen.add(ratio)
    return sorted(ratios) if ratios else [0.0]


def resolve_score_orders(args):
    return list(dict.fromkeys(args.score_order))


def compute_scores(args, model, tokenizer, model_device, score_dir):
    target = "memory" if score_dir is None else score_dir
    log_stage(f"Computing {args.prune_method} scores into {target}")
    if args.prune_method == "wanda":
        scores = compute_wanda_scores(args, model, tokenizer, model_device, save_dir=score_dir)
    elif args.prune_method == "magnitude":
        scores = compute_magnitude_scores(model, save_dir=score_dir, prune_ops=args.prune_ops)
    elif args.prune_method == "sparsegpt":
        scores = compute_sparsegpt_scores(args, model, tokenizer, model_device, save_dir=score_dir)
    else:
        raise ValueError(f"Unsupported prune_method: {args.prune_method}")
    log_stage(f"Finished computing {args.prune_method} scores")
    return score_dir if score_dir is not None else scores


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _format_sparsity_for_path(value):
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def pruned_model_dir(args, out_dir, score_order, target_sparsity):
    root = args.pruned_model_root or os.path.join(out_dir, "pruned_models")
    return os.path.join(
        root,
        f"{args.prune_method}_{score_order}_sparsity_{_format_sparsity_for_path(target_sparsity)}",
    )


def save_pruned_checkpoint_if_needed(args, model, tokenizer, out_dir, score_order, target_sparsity):
    if not (args.save_pruned_model or args.downstream_backend == "vllm"):
        return None
    if args.downstream_backend == "vllm" and args.do_downstream_eval and not args.skip_pp_eval:
        raise ValueError("downstream_backend='vllm' cannot share the live Transformers model with PP eval in one pass; set run_pp_eval=0 or use downstream_backend=transformers")
    path = pruned_model_dir(args, out_dir, score_order, target_sparsity)
    os.makedirs(path, exist_ok=True)
    log_stage(f"Saving pruned model checkpoint for downstream eval: {path}")
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    return path


def run_vllm_downstream_eval_subprocess(args, pruned_model_path, response_log_path, metrics_path):
    if pruned_model_path is None:
        raise ValueError("vLLM downstream eval requires a saved pruned model checkpoint")
    cmd = [
        sys.executable,
        "-m",
        "downstream_eval.vllm_accuracy_runner",
        "--model_path",
        str(pruned_model_path),
        "--dataset_path",
        str(args.downstream_task_data),
        "--output_path",
        str(response_log_path),
        "--metrics_path",
        str(metrics_path),
        "--prompt_key",
        str(args.downstream_prompt_key),
        "--start_index",
        str(args.downstream_start_index),
        "--max_examples",
        str(args.downstream_max_examples),
        "--seed",
        str(args.seed),
        "--max_prompt_length",
        str(args.downstream_max_prompt_length),
        "--max_new_tokens",
        str(args.downstream_max_new_tokens),
        "--batch_size",
        str(args.downstream_batch_size),
        "--generation_max_batch_tokens",
        str(args.downstream_generation_max_batch_tokens),
        "--temperature",
        str(args.downstream_temperature),
        "--top_p",
        str(args.downstream_top_p),
        "--top_k",
        str(args.downstream_top_k),
        "--response_log_max",
        str(args.downstream_response_log_max),
        "--tensor_parallel_size",
        str(args.vllm_tensor_parallel_size),
        "--gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--dtype",
        str(args.vllm_dtype),
    ]
    if args.downstream_response_key:
        cmd.extend(["--response_key", str(args.downstream_response_key)])
    if args.downstream_reward_score_dir:
        cmd.extend(["--reward_score_dir", str(args.downstream_reward_score_dir)])
    if args.downstream_shuffle:
        cmd.append("--shuffle")
    log_stage("Starting vLLM subprocess for downstream generation")
    print("vLLM command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(metrics_path, encoding="utf-8") as handle:
        return json.load(handle)


def run_downstream_eval(args, model, tokenizer, model_device, out_dir, score_order, target_sparsity, actual_sparsity, examples, pruned_model_path=None):
    response_log_path = os.path.join(
        out_dir,
        f"downstream_task_responses_{score_order}_sparsity_{target_sparsity:.6f}.jsonl",
    )
    log_stage(
        "Starting downstream eval "
        f"score_order={score_order} sparsity={target_sparsity:.4f} "
        f"examples={args.downstream_max_examples} "
        f"batch_size={args.downstream_batch_size} "
        f"generation_max_batch_tokens={args.downstream_generation_max_batch_tokens} "
        f"use_cache={args.downstream_use_cache} "
        f"max_new_tokens={args.downstream_max_new_tokens} "
        f"max_prompt_length={args.downstream_max_prompt_length} "
        f"response_log_max={args.downstream_response_log_max} "
        f"backend={args.downstream_backend} "
        f"response_log={response_log_path}"
    )
    config = DownstreamEvalConfig(
        device=str(model_device),
        max_prompt_length=args.downstream_max_prompt_length,
        max_new_tokens=args.downstream_max_new_tokens,
        batch_size=args.downstream_batch_size,
        generation_max_batch_tokens=args.downstream_generation_max_batch_tokens,
        use_cache=args.downstream_use_cache,
        temperature=args.downstream_temperature,
        top_p=args.downstream_top_p,
        top_k=args.downstream_top_k,
        response_log_max=args.downstream_response_log_max,
        backend=args.downstream_backend,
        model_path=pruned_model_path,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        dtype=args.vllm_dtype,
    )
    if args.downstream_backend == "vllm":
        metrics_path = os.path.join(
            out_dir,
            f"downstream_task_metrics_{score_order}_sparsity_{target_sparsity:.6f}.json",
        )
        metrics = run_vllm_downstream_eval_subprocess(
            args,
            pruned_model_path,
            response_log_path,
            metrics_path,
        )
    else:
        metrics = evaluate_downstream_task_accuracy(
            model,
            tokenizer,
            args.downstream_task_data,
            examples=examples,
            prompt_key=args.downstream_prompt_key,
            response_key=args.downstream_response_key,
            start_index=args.downstream_start_index,
            max_examples=args.downstream_max_examples,
            shuffle=args.downstream_shuffle,
            seed=args.seed,
            config=config,
            output_path=response_log_path,
            reward_score_dir=args.downstream_reward_score_dir,
        )
    csv_path = os.path.join(out_dir, "downstream_task_results.csv")
    append_downstream_result_csv(
        csv_path,
        {
            "seed": args.seed,
            "method": args.prune_method,
            "score_order": score_order,
            "target_sparsity": f"{target_sparsity:.6f}",
            "actual_sparsity": f"{actual_sparsity:.6f}",
            "task_data": args.downstream_task_data,
            "backend": args.downstream_backend,
            "pruned_model_path": pruned_model_path or "",
            "num_examples": metrics.get("num_examples", ""),
            "num_scored": metrics.get("num_scored", ""),
            "num_unscored": metrics.get("num_unscored", ""),
            "accuracy": metrics.get("accuracy", ""),
            "pass@1": metrics.get("pass@1", ""),
            "mean_score": metrics.get("mean_score", ""),
            "score_sum": metrics.get("score_sum", ""),
            "num_correct": metrics.get("num_correct", ""),
        },
    )
    print(f"downstream metrics score_order={score_order} sparsity={target_sparsity:.4f}: {metrics}")


def run_model_evals(
    args,
    model,
    tokenizer,
    model_device,
    out_dir,
    result_csv_path,
    eval_loader,
    eval_seq_lens,
    score_order,
    target_sparsity,
    actual_sparsity,
    downstream_examples,
    pruned_model_path=None,
):
    if not args.skip_pp_eval:
        log_stage(f"Running perplexity eval for {len(eval_seq_lens)} sequence length(s)")
        for seq_len in tqdm(eval_seq_lens, desc="PPL eval", unit="seq_len"):
            model.seqlen = int(seq_len)
            ppl_test = eval_ppl_with_loader(model, eval_loader, model_device)
            print(
                f"{args.pp_eval_data} perplexity {ppl_test} using score_order={score_order}, "
                f"target_sparsity={target_sparsity:.4f}, pp_seqlen={seq_len}"
            )
            append_result_csv(
                result_csv_path,
                {
                    "seed": args.seed,
                    "method": args.prune_method,
                    "score_order": score_order,
                    "target_sparsity": f"{target_sparsity:.6f}",
                    "actual_sparsity": f"{actual_sparsity:.6f}",
                    "pp_eval_data": args.pp_eval_data,
                    "pp_seq_len": int(seq_len),
                    "ppl_test": f"{ppl_test:.6f}",
                },
            )

    if args.do_downstream_eval:
        eval_model = model
        if args.downstream_backend == "vllm":
            if pruned_model_path is None:
                raise ValueError("vLLM downstream eval requires a saved pruned model checkpoint")
            if args.skip_pp_eval:
                eval_model = None
        run_downstream_eval(
            args,
            eval_model,
            tokenizer,
            model_device,
            out_dir,
            score_order,
            target_sparsity,
            actual_sparsity,
            downstream_examples,
            pruned_model_path=pruned_model_path,
        )


def run_score_eval(args, score_dir):
    model_device = resolve_model_device(args.model_device)
    log_stage(f"Using {model_device} model device")

    args.prune_ops = normalize_prune_ops(args.prune_ops)
    print(f"Prune ops: {'all' if args.prune_ops is None else ' '.join(args.prune_ops)}", flush=True)
    sparsity_ratios = resolve_sparsity_ratios(args)
    score_orders = resolve_score_orders(args)
    total_points = len(score_orders) * len(sparsity_ratios)
    log_stage(
        f"Prepared sweep: method={args.prune_method}, score_orders={score_orders}, "
        f"sparsity_ratios={sparsity_ratios}, eval_points={total_points}"
    )
    tokenizer = get_tokenizer(args.model, args.cache_dir)

    score_source = None
    reusable_model = None
    if args.prune_method == "random":
        print("Skipping score computation for random pruning.")
    else:
        model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
        model.eval()
        score_source = compute_scores(args, model, tokenizer, model_device, score_dir)
        reusable_model = model

    if args.skip_pp_eval and not args.do_downstream_eval:
        print("Skipping PP eval and downstream eval.")
        if reusable_model is not None:
            del reusable_model
        del score_source
        del tokenizer
        cleanup_cuda()
        return

    eval_seq_lens = args.pp_seqlen if len(args.pp_seqlen) >= 1 else [args.seqlen]
    out_dir = result_dir(args)
    result_csv_path = os.path.join(out_dir, "pp_eval_results.csv")
    plot_path = os.path.join(out_dir, "pp_vs_sparsity.png")
    downstream_csv_path = os.path.join(out_dir, "downstream_task_results.csv")
    accuracy_plot_path = os.path.join(out_dir, "accuracy_vs_sparsity.png")
    eval_loader = None
    downstream_examples = None

    if args.skip_pp_eval:
        print("Skipping PP eval.")
    else:
        log_stage(f"Loading {args.pp_eval_data} perplexity eval data")
        eval_loader = load_ppl_eval_data(
            args.pp_eval_data,
            tokenizer,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
        )

    if args.do_downstream_eval:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        log_stage(f"Loading {args.downstream_task_data} downstream eval data")
        downstream_examples = load_examples(
            args.downstream_task_data,
            tokenizer,
            prompt_key=args.downstream_prompt_key,
            response_key=args.downstream_response_key,
            start_index=args.downstream_start_index,
            max_examples=args.downstream_max_examples,
            shuffle=args.downstream_shuffle,
            seed=args.seed,
        )


    progress = tqdm(total=total_points, desc=f"{args.prune_method} sweep", unit="eval")
    try:
        for score_order in score_orders:
            log_stage(f"Starting score_order={score_order}")
            if args.prune_method == "random":
                for target_sparsity in sparsity_ratios:
                    current_model_device = resolve_model_device(args.model_device)
                    current_model = get_llm(args.model, args.cache_dir, current_model_device, args.seqlen)
                    try:
                        current_model.eval()
                        progress.set_postfix(score_order=score_order, sparsity=f"{target_sparsity:.4f}")
                        log_stage(
                            f"Eval point: method={args.prune_method}, score_order={score_order}, "
                            f"target_sparsity={target_sparsity:.4f}"
                        )
                        if target_sparsity > 0:
                            log_stage(
                                f"Applying random pruning: score_order={score_order}, "
                                f"target_sparsity={target_sparsity:.4f}"
                            )
                            summary = prune_random(current_model, target_sparsity, score_order, args.seed, args.prune_ops)
                            print(
                                f"{args.prune_method} prune summary: {summary['pruned']}/{summary['total']} "
                                f"({summary['actual_sparsity']:.4f})",
                                flush=True,
                            )

                        log_stage("Checking actual sparsity")
                        actual_sparsity = check_sparsity(current_model, args.prune_ops)
                        pruned_model_path = save_pruned_checkpoint_if_needed(
                            args, current_model, tokenizer, out_dir, score_order, target_sparsity
                        )
                        eval_model = current_model
                        if args.downstream_backend == "vllm" and args.skip_pp_eval:
                            eval_model = None
                            del current_model
                            current_model = None
                            cleanup_cuda()
                        run_model_evals(
                            args,
                            eval_model,
                            tokenizer,
                            current_model_device,
                            out_dir,
                            result_csv_path,
                            eval_loader,
                            eval_seq_lens,
                            score_order,
                            target_sparsity,
                            actual_sparsity,
                            downstream_examples,
                            pruned_model_path=pruned_model_path,
                        )
                    finally:
                        if current_model is not None:
                            del current_model
                        cleanup_cuda()
                    progress.update(1)
                continue

            current_model_device = resolve_model_device(args.model_device)
            current_model = None

            try:
                for target_sparsity in sparsity_ratios:
                    if current_model is None:
                        log_stage(f"Loading model for score_order={score_order}")
                        if reusable_model is None:
                            current_model = get_llm(args.model, args.cache_dir, current_model_device, args.seqlen)
                        else:
                            current_model = reusable_model
                            reusable_model = None
                        current_model.eval()
                    progress.set_postfix(score_order=score_order, sparsity=f"{target_sparsity:.4f}")
                    log_stage(
                        f"Eval point: method={args.prune_method}, score_order={score_order}, "
                        f"target_sparsity={target_sparsity:.4f}"
                    )
                    if target_sparsity > 0:
                        log_stage(
                            f"Applying score pruning from {'memory' if score_dir is None else score_dir}: "
                            f"score_order={score_order}, target_sparsity={target_sparsity:.4f}"
                        )
                        summary = prune_by_scores(current_model, score_source, target_sparsity, score_order, args.prune_ops)
                        print(
                            f"{args.prune_method} prune summary: {summary['pruned']}/{summary['total']} "
                            f"({summary['actual_sparsity']:.4f})",
                            flush=True,
                        )

                    log_stage("Checking actual sparsity")
                    actual_sparsity = check_sparsity(current_model, args.prune_ops)
                    pruned_model_path = save_pruned_checkpoint_if_needed(
                        args, current_model, tokenizer, out_dir, score_order, target_sparsity
                    )
                    eval_model = current_model
                    if args.downstream_backend == "vllm" and args.skip_pp_eval:
                        eval_model = None
                        del current_model
                        current_model = None
                        cleanup_cuda()
                    run_model_evals(
                        args,
                        eval_model,
                        tokenizer,
                        current_model_device,
                        out_dir,
                        result_csv_path,
                        eval_loader,
                        eval_seq_lens,
                        score_order,
                        target_sparsity,
                        actual_sparsity,
                        downstream_examples,
                        pruned_model_path=pruned_model_path,
                    )
                    progress.update(1)

            finally:
                if current_model is not None:
                    del current_model
                cleanup_cuda()
    finally:
        progress.close()

    if not args.skip_pp_eval:
        drawn_path = draw_pp_vs_sparsity(result_csv_path, plot_path)
        if drawn_path is not None:
            print(f"Saved PP vs sparsity plot: {drawn_path}")
    if args.do_downstream_eval:
        drawn_path = draw_accuracy_vs_sparsity(downstream_csv_path, accuracy_plot_path)
        if drawn_path is not None:
            print(f"Saved accuracy vs sparsity plot: {drawn_path}")
    if eval_loader is not None:
        del eval_loader
    if downstream_examples is not None:
        del downstream_examples
    del score_source
    del tokenizer
    cleanup_cuda()
