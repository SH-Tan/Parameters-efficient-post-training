import gc
import os

import torch

from downstream_eval.model_accuracy_test import DownstreamEvalConfig, evaluate_downstream_task_accuracy, load_examples
from eval.eval import eval_ppl_with_loader, load_ppl_eval_data
from eval.result_utils import (
    append_downstream_result_csv,
    append_result_csv,
    draw_accuracy_vs_sparsity,
    draw_pp_vs_sparsity,
    result_dir,
)
from prune import check_sparsity, normalize_prune_ops
from prune.prune_magnitude import compute_magnitude_scores
from prune.prune_random import prune_random
from prune.prune_sparsegpt import compute_sparsegpt_scores
from prune.prune_wanda import compute_wanda_scores
from prune.score_prune_utils import prune_by_scores
from utils.model_utils import get_llm, get_tokenizer, resolve_model_device


def score_save_dir(args):
    path = os.path.join(
        args.save,
        args.calib_data,
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
    return ratios or [0.0]


def resolve_score_orders(args):
    return list(dict.fromkeys(args.score_order))


def compute_scores(args, model, tokenizer, model_device, score_dir):
    target = "memory" if score_dir is None else score_dir
    print(f"Computing {args.prune_method} scores into {target}")
    if args.prune_method == "wanda":
        scores = compute_wanda_scores(args, model, tokenizer, model_device, save_dir=score_dir)
    elif args.prune_method == "magnitude":
        scores = compute_magnitude_scores(model, save_dir=score_dir, prune_ops=args.prune_ops)
    elif args.prune_method == "sparsegpt":
        scores = compute_sparsegpt_scores(args, model, tokenizer, model_device, save_dir=score_dir)
    else:
        raise ValueError(f"Unsupported prune_method: {args.prune_method}")
    print(f"Finished computing {args.prune_method} scores")
    return score_dir if score_dir is not None else scores


def run_downstream_eval(args, model, tokenizer, model_device, out_dir, score_order, target_sparsity, actual_sparsity, examples):
    response_log_path = os.path.join(
        out_dir,
        f"downstream_task_responses_{score_order}_sparsity_{target_sparsity:.6f}.jsonl",
    )
    print(
        "starting downstream eval "
        f"score_order={score_order} sparsity={target_sparsity:.4f} "
        f"examples={args.downstream_max_examples} "
        f"batch_size={args.downstream_batch_size} "
        f"max_new_tokens={args.downstream_max_new_tokens} "
        f"max_prompt_length={args.downstream_max_prompt_length} "
        f"response_log_max={args.downstream_response_log_max} "
        f"response_log={response_log_path}"
    )
    config = DownstreamEvalConfig(
        device=str(model_device),
        max_prompt_length=args.downstream_max_prompt_length,
        max_new_tokens=args.downstream_max_new_tokens,
        batch_size=args.downstream_batch_size,
        temperature=args.downstream_temperature,
        top_p=args.downstream_top_p,
        top_k=args.downstream_top_k,
        response_log_max=args.downstream_response_log_max,
    )
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


def run_score_eval(args, score_dir):
    model_device = resolve_model_device(args.model_device)
    print(f"Using {model_device} model device")

    args.prune_ops = normalize_prune_ops(args.prune_ops)
    print(f"Prune ops: {'all' if args.prune_ops is None else ' '.join(args.prune_ops)}")
    sparsity_ratios = resolve_sparsity_ratios(args)
    score_orders = resolve_score_orders(args)
    tokenizer = get_tokenizer(args.model, args.cache_dir)

    score_source = None
    if args.prune_method == "random":
        print("Skipping score computation for random pruning.")
    else:
        model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
        try:
            model.eval()
            score_source = compute_scores(args, model, tokenizer, model_device, score_dir)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.skip_pp_eval and not args.do_downstream_eval:
        print("Skipping PP eval and downstream eval.")
        del score_source
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        print(f"loading {args.pp_eval_data} perplexity eval data")
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
        print(f"loading {args.downstream_task_data} downstream eval data")
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

    dense_eval_cache = None

    for score_order in score_orders:
        for target_sparsity in sparsity_ratios:
            if target_sparsity == 0 and dense_eval_cache is not None and not args.do_downstream_eval:
                actual_sparsity, ppl_by_seq = dense_eval_cache
                if args.skip_pp_eval:
                    continue
                for seq_len, ppl_test in ppl_by_seq:
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
                continue

            print(
                f"starting prune eval score_order={score_order} "
                f"sparsity={target_sparsity:.4f}"
            )
            current_model_device = resolve_model_device(args.model_device)
            current_model = get_llm(args.model, args.cache_dir, current_model_device, args.seqlen)
            try:
                current_model.eval()
                if target_sparsity > 0:
                    if args.prune_method == "random":
                        print(
                            f"Random pruning score_order={score_order} "
                            f"target_sparsity={target_sparsity:.4f}"
                        )
                        summary = prune_random(current_model, target_sparsity, score_order, args.seed, args.prune_ops)
                    else:
                        print(
                            f"Pruning with recomputed scores from {'memory' if score_dir is None else score_dir} "
                            f"score_order={score_order} target_sparsity={target_sparsity:.4f}"
                        )
                        summary = prune_by_scores(current_model, score_source, target_sparsity, score_order, args.prune_ops)
                    print(
                        f"{args.prune_method} prune summary: {summary['pruned']}/{summary['total']} "
                        f"({summary['actual_sparsity']:.4f})"
                    )

                actual_sparsity = check_sparsity(current_model, args.prune_ops)
                ppl_by_seq = []
                if not args.skip_pp_eval:
                    for seq_len in eval_seq_lens:
                        current_model.seqlen = int(seq_len)
                        ppl_test = eval_ppl_with_loader(current_model, eval_loader, current_model_device)
                        ppl_by_seq.append((int(seq_len), ppl_test))
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
                    run_downstream_eval(
                        args,
                        current_model,
                        tokenizer,
                        current_model_device,
                        out_dir,
                        score_order,
                        target_sparsity,
                        actual_sparsity,
                        downstream_examples,
                    )

                if target_sparsity == 0 and not args.skip_pp_eval:
                    dense_eval_cache = (actual_sparsity, ppl_by_seq)
            finally:
                del current_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
