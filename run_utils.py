import os

import torch

from eval import eval_ppl_with_loader, load_wikitext2_eval
from model_utils import get_llm, get_tokenizer, resolve_model_device
from prune import check_sparsity
from prune_magnitude import compute_magnitude_scores
from prune_sparsegpt import compute_sparsegpt_scores
from prune_wanda import compute_wanda_scores
from result_utils import (
    append_pp_result,
    append_result_csv,
    draw_pp_vs_sparsity,
    result_dir,
)
from score_prune_utils import prune_by_scores


def score_save_dir(args):
    path = os.path.join(
        args.save,
        args.prune_method,
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
    if args.prune_method == "wanda":
        compute_wanda_scores(args, model, tokenizer, model_device, save_dir=score_dir)
    elif args.prune_method == "magnitude":
        compute_magnitude_scores(model, save_dir=score_dir)
    elif args.prune_method == "sparsegpt":
        compute_sparsegpt_scores(args, model, tokenizer, model_device, save_dir=score_dir)
    else:
        raise ValueError(f"Unsupported prune_method: {args.prune_method}")


def run_score_eval(args, score_dir):
    model_device = resolve_model_device(args.model_device)
    print(f"Using {model_device} model device")

    sparsity_ratios = resolve_sparsity_ratios(args)
    score_orders = resolve_score_orders(args)
    tokenizer = get_tokenizer(args.model, args.cache_dir)

    model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
    model.eval()
    compute_scores(args, model, tokenizer, model_device, score_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_pp_eval:
        print("Skipping PP eval.")
        return

    eval_seq_lens = args.pp_seqlen if len(args.pp_seqlen) >= 1 else [args.seqlen]
    print("loading wikitext2 eval data")
    eval_loader = load_wikitext2_eval(tokenizer)
    out_dir = result_dir(args)
    pp_log_path = os.path.join(out_dir, f"pp_eval_{args.prune_method}.txt")
    result_csv_path = os.path.join(out_dir, "pp_eval_results.csv")
    plot_path = os.path.join(out_dir, "pp_vs_sparsity.png")

    with open(pp_log_path, "a+", encoding="utf-8") as f:
        print(
            f"{'method':<12}{'score_order':<12}{'target_sparsity':<18}"
            f"{'actual_sparsity':<18}{'pp_seq_len':<12}{'ppl_test':<12}",
            file=f,
            flush=True,
        )

    dense_eval_cache = None

    for score_order in score_orders:
        for target_sparsity in sparsity_ratios:
            if target_sparsity == 0 and dense_eval_cache is not None:
                actual_sparsity, ppl_by_seq = dense_eval_cache
                for seq_len, ppl_test in ppl_by_seq:
                    append_pp_result(
                        pp_log_path,
                        args.prune_method,
                        score_order,
                        target_sparsity,
                        actual_sparsity,
                        int(seq_len),
                        ppl_test,
                    )
                    append_result_csv(
                        result_csv_path,
                        {
                            "method": args.prune_method,
                            "score_order": score_order,
                            "target_sparsity": f"{target_sparsity:.6f}",
                            "actual_sparsity": f"{actual_sparsity:.6f}",
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
            current_model.eval()
            if target_sparsity > 0:
                summary = prune_by_scores(current_model, score_dir, target_sparsity, score_order)
                print(
                    f"score prune summary: {summary['pruned']}/{summary['total']} "
                    f"({summary['actual_sparsity']:.4f})"
                )

            actual_sparsity = check_sparsity(current_model)
            ppl_by_seq = []
            for seq_len in eval_seq_lens:
                current_model.seqlen = int(seq_len)
                ppl_test = eval_ppl_with_loader(current_model, eval_loader, current_model_device)
                ppl_by_seq.append((int(seq_len), ppl_test))
                print(
                    f"wikitext perplexity {ppl_test} using score_order={score_order}, "
                    f"target_sparsity={target_sparsity:.4f}, pp_seqlen={seq_len}"
                )
                append_pp_result(
                    pp_log_path,
                    args.prune_method,
                    score_order,
                    target_sparsity,
                    actual_sparsity,
                    int(seq_len),
                    ppl_test,
                )
                append_result_csv(
                    result_csv_path,
                    {
                        "method": args.prune_method,
                        "score_order": score_order,
                        "target_sparsity": f"{target_sparsity:.6f}",
                        "actual_sparsity": f"{actual_sparsity:.6f}",
                        "pp_seq_len": int(seq_len),
                        "ppl_test": f"{ppl_test:.6f}",
                    },
                )

            if target_sparsity == 0:
                dense_eval_cache = (actual_sparsity, ppl_by_seq)

            del current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    drawn_path = draw_pp_vs_sparsity(result_csv_path, plot_path)
    if drawn_path is not None:
        print(f"Saved PP vs sparsity plot: {drawn_path}")
    del eval_loader
