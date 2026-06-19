from eval.run_utils import run_score_eval, score_save_dir


PRUNE_METHODS = ["wanda", "magnitude", "sparsegpt", "random"]
CALIB_DATASETS = [
    "c4",
    "c4_train",
    "c4_test",
    "c4_validation",
    "metamathqa_math_500",
    "MetaMathQA-math-500",
    "math_500",
    "actor_math_500_response",
    "actor_math_500_response_ids",
    "deepseek_1d5_8192_response",
    "deepseek_1.5b_8192_response",
]
PP_EVAL_DATASETS = ["wikitext2", "c4_test", "c4_validation"]
SCORE_ORDERS = ["global", "local", "per_op"]


def add_common_prune_args(parser, require_model=True, default_model=None, model_help="HF model name or local path."):
    parser.add_argument("--model", type=str, required=require_model, default=default_model, help=model_help)
    parser.add_argument("--seed", type=int, default=13, help="Seed for calibration sampling.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--prune_method", type=str, choices=PRUNE_METHODS, required=True)
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        nargs="+",
        default=[0.0],
        help="One or more sparsity levels for prune eval.",
    )
    parser.add_argument(
        "--score_order",
        type=str,
        nargs="+",
        choices=SCORE_ORDERS,
        default=SCORE_ORDERS,
        help="Score ordering for prune eval: global all layers/ops, local per layer, or per op.",
    )
    parser.add_argument(
        "--prune_ops",
        type=str,
        nargs="+",
        default=None,
        help="Ops to prune. Accepts q,k,v or q k v; full names like q_proj also work.",
    )
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--save", type=str, default="scores", help="Directory to save outputs.")
    parser.add_argument("--calib_forward_batch_size", type=int, default=1, help="WANDA/SparseGPT calibration forward microbatch size.")
    parser.add_argument("--sparsegpt_hessian_chunk_size", type=int, default=8192, help="Token chunk size for SparseGPT Hessian updates. Use <=0 to disable chunking.")
    parser.add_argument(
        "--model_device",
        type=str,
        default="auto_free",
        help="Device map for model load. Use auto_free to pick the GPU with most free memory.",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="c4",
        choices=CALIB_DATASETS,
        help="Calibration data for score calculation.",
    )
    parser.add_argument("--seqlen", type=int, default=1024, help="Calibration sequence length.")
    parser.add_argument(
        "--pp_eval_data",
        type=str,
        default="wikitext2",
        choices=PP_EVAL_DATASETS,
        help="Dataset for perplexity evaluation.",
    )
    parser.add_argument(
        "--pp_seqlen",
        type=int,
        nargs="*",
        default=[],
        help="Perplexity eval sequence lengths. Defaults to --seqlen.",
    )
    parser.add_argument("--skip_pp_eval", action="store_true", help="Only save scores; skip PPL eval.")
    parser.add_argument("--do_downstream_eval", action="store_true", help="Run downstream task accuracy after pruning.")
    parser.add_argument(
        "--downstream_task_data",
        default="ShuoZheLi/MetaMathQA-math-500",
        help="Local parquet path or Hugging Face dataset ID for downstream task accuracy.",
    )
    parser.add_argument("--downstream_prompt_key", default="prompt")
    parser.add_argument("--downstream_response_key", default=None)
    parser.add_argument("--downstream_reward_score_dir", default=None)
    parser.add_argument("--downstream_max_examples", type=int, default=500, help="Use -1 for all downstream examples.")
    parser.add_argument("--downstream_start_index", type=int, default=0)
    parser.add_argument("--downstream_shuffle", action="store_true")
    parser.add_argument("--downstream_batch_size", type=int, default=1)
    parser.add_argument("--downstream_generation_max_batch_tokens", type=int, default=32768, help="Cap prompt+generation tokens per downstream generation microbatch. Use <=0 to disable.")
    parser.add_argument("--downstream_use_cache", action="store_true", help="Use generation KV cache for downstream eval. Faster but uses more GPU memory.")
    parser.add_argument("--downstream_max_prompt_length", type=int, default=2048)
    parser.add_argument("--downstream_max_new_tokens", type=int, default=2048)
    parser.add_argument("--downstream_temperature", type=float, default=0.0)
    parser.add_argument("--downstream_top_p", type=float, default=1.0)
    parser.add_argument("--downstream_top_k", type=int, default=0)
    parser.add_argument("--downstream_response_log_max", type=int, default=-1, help="Maximum downstream responses to write; -1 writes all.")
    parser.add_argument(
        "--save_score_pkl",
        dest="save_score_pkl",
        action="store_true",
        default=False,
        help="Persist per-layer score PKLs.",
    )
    parser.add_argument(
        "--no_save_score_pkl",
        dest="save_score_pkl",
        action="store_false",
        help="Recompute scores in memory without saving or loading score PKLs.",
    )


def run_prune_args(args):
    if args.prune_method == "random":
        print("Using random pruning; no score PKLs will be loaded or saved.")
        run_score_eval(args, None)
        return
    if args.save_score_pkl:
        score_dir = score_save_dir(args)
        print(f"Using persistent score directory: {score_dir}")
        run_score_eval(args, score_dir)
    else:
        print(f"Using in-memory recomputed {args.prune_method} scores; no score PKLs will be loaded or saved.")
        run_score_eval(args, None)
