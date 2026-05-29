import argparse
import os

import torch

from entry_utils import add_common_prune_args, run_prune_args
from model_utils import safe_hf_login, set_seed


def build_parser():
    parser = argparse.ArgumentParser()
    add_common_prune_args(parser)
    return parser


def main():
    safe_hf_login(os.environ.get("HF_TOKEN"))
    print("# of gpus: ", torch.cuda.device_count())

    args = build_parser().parse_args()
    set_seed(args.seed)
    run_prune_args(args)


if __name__ == "__main__":
    main()
