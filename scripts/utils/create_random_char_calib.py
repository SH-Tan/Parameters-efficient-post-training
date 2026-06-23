#!/usr/bin/env python3
"""Create a synthetic random-character calibration parquet for sanity checks."""

import argparse
import random
import string
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output", default="dataset/deepseek1.5b/random_chars_8192.parquet")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--tokens-per-row", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-chars", type=int, default=32768)
    return parser.parse_args()


def random_text(rng, length):
    alphabet = string.ascii_letters + string.digits + string.punctuation + " \n\t"
    return "".join(rng.choice(alphabet) for _ in range(length))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []
    for _ in range(args.rows):
        token_ids = []
        while len(token_ids) < args.tokens_per_row:
            encoded = tokenizer(random_text(rng, args.chunk_chars), add_special_tokens=False).input_ids
            token_ids.extend(encoded)
        rows.append({"prompt_generated_trajectory_ids": token_ids[: args.tokens_per_row]})

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
