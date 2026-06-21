from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DEFAULT_DATASET = "ShuoZheLi/MetaMathQA-math-500"
METAMATHQA_MATH_500_ALIASES = {
    "ShuoZheLi/MetaMathQA-math-500",
    "MetaMathQA-math-500",
    "metamathqa_math_500",
    "math_500",
}
MATH_DATA_SOURCES = {"lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math_500"}
METAMATHQA_MATH_500_TEST_FILE = "test.parquet"


@dataclass(frozen=True)
class ExampleRecord:
    example_id: int
    prompt_text: str
    data_source: str
    ground_truth: Any


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def normalize_prompt(prompt: Any, tokenizer) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        if "messages" in prompt:
            return normalize_prompt(prompt["messages"], tokenizer)
        for key in ("prompt", "text", "content"):
            if key in prompt:
                return str(prompt[key])
        return json.dumps(prompt, ensure_ascii=True)
    if isinstance(prompt, list):
        if not prompt:
            return ""
        if all(isinstance(item, dict) for item in prompt) and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        if all(isinstance(item, dict) for item in prompt):
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in prompt)
        if all(isinstance(item, str) for item in prompt):
            return "\n".join(prompt)
        return "\n".join(str(item) for item in prompt)
    return str(prompt)


def extract_ground_truth(row: pd.Series, response_key: str | None) -> Any:
    if response_key and response_key in row and not _is_missing(row[response_key]):
        return row[response_key]
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")
    for key in ("ground_truth", "answer", "solution", "response"):
        if key in row and not _is_missing(row[key]):
            return row[key]
    return None


def _load_dataframe(path: str | Path) -> pd.DataFrame:
    path_str = str(path)
    local_path = Path(path_str).expanduser()
    if local_path.is_file() or path_str.endswith(".parquet"):
        return pd.read_parquet(local_path)
    from datasets import load_dataset
    dataset_name = "ShuoZheLi/MetaMathQA-math-500" if path_str in METAMATHQA_MATH_500_ALIASES else path_str
    if path_str in METAMATHQA_MATH_500_ALIASES:
        dataset = load_dataset(dataset_name, data_files={"test": METAMATHQA_MATH_500_TEST_FILE}, split="test")
    else:
        dataset = load_dataset(dataset_name, split="test")
    return dataset.to_pandas()


def load_examples(path: str | Path, tokenizer, *, prompt_key: str, response_key: str | None, start_index: int, max_examples: int, shuffle: bool, seed: int) -> list[ExampleRecord]:
    dataframe = _load_dataframe(path)
    indices = list(range(len(dataframe)))
    if start_index:
        indices = indices[start_index:]
    if shuffle:
        random.Random(seed).shuffle(indices)
    if max_examples >= 0:
        indices = indices[:max_examples]
    examples = []
    for index in indices:
        row = dataframe.iloc[index]
        data_source = row.get("data_source", "")
        data_source = "" if _is_missing(data_source) else str(data_source)
        if not data_source and str(path) in METAMATHQA_MATH_500_ALIASES:
            data_source = "math_500"
        examples.append(
            ExampleRecord(
                example_id=int(index),
                prompt_text=normalize_prompt(row[prompt_key], tokenizer),
                data_source=data_source,
                ground_truth=extract_ground_truth(row, response_key),
            )
        )
    return examples


def _extract_math_answer(text: Any) -> str:
    text = str(text)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1]
    return text.strip().splitlines()[-1] if text.strip() else ""


def _normalize_math_answer(answer: Any) -> str:
    answer = _extract_math_answer(answer).strip().strip("$").strip()
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", answer)
    answer = answer.replace(",", "")
    answer = re.sub(r"\s+", "", answer)
    return answer.lower()


def _fallback_math_score(response_text: str, ground_truth: Any) -> float:
    prediction = _normalize_math_answer(response_text)
    target = _normalize_math_answer(ground_truth)
    return float(bool(prediction) and prediction == target)


def _reward_module_path(module_name: str, reward_score_dir: str | Path | None = None) -> Path:
    if reward_score_dir is not None:
        return Path(reward_score_dir).expanduser() / f"{module_name}.py"
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "verl" / "utils" / "reward_score" / f"{module_name}.py",
        here.parents[2] / "verl" / "utils" / "reward_score" / f"{module_name}.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def _load_reward_module(module_name: str, reward_score_dir: str | Path | None = None):
    module_path = _reward_module_path(module_name, reward_score_dir)
    if not module_path.is_file():
        raise FileNotFoundError(f"Reward module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"_vllm_accuracy_reward_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_response(example: ExampleRecord, response_text: str, reward_score_dir: str | Path | None = None) -> float:
    if example.data_source == "openai/gsm8k":
        score = _load_reward_module("gsm8k", reward_score_dir).compute_score(response_text, example.ground_truth)
    elif example.data_source in MATH_DATA_SOURCES:
        try:
            score = _load_reward_module("math_reward", reward_score_dir).compute_score(response_text, example.ground_truth)
        except FileNotFoundError:
            score = _fallback_math_score(response_text, example.ground_truth)
    elif example.data_source in {"math_dapo", "math", "math_dapo_reasoning"} or example.data_source.startswith("aime"):
        try:
            score = _load_reward_module("math_dapo", reward_score_dir).compute_score(response_text, example.ground_truth, incorrect_reward=0.0)
        except FileNotFoundError:
            score = _fallback_math_score(response_text, example.ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for data_source={example.data_source!r}")
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream task accuracy with vLLM in an isolated process.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None)
    parser.add_argument("--reward_score_dir", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--generation_max_batch_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--response_log_max", type=int, default=-1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    eager_group = parser.add_mutually_exclusive_group()
    eager_group.add_argument("--enforce_eager", "--enforce-eager", dest="enforce_eager", action="store_true")
    eager_group.add_argument("--no_enforce_eager", "--no-enforce_eager", "--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.set_defaults(enforce_eager=True)
    return parser.parse_args()


def _prompt_token_count(tokenizer, prompt_text: str, max_prompt_length: int) -> int:
    return len(tokenizer(prompt_text, truncation=True, max_length=max_prompt_length, return_attention_mask=False, return_token_type_ids=False)["input_ids"])


def _generation_microbatches(examples: list[ExampleRecord], tokenizer, args: argparse.Namespace):
    batch_size = max(1, int(args.batch_size))
    max_batch_tokens = int(args.generation_max_batch_tokens)
    if max_batch_tokens <= 0:
        for start in range(0, len(examples), batch_size):
            yield examples[start:start + batch_size]
        return
    batch = []
    batch_tokens = 0
    for example in examples:
        example_tokens = max(1, _prompt_token_count(tokenizer, example.prompt_text, args.max_prompt_length) + int(args.max_new_tokens))
        if batch and (len(batch) >= batch_size or batch_tokens + example_tokens > max_batch_tokens):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(example)
        batch_tokens += example_tokens
    if batch:
        yield batch


def _sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams
    kwargs = {
        "temperature": float(args.temperature) if args.temperature > 0 else 0.0,
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_new_tokens),
    }
    if int(args.top_k) > 0:
        kwargs["top_k"] = int(args.top_k)
    return SamplingParams(**kwargs)


def configure_cuda_multiprocessing() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def main() -> None:
    configure_cuda_multiprocessing()
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    examples = load_examples(
        args.dataset_path,
        tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    from vllm import LLM
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=max(1, int(args.tensor_parallel_size)),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        dtype=str(args.dtype),
        max_model_len=int(args.max_prompt_length) + int(args.max_new_tokens),
        trust_remote_code=True,
        enforce_eager=bool(args.enforce_eager),
    )
    sampling_params = _sampling_params(args)
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response_log_max = int(args.response_log_max)
    logged_responses = 0
    scores: list[float] = []
    correct: list[bool] = []
    num_unscored = 0
    try:
        with output_path.open("w", encoding="utf-8") as output_handle:
            with tqdm(total=len(examples), desc="Evaluating") as progress:
                for batch_examples in _generation_microbatches(examples, tokenizer, args):
                    outputs = llm.generate([example.prompt_text for example in batch_examples], sampling_params, use_tqdm=False)
                    responses = [output.outputs[0].text if output.outputs else "" for output in outputs]
                    for example, response in zip(batch_examples, responses):
                        row = None
                        if response_log_max < 0 or logged_responses < response_log_max:
                            row = {"example_id": example.example_id, "prompt": example.prompt_text, "response": response}
                        if example.ground_truth is None:
                            num_unscored += 1
                            if row is not None:
                                row["task_score"] = None
                        else:
                            score = score_response(example, response, reward_score_dir=args.reward_score_dir)
                            is_correct = bool(score == 1.0)
                            scores.append(score)
                            correct.append(is_correct)
                            if row is not None:
                                row["task_score"] = score
                                row["is_correct"] = is_correct
                        if row is not None:
                            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output_handle.flush()
                            logged_responses += 1
                    progress.update(len(batch_examples))
    finally:
        try:
            llm.shutdown()
        except AttributeError:
            pass
        del llm

    metrics = {"num_examples": len(examples), "num_scored": len(scores), "num_unscored": num_unscored}
    if scores:
        metrics.update(
            {
                "pass@1": float(np.mean(correct)),
                "accuracy": float(np.mean(correct)),
                "mean_score": float(np.mean(scores)),
                "score_sum": float(np.sum(scores)),
                "num_correct": int(np.sum(correct)),
            }
        )
    metrics_path = Path(args.metrics_path).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
