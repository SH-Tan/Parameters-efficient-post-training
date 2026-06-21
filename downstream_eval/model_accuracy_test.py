from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATASET = "ShuoZheLi/MetaMathQA-math-500"
DEFAULT_OUTPUT = None
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


@dataclass(frozen=True)
class DownstreamEvalConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_prompt_length: int = 2048
    max_new_tokens: int = 2048
    batch_size: int = 1
    generation_max_batch_tokens: int = 32768
    use_cache: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    response_log_max: int = -1
    backend: str = "transformers"
    model_path: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    enforce_eager: bool = True


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


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


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


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


def _resolve_dataset_name(path: str | Path) -> str:
    path = str(path)
    if path in METAMATHQA_MATH_500_ALIASES:
        return "ShuoZheLi/MetaMathQA-math-500"
    return path


def _cached_metamathqa_math_500_test_path() -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--ShuoZheLi--MetaMathQA-math-500"
    ref_path = cache_root / "refs" / "main"
    if not ref_path.is_file():
        return None
    test_path = cache_root / "snapshots" / ref_path.read_text(encoding="utf-8").strip() / METAMATHQA_MATH_500_TEST_FILE
    return test_path if test_path.is_file() else None


def _load_dataframe(path: str | Path) -> pd.DataFrame:
    path_str = str(path)
    local_path = Path(path_str).expanduser()
    if local_path.is_file() or path_str.endswith(".parquet"):
        return pd.read_parquet(local_path)

    if path_str in METAMATHQA_MATH_500_ALIASES:
        cached_test_path = _cached_metamathqa_math_500_test_path()
        if cached_test_path is not None:
            return pd.read_parquet(cached_test_path)
        dataset = load_dataset(
            _resolve_dataset_name(path_str),
            data_files={"test": METAMATHQA_MATH_500_TEST_FILE},
            split="test",
        )
        return dataset.to_pandas()

    dataset = load_dataset(_resolve_dataset_name(path_str), split="test")
    return dataset.to_pandas()


def _extract_prompt_value(row: pd.Series, prompt_key: str) -> Any:
    if prompt_key in row and not _is_missing(row[prompt_key]):
        return row[prompt_key]
    for key in ("prompt", "query", "problem", "question", "original_question"):
        if key in row and not _is_missing(row[key]):
            return row[key]
    raise KeyError(f"Cannot find prompt column. Requested {prompt_key!r}; available columns: {list(row.index)}")


def _extract_data_source(row: pd.Series, dataset_path: str | Path) -> str:
    data_source = row.get("data_source", "")
    data_source = "" if _is_missing(data_source) else str(data_source)
    if not data_source and str(dataset_path) in METAMATHQA_MATH_500_ALIASES:
        return "math_500"
    return data_source


def load_examples(
    path: str | Path,
    tokenizer,
    *,
    prompt_key: str,
    response_key: str | None,
    start_index: int,
    max_examples: int,
    shuffle: bool,
    seed: int,
) -> list[ExampleRecord]:
    dataframe = _load_dataframe(path)
    indices = list(range(len(dataframe)))
    if start_index:
        indices = indices[start_index:]
    if shuffle:
        random.Random(seed).shuffle(indices)
    if max_examples >= 0:
        indices = indices[:max_examples]

    examples: list[ExampleRecord] = []
    for index in indices:
        row = dataframe.iloc[index]
        examples.append(
            ExampleRecord(
                example_id=int(index),
                prompt_text=normalize_prompt(_extract_prompt_value(row, prompt_key), tokenizer),
                data_source=_extract_data_source(row, path),
                ground_truth=extract_ground_truth(row, response_key=response_key),
            )
        )
    return examples


def _reward_module_path(module_name: str, reward_score_dir: str | Path | None = None) -> Path:
    if reward_score_dir is not None:
        return Path(reward_score_dir).expanduser() / f"{module_name}.py"
    if os.environ.get("VERL_REWARD_SCORE_DIR"):
        return Path(os.environ["VERL_REWARD_SCORE_DIR"]).expanduser() / f"{module_name}.py"

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
    spec = importlib.util.spec_from_file_location(f"_task_accuracy_reward_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _last_braced_content(text: str, marker: str) -> str | None:
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    if index >= len(text) or text[index] != "{":
        return None

    depth = 0
    for pos in range(index, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1:pos]
    return None


def _extract_math_answer(text: Any) -> str:
    text = "" if text is None else str(text)
    boxed = _last_braced_content(text, "\\boxed")
    if boxed is not None:
        return boxed

    matches = re.findall(r"(?:final answer|answer is|answer:)\s*([^\n\.]+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    return text.strip().splitlines()[-1] if text.strip() else ""


def _normalize_math_answer(answer: Any) -> str:
    answer = _extract_math_answer(answer)
    answer = answer.strip().strip("$").strip()
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


def compute_score_with_reward_module(
    data_source: str,
    response_text: str,
    ground_truth: Any,
    reward_score_dir: str | Path | None = None,
) -> Any:
    if data_source == "openai/gsm8k":
        return _load_reward_module("gsm8k", reward_score_dir).compute_score(response_text, ground_truth)
    if data_source in MATH_DATA_SOURCES:
        try:
            return _load_reward_module("math_reward", reward_score_dir).compute_score(response_text, ground_truth)
        except FileNotFoundError:
            return _fallback_math_score(response_text, ground_truth)
    if data_source in {"math_dapo", "math", "math_dapo_reasoning"} or data_source.startswith("aime"):
        try:
            return _load_reward_module("math_dapo", reward_score_dir).compute_score(
                response_text,
                ground_truth,
                incorrect_reward=0.0,
            )
        except FileNotFoundError:
            return _fallback_math_score(response_text, ground_truth)
    raise NotImplementedError(f"Reward function is not implemented for data_source={data_source!r}")


def score_response(
    example: ExampleRecord,
    response_text: str,
    reward_score_dir: str | Path | None = None,
) -> float:
    score = compute_score_with_reward_module(
        example.data_source,
        response_text,
        example.ground_truth,
        reward_score_dir=reward_score_dir,
    )
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)


def _sampling_kwargs(args: argparse.Namespace | DownstreamEvalConfig) -> dict[str, Any]:
    do_sample = args.temperature > 0
    kwargs = {
        "temperature": float(args.temperature) if do_sample else 0.0,
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_new_tokens),
    }
    top_k = int(getattr(args, "top_k", 0))
    if top_k > 0:
        kwargs["top_k"] = top_k
    return kwargs


def _generation_kwargs(model, tokenizer, args: argparse.Namespace | DownstreamEvalConfig) -> dict[str, Any]:
    do_sample = args.temperature > 0

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "use_cache": bool(getattr(args, "use_cache", False)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        generation_kwargs["top_k"] = args.top_k
    return generation_kwargs


def generate_response(model, tokenizer, prompt_text: str, args: argparse.Namespace, device: torch.device) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
        return_token_type_ids=False,
    ).to(device)

    generated = model.generate(**inputs, **_generation_kwargs(model, tokenizer, args))
    response_ids = generated[0, inputs["input_ids"].shape[1] :].detach().cpu()
    del inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    del response_ids
    return response


def generate_responses(model, tokenizer, prompt_texts: list[str], args: argparse.Namespace | DownstreamEvalConfig, device: torch.device) -> list[str]:
    if len(prompt_texts) == 1:
        return [generate_response(model, tokenizer, prompt_texts[0], args, device)]

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
        padding=True,
        return_token_type_ids=False,
    ).to(device)
    prompt_width = inputs["input_ids"].shape[1]
    generated = model.generate(**inputs, **_generation_kwargs(model, tokenizer, args))
    generated = generated.detach().cpu()
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    responses = []
    for row_idx in range(len(prompt_texts)):
        response_ids = generated[row_idx, prompt_width:]
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    del generated, response_ids
    return responses


def _prompt_token_count(tokenizer, prompt_text: str, max_prompt_length: int) -> int:
    input_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_prompt_length,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return len(input_ids)


def _generation_microbatches(examples: list[ExampleRecord], tokenizer, args: argparse.Namespace | DownstreamEvalConfig):
    batch_size = max(1, int(getattr(args, "batch_size", 1)))
    max_batch_tokens = int(getattr(args, "generation_max_batch_tokens", 0))
    if max_batch_tokens <= 0:
        for start in range(0, len(examples), batch_size):
            yield examples[start:start + batch_size]
        return

    batch = []
    batch_tokens = 0
    max_prompt_length = int(args.max_prompt_length)
    max_new_tokens = int(args.max_new_tokens)
    for example in examples:
        prompt_tokens = _prompt_token_count(tokenizer, example.prompt_text, max_prompt_length)
        example_tokens = max(1, prompt_tokens + max_new_tokens)
        if batch and (len(batch) >= batch_size or batch_tokens + example_tokens > max_batch_tokens):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(example)
        batch_tokens += example_tokens

    if batch:
        yield batch


def _open_response_log(output_path: str | Path | None):
    if output_path is None:
        return None
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8")


def _score_generated_responses(
    examples: list[ExampleRecord],
    responses: list[str],
    *,
    output_handle,
    response_log_max: int,
    logged_responses: int,
    reward_score_dir: str | Path | None = None,
) -> tuple[list[float], list[bool], int, int]:
    scores: list[float] = []
    correct: list[bool] = []
    num_unscored = 0

    if len(examples) != len(responses):
        raise ValueError(f"Response count mismatch: {len(responses)} responses for {len(examples)} examples")

    for example, response in zip(examples, responses):
        row = None
        should_log_response = output_handle is not None and (
            response_log_max < 0 or logged_responses < response_log_max
        )
        if should_log_response:
            row = {"example_id": example.example_id, "prompt": example.prompt_text, "response": response}
        if example.ground_truth is None:
            num_unscored += 1
            if row is not None:
                row["task_score"] = None
        else:
            score = score_response(example, response, reward_score_dir=reward_score_dir)
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

    return scores, correct, num_unscored, logged_responses


def _metrics_from_scores(num_examples: int, scores: list[float], correct: list[bool], num_unscored: int) -> dict[str, Any]:
    metrics = {
        "num_examples": num_examples,
        "num_scored": len(scores),
        "num_unscored": num_unscored,
    }
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
    return metrics


def _vllm_outputs_to_texts(outputs) -> list[str]:
    texts = []
    for output in outputs:
        if not output.outputs:
            texts.append("")
        else:
            texts.append(output.outputs[0].text)
    return texts


def evaluate_vllm_task_accuracy(
    model_path: str | Path,
    tokenizer,
    examples: list[ExampleRecord],
    args: argparse.Namespace | DownstreamEvalConfig,
    *,
    output_path: str | Path | None = None,
    reward_score_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No examples were loaded. Check dataset path and slicing arguments.")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("downstream_backend='vllm' requires the vllm package in the active environment") from exc

    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=max(1, int(getattr(args, "tensor_parallel_size", 1))),
        gpu_memory_utilization=float(getattr(args, "gpu_memory_utilization", 0.9)),
        dtype=str(getattr(args, "dtype", "auto")),
        max_model_len=int(args.max_prompt_length) + int(args.max_new_tokens),
        trust_remote_code=True,
        enforce_eager=bool(getattr(args, "enforce_eager", True)),
    )
    sampling_params = SamplingParams(**_sampling_kwargs(args))
    response_log_max = int(getattr(args, "response_log_max", -1))
    logged_responses = 0
    scores: list[float] = []
    correct: list[bool] = []
    num_unscored = 0
    output_handle = _open_response_log(output_path)

    try:
        requested_batch_size = max(1, int(getattr(args, "batch_size", 1)))
        max_batch_tokens = int(getattr(args, "generation_max_batch_tokens", 0))
        if max_batch_tokens > 0:
            print(
                f"using vLLM downstream dynamic microbatches "
                f"(requested_batch={requested_batch_size}, max_batch_tokens={max_batch_tokens})"
            )
        with tqdm(total=len(examples), desc="Evaluating") as progress:
            for batch_examples in _generation_microbatches(examples, tokenizer, args):
                prompts = [example.prompt_text for example in batch_examples]
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                responses = _vllm_outputs_to_texts(outputs)
                batch_scores, batch_correct, batch_unscored, logged_responses = _score_generated_responses(
                    batch_examples,
                    responses,
                    output_handle=output_handle,
                    response_log_max=response_log_max,
                    logged_responses=logged_responses,
                    reward_score_dir=reward_score_dir,
                )
                scores.extend(batch_scores)
                correct.extend(batch_correct)
                num_unscored += batch_unscored
                progress.update(len(batch_examples))
    finally:
        if output_handle is not None:
            output_handle.close()
        try:
            llm.shutdown()
        except AttributeError:
            pass
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _metrics_from_scores(len(examples), scores, correct, num_unscored)


def evaluate_model_task_accuracy(
    model,
    tokenizer,
    examples: list[ExampleRecord],
    args: argparse.Namespace | DownstreamEvalConfig,
    *,
    output_path: str | Path | None = None,
    reward_score_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No examples were loaded. Check dataset path and slicing arguments.")

    device = torch.device(args.device)
    scores: list[float] = []
    correct: list[bool] = []
    num_unscored = 0
    response_log_max = int(getattr(args, "response_log_max", -1))
    logged_responses = 0
    output_handle = _open_response_log(output_path)

    try:
        requested_batch_size = max(1, int(getattr(args, "batch_size", 1)))
        max_batch_tokens = int(getattr(args, "generation_max_batch_tokens", 0))
        if max_batch_tokens > 0:
            print(
                f"using downstream dynamic microbatches "
                f"(requested_batch={requested_batch_size}, max_batch_tokens={max_batch_tokens})"
            )
        with torch.inference_mode():
            with tqdm(total=len(examples), desc="Evaluating") as progress:
                for batch_examples in _generation_microbatches(examples, tokenizer, args):
                    responses = generate_responses(
                        model,
                        tokenizer,
                        [example.prompt_text for example in batch_examples],
                        args,
                        device,
                    )
                    batch_scores, batch_correct, batch_unscored, logged_responses = _score_generated_responses(
                        batch_examples,
                        responses,
                        output_handle=output_handle,
                        response_log_max=response_log_max,
                        logged_responses=logged_responses,
                        reward_score_dir=reward_score_dir,
                    )
                    scores.extend(batch_scores)
                    correct.extend(batch_correct)
                    num_unscored += batch_unscored
                    progress.update(len(batch_examples))
                    del responses, batch_examples
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        if output_handle is not None:
            output_handle.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _metrics_from_scores(len(examples), scores, correct, num_unscored)


def evaluate_downstream_task_accuracy(
    model,
    tokenizer,
    dataset_path: str | Path = DEFAULT_DATASET,
    *,
    examples: list[ExampleRecord] | None = None,
    prompt_key: str = "prompt",
    response_key: str | None = None,
    start_index: int = 0,
    max_examples: int = 500,
    shuffle: bool = False,
    seed: int = 42,
    config: DownstreamEvalConfig | None = None,
    output_path: str | Path | None = None,
    reward_score_dir: str | Path | None = None,
) -> dict[str, Any]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if examples is None:
        examples = load_examples(
            dataset_path,
            tokenizer,
            prompt_key=prompt_key,
            response_key=response_key,
            start_index=start_index,
            max_examples=max_examples,
            shuffle=shuffle,
            seed=seed,
        )
    eval_config = config or DownstreamEvalConfig(device=str(_model_device(model)))
    backend = getattr(eval_config, "backend", "transformers")
    if backend == "transformers":
        return evaluate_model_task_accuracy(
            model,
            tokenizer,
            examples,
            eval_config,
            output_path=output_path,
            reward_score_dir=reward_score_dir,
        )
    if backend == "vllm":
        if not eval_config.model_path:
            raise ValueError("vLLM downstream eval requires config.model_path pointing to a saved HF checkpoint")
        return evaluate_vllm_task_accuracy(
            eval_config.model_path,
            tokenizer,
            examples,
            eval_config,
            output_path=output_path,
            reward_score_dir=reward_score_dir,
        )
    raise ValueError(f"Unsupported downstream backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream task accuracy on a local or Hugging Face dataset.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None, help="Optional dataset column containing ground-truth answers.")
    parser.add_argument("--reward_score_dir", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=500, help="Use -1 for all examples.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--generation_max_batch_tokens", type=int, default=32768, help="Cap prompt+generation tokens per generation microbatch. Use <=0 to disable.")
    parser.add_argument("--use_cache", action="store_true", help="Use generation KV cache. Faster but uses more GPU memory.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--response_log_max", type=int, default=-1, help="Maximum responses to write; -1 writes all.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

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
    metrics = evaluate_model_task_accuracy(
        model,
        tokenizer,
        examples,
        args,
        output_path=args.output_path,
        reward_score_dir=args.reward_score_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
