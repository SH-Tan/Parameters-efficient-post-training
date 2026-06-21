import json
import os
import random
import subprocess

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "fp32": "float32",
    "float32": "float32",
    "full": "float32",
    "torch.float32": "float32",
}


TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def enable_hf_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def safe_hf_login(token):
    if not token:
        print("Hugging Face login skipped: HF_TOKEN is not set")
        return
    try:
        login(token)
        print("Hugging Face login succeeded")
    except Exception as exc:
        enable_hf_offline_mode()
        print(f"Hugging Face login skipped: {exc}")


def is_network_error(exc):
    msg = str(exc).lower()
    return (
        "name resolution" in msg
        or "connecterror" in msg
        or "connection error" in msg
        or "temporary failure" in msg
        or "offline" in msg
    )


def normalize_dtype_name(dtype_name):
    if dtype_name is None:
        return "auto"
    normalized = DTYPE_ALIASES.get(str(dtype_name).strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(DTYPE_ALIASES))
        raise ValueError(f"Unsupported model dtype '{dtype_name}'. Use one of: {choices}")
    return normalized


def config_torch_dtype(model_name):
    config_path = os.path.join(str(model_name), "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except Exception as exc:
        print(f"Could not read dtype from {config_path}: {exc}")
        return None
    dtype_name = config.get("torch_dtype") or config.get("dtype")
    if dtype_name is None:
        return None
    try:
        return normalize_dtype_name(dtype_name)
    except ValueError as exc:
        print(f"Ignoring unsupported dtype in {config_path}: {exc}")
        return None


def resolve_model_dtype(model_name, requested_dtype="auto"):
    normalized = normalize_dtype_name(requested_dtype)
    if normalized == "auto":
        normalized = config_torch_dtype(model_name) or "float16"
    return TORCH_DTYPES[normalized]


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def most_free_cuda_device():
    if not torch.cuda.is_available():
        return "cpu"

    try:
        best_idx = 0
        best_free = -1
        for idx in range(torch.cuda.device_count()):
            free_mem, _ = torch.cuda.mem_get_info(idx)
            if free_mem > best_free:
                best_idx = idx
                best_free = free_mem
        return f"cuda:{best_idx}"
    except Exception as exc:
        print(f"Could not query torch CUDA free memory: {exc}")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        best_idx = None
        best_free = -1
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            free_mem = int(parts[1])
            if free_mem > best_free:
                best_idx = idx
                best_free = free_mem
        if best_idx is not None:
            return f"cuda:{best_idx}"
    except Exception as exc:
        print(f"Could not query nvidia-smi for free GPU memory: {exc}")

    return "cuda:0"


def resolve_model_device(requested_device):
    if not torch.cuda.is_available():
        return "cpu"
    if requested_device in {"auto", "auto_free"}:
        return most_free_cuda_device()
    return requested_device


def get_llm(model_name, cache_dir="llm_weights", device="cpu", seqlen=1024, dtype="auto"):
    print("Loading model:", model_name)
    model_dtype = resolve_model_dtype(model_name, dtype)
    print("Loading model dtype:", model_dtype)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device,
        )
    except Exception as exc:
        if not is_network_error(exc):
            raise
        enable_hf_offline_mode()
        print(f"Falling back to local cached model files: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device,
            local_files_only=True,
        )

    if hasattr(model, "hf_device_map"):
        print("hf_device_map = ", model.hf_device_map)

    model.seqlen = int(seqlen)
    return model


def get_tokenizer(model_name, cache_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False,
        )
    except Exception as exc:
        if not is_network_error(exc):
            raise
        enable_hf_offline_mode()
        print(f"Falling back to local cached tokenizer files: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False,
            local_files_only=True,
        )
    return tokenizer
