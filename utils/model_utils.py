import os
import random
import subprocess

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def cuda_visible_devices_for_device(device):
    device = str(device)
    if not device.startswith("cuda"):
        return None

    if device == "cuda":
        logical_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    else:
        logical_idx = int(device.split(":", 1)[1])

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return str(logical_idx)

    entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
    if logical_idx >= len(entries):
        return str(logical_idx)
    return entries[logical_idx]


def _torch_dtype_from_config(config):
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = dtype.removeprefix("torch.")
        return getattr(torch, dtype, None)
    return dtype


def _cuda_supports_bfloat16(device):
    if not torch.cuda.is_available():
        return False
    try:
        cuda_device = torch.device(device)
        major, _ = torch.cuda.get_device_capability(cuda_device)
        return major >= 8
    except Exception:
        return torch.cuda.is_bf16_supported()


def _use_config_bfloat16(model_name, config):
    name = str(model_name).lower()
    return "deepseek" in name and _torch_dtype_from_config(config) is torch.bfloat16


def resolve_model_dtype(model_name, cache_dir, device):
    try:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as exc:
        if not is_network_error(exc):
            raise
        enable_hf_offline_mode()
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)

    dtype = torch.bfloat16 if _use_config_bfloat16(model_name, config) else None
    if dtype is torch.bfloat16 and str(device).startswith("cuda") and not _cuda_supports_bfloat16(device):
        print("Model config requests bfloat16 but selected CUDA device does not support BF16; using float16")
        return torch.float16
    if dtype is not None:
        return dtype
    if str(device).startswith("cuda"):
        return torch.float16
    return "auto"


def get_llm(model_name, cache_dir="llm_weights", device="cpu", seqlen=1024):
    print("Loading model:", model_name)
    dtype = resolve_model_dtype(model_name, cache_dir, device)
    print(f"Model load dtype: {dtype}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
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
            dtype=dtype,
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
