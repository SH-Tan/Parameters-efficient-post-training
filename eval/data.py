import random
import torch
from datasets import load_dataset


class TokenizerWrapper:
    def __init__(self, input_ids, pad_token_id=None):
        self.input_ids = input_ids
        self.pad_token_id = pad_token_id


class StreamingTextEvalWrapper:
    def __init__(self, dataset, tokenizer, seqlen, text_key="text", row_indices=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seqlen = int(seqlen)
        self.text_key = text_key
        self.row_indices = None if row_indices is None else set(row_indices)
        self.max_row_index = None if self.row_indices is None else max(self.row_indices)
        self.pad_token_id = _pad_token_id(tokenizer)

    def iter_input_ids(self):
        token_buffer = torch.empty(0, dtype=torch.long)
        for row_idx, sample in enumerate(self.dataset):
            if self.row_indices is not None and row_idx not in self.row_indices:
                if row_idx <= self.max_row_index:
                    continue
                break

            text = sample.get(self.text_key, "")
            if not text:
                continue

            input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
            if input_ids.numel() == 0:
                continue

            token_buffer = torch.cat((token_buffer, input_ids), dim=0)
            while token_buffer.numel() >= self.seqlen:
                yield token_buffer[: self.seqlen].unsqueeze(0)
                token_buffer = token_buffer[self.seqlen :]

        if token_buffer.numel() > 0:
            yield _left_pad_1d(token_buffer, self.seqlen, self.pad_token_id).unsqueeze(0)


def _pad_token_id(tokenizer):
    if tokenizer is None:
        return 0
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def _left_pad_1d(input_ids, seqlen, pad_token_id):
    if input_ids.numel() >= seqlen:
        return input_ids[:seqlen]

    padded = torch.full((seqlen,), int(pad_token_id), dtype=torch.long)
    padded[-input_ids.numel() :] = input_ids
    return padded


def _calibration_pair(input_ids):
    inp = input_ids.unsqueeze(0)
    tar = inp.clone()
    tar[:, :-1] = -100
    return inp, tar


def _sample_from_token_buffer(token_buffer, nsamples, seqlen, seed):
    total_tokens = token_buffer.shape[1]
    if total_tokens < seqlen:
        raise ValueError(
            f"Not enough tokens to build a sequence of length {seqlen}. "
            f"Only found {total_tokens} tokens."
        )

    random.seed(seed)
    trainloader = []
    max_start = total_tokens - seqlen
    for _ in range(nsamples):
        start = 0 if max_start == 0 else random.randint(0, max_start)
        inp = token_buffer[:, start:start + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def _build_token_buffer_from_texts(texts, tokenizer, min_tokens):
    chunks = []
    total_tokens = 0

    for text in texts:
        if not text:
            continue

        enc = tokenizer(text, return_tensors='pt')
        input_ids = enc.input_ids
        if input_ids.numel() == 0:
            continue

        chunks.append(input_ids)
        total_tokens += input_ids.shape[1]

        if total_tokens >= min_tokens:
            break

    if not chunks:
        raise ValueError("Dataset did not contain any tokenizable text.")

    return torch.cat(chunks, dim=1)


def _build_calibration_samples_from_token_ids(rows, ids_key, nsamples, seqlen, pad_token_id):
    trainloader = []
    for row in rows:
        ids = row.get(ids_key)
        if not ids:
            continue

        input_ids = torch.tensor(ids, dtype=torch.long)
        input_ids = _left_pad_1d(input_ids, seqlen, pad_token_id)
        trainloader.append(_calibration_pair(input_ids))

        if len(trainloader) >= nsamples:
            break

    if not trainloader:
        raise ValueError(f"Dataset did not contain any token ids in column {ids_key}.")
    if len(trainloader) < nsamples:
        raise ValueError(
            f"Not enough rows with token ids in column {ids_key}. "
            f"Need {nsamples}, found {len(trainloader)}."
        )

    return trainloader


def _metamathqa_text(row, include_response=True):
    query = row.get("query") or row.get("problem") or row.get("question")
    response = row.get("response") or row.get("solution") or row.get("answer")
    original_question = row.get("original_question")
    prompt = row.get("prompt")
    reward_model = row.get("reward_model")

    if not query and prompt:
        if isinstance(prompt, list):
            query = "\n".join(
                str(item.get("content", ""))
                for item in prompt
                if isinstance(item, dict) and item.get("content")
            )
        else:
            query = str(prompt)

    if include_response and not response and isinstance(reward_model, dict):
        response = reward_model.get("ground_truth")

    parts = []
    if original_question and original_question != query:
        parts.append(str(original_question))
    if query:
        parts.append(str(query))
    if include_response and response:
        parts.append(str(response))
    return "\n\n".join(parts)

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    trainloader = _sample_from_token_buffer(trainenc.input_ids, nsamples, seqlen, seed)
    return trainloader, testenc


def _load_c4_split(split_name):
    if split_name == "train":
        return load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )
    if split_name == "test":
        return load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
        )
    raise ValueError(f"Unsupported C4 split: {split_name}")


def _spread_indices(total_rows, sample_rows):
    if sample_rows >= total_rows:
        return list(range(total_rows))
    return sorted({int(index * total_rows / sample_rows) for index in range(sample_rows)})


def get_c4(nsamples, seed, seqlen, tokenizer, split_name="train"):
    random.seed(seed)

    traindata = _load_c4_split(split_name)
    trainenc = _build_token_buffer_from_texts(
        (sample["text"] for sample in traindata), tokenizer, nsamples * seqlen
    )
    trainloader = _sample_from_token_buffer(trainenc, nsamples, seqlen, seed)

    valenc = None
    return trainloader, valenc


def get_c4_eval(seqlen, tokenizer, sample_rows=3000, total_rows=364608):
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    return StreamingTextEvalWrapper(
        dataset,
        tokenizer,
        seqlen,
        row_indices=_spread_indices(total_rows, sample_rows),
    )


def get_metamathqa_math_500(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset("ShuoZheLi/MetaMathQA-math-500", split="train")
    texts = (_metamathqa_text(row, include_response=True) for row in dataset)
    token_buffer = _build_token_buffer_from_texts(texts, tokenizer, nsamples * seqlen)
    trainloader = _sample_from_token_buffer(token_buffer, nsamples, seqlen, seed)

    eval_dataset = load_dataset("ShuoZheLi/MetaMathQA-math-500", split="test")
    eval_text = "\n\n".join(_metamathqa_text(row, include_response=False) for row in eval_dataset)
    valenc = TokenizerWrapper(tokenizer(eval_text, return_tensors="pt").input_ids)
    return trainloader, valenc


def get_actor_math_500_response(nsamples, seed, seqlen, tokenizer):
    data_file = "job_05b_vh_init_e5_metamath_global_step_800/job_05b_vh_init_e5_metamath_global_step_800.parquet"
    dataset = load_dataset(
        "ShuoZheLi/actor_math_500_response",
        data_files=data_file,
        split="train",
        streaming=True,
    )
    trainloader = _build_calibration_samples_from_token_ids(
        dataset,
        "prompt_generated_trajectory_ids",
        nsamples,
        seqlen,
        _pad_token_id(tokenizer),
    )
    return trainloader, None


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name in {"c4", "c4_train"}:
        return get_c4(nsamples, seed, seqlen, tokenizer, split_name="train")
    if name in {"c4_test", "c4_validation"}:
        return get_c4(nsamples, seed, seqlen, tokenizer, split_name="test")
    if name in {"metamathqa_math_500", "MetaMathQA-math-500", "math_500"}:
        return get_metamathqa_math_500(nsamples, seed, seqlen, tokenizer)
    if name in {"actor_math_500_response", "actor_math_500_response_ids"}:
        return get_actor_math_500_response(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported dataset name: {name}")
