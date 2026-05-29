import random
import torch
from datasets import load_dataset


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


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


def _metamathqa_text(row, include_response=True):
    query = row.get("query") or row.get("problem") or row.get("question")
    response = row.get("response") or row.get("solution") or row.get("answer")
    original_question = row.get("original_question")

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


def get_c4(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
    trainenc = _build_token_buffer_from_texts(
        (sample["text"] for sample in traindata), tokenizer, nsamples * seqlen
    )
    trainloader = _sample_from_token_buffer(trainenc, nsamples, seqlen, seed)

    valenc = None
    return trainloader, valenc


def get_metamathqa_math_500(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset("ShuoZheLi/MetaMathQA-math-500", split="test")
    texts = (_metamathqa_text(row, include_response=True) for row in dataset)
    token_buffer = _build_token_buffer_from_texts(texts, tokenizer, nsamples * seqlen)
    trainloader = _sample_from_token_buffer(token_buffer, nsamples, seqlen, seed)

    eval_texts = [_metamathqa_text(row, include_response=False) for row in dataset]
    valenc = TokenizerWrapper(tokenizer("\n\n".join(eval_texts), return_tensors="pt").input_ids)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if name in {"metamathqa_math_500", "MetaMathQA-math-500", "math_500"}:
        return get_metamathqa_math_500(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported dataset name: {name}")
