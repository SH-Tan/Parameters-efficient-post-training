import gc

import torch
import torch.nn as nn
from datasets import load_dataset

from eval.data import TokenizerWrapper, _metamathqa_text, _pad_token_id, get_c4_eval, get_loaders


def _loader_pad_token_id(testloader):
    return getattr(testloader, "pad_token_id", None)


def _left_pad_testenc(testenc, seqlen, pad_token_id):
    remainder = testenc.numel() % seqlen
    if remainder == 0:
        return testenc

    pad_len = seqlen - remainder
    padding = torch.full((testenc.shape[0], pad_len), int(pad_token_id), dtype=testenc.dtype)
    return torch.cat((padding, testenc), dim=1)


def load_wikitext2_eval(tokenizer):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return TokenizerWrapper(
        tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids,
        pad_token_id=_pad_token_id(tokenizer),
    )


def load_ppl_eval_data(name, tokenizer, nsamples, seed, seqlen):
    if "wikitext2" in name:
        return load_wikitext2_eval(tokenizer)
    if name in {"c4_test", "c4_validation"}:
        print("using deterministic spread sample of 3000 rows from allenai/c4 validation for PPL eval")
        return get_c4_eval(seqlen=seqlen, tokenizer=tokenizer)
    if name in {"metamathqa_math_500", "MetaMathQA-math-500", "math_500"}:
        eval_dataset = load_dataset("ShuoZheLi/MetaMathQA-math-500", split="test")
        eval_text = "\n\n".join(_metamathqa_text(row, include_response=False) for row in eval_dataset)
        return TokenizerWrapper(
            tokenizer(eval_text, return_tensors="pt").input_ids,
            pad_token_id=_pad_token_id(tokenizer),
        )

    _, eval_data = get_loaders(name, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
    if eval_data is None:
        raise ValueError(f"No perplexity eval split is implemented for pp_eval_data={name!r}")
    return eval_data


def eval_ppl_with_loader(model, testloader, device=torch.device("cuda:0")):
    with torch.no_grad():
        if hasattr(testloader, "iter_input_ids"):
            ppl_test = eval_ppl_streaming(model, testloader, device)
        else:
            ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl_test 


def eval_ppl_streaming(model, testloader, device=None):
    total_nll = 0.0
    total_tokens = 0
    pad_token_id = _loader_pad_token_id(testloader)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    testloader.seqlen = int(model.seqlen)

    for index, inputs in enumerate(testloader.iter_input_ids()):
        if index % 50 == 0:
            print(f"sample {index}")

        inputs = inputs.to(device)
        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:].clone()
        if pad_token_id is not None:
            shift_labels[shift_labels == pad_token_id] = -100

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        token_count = int((shift_labels != -100).sum().item())
        if token_count == 0:
            continue
        neg_log_likelihood = loss.float() * token_count

        total_nll += float(neg_log_likelihood.detach().cpu())
        total_tokens += token_count

        del inputs, lm_logits, shift_logits, shift_labels, loss, neg_log_likelihood
        if torch.cuda.is_available() and index % 50 == 0:
            torch.cuda.empty_cache()

    if total_tokens == 0:
        raise ValueError("No full sequences were available for streaming PPL evaluation.")

    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl.item()


def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    pad_token_id = _loader_pad_token_id(testenc)
    testenc = testenc.input_ids
    if pad_token_id is not None:
        testenc = _left_pad_testenc(testenc, model.seqlen, pad_token_id)
    nsamples = testenc.numel() // model.seqlen

    total_nll = 0.0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:].clone()
        if pad_token_id is not None:
            shift_labels[shift_labels == pad_token_id] = -100

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        token_count = int((shift_labels != -100).sum().item())
        if token_count == 0:
            continue
        neg_log_likelihood = loss.float() * token_count

        total_nll += float(neg_log_likelihood.detach().cpu())
        total_tokens += token_count

        del inputs, lm_logits, shift_logits, shift_labels, loss, neg_log_likelihood
        if torch.cuda.is_available() and i % 50 == 0:
            torch.cuda.empty_cache()

    if total_tokens == 0:
        raise ValueError("No tokens were available for PPL evaluation.")

    ppl = torch.exp(torch.tensor(total_nll / total_tokens))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ppl.item()
