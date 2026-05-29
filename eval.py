import torch
import torch.nn as nn
import gc

from data import TokenizerWrapper
from datasets import load_dataset


def load_wikitext2_eval(tokenizer):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return TokenizerWrapper(tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids)


def eval_ppl_with_loader(model, testloader, device=torch.device("cuda:0")):
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl_test 


def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    total_nll = 0.0
    loss_fct = nn.CrossEntropyLoss()
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        total_nll += float(neg_log_likelihood.detach().cpu())

        del inputs, lm_logits, shift_logits, shift_labels, loss, neg_log_likelihood
        if torch.cuda.is_available() and i % 50 == 0:
            torch.cuda.empty_cache()

    ppl = torch.exp(torch.tensor(total_nll / (nsamples * model.seqlen)))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ppl.item()
