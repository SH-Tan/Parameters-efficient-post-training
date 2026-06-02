import torch


class WrappedGPT:
    def __init__(self, layer):
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=layer.weight.device)
        self.nsamples = 0

    def add_batch(self, inp, _):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        inp = inp.reshape((-1, inp.shape[-1]))
        batch_size = inp.shape[0]
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        self.scaler_row += torch.norm(inp, p=2, dim=0) ** 2 / self.nsamples


def layer_forward(model, layer, hidden_states, attention_mask, position_ids):
    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    if position_ids is not None and hasattr(model.model, "rotary_emb"):
        cos, sin = model.model.rotary_emb(hidden_states.unsqueeze(0), position_ids)
        kwargs["position_embeddings"] = (cos, sin)
    return layer(hidden_states.unsqueeze(0), **kwargs)[0]
