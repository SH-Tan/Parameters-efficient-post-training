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
    squeeze_output = hidden_states.dim() == 2
    if squeeze_output:
        hidden_states = hidden_states.unsqueeze(0)

    kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}
    if position_ids is not None and hasattr(model.model, "rotary_emb"):
        cos, sin = model.model.rotary_emb(hidden_states, position_ids)
        kwargs["position_embeddings"] = (cos, sin)
    output = layer(hidden_states, **kwargs)[0]
    return output.squeeze(0) if squeeze_output else output


def calibration_batch_tensor(tensor, start, end, batch_size, device):
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor.to(device)
    if tensor.shape[0] == 1 and batch_size > 1:
        return tensor.expand(batch_size, *tensor.shape[1:]).to(device)
    return tensor[start:end].to(device)
