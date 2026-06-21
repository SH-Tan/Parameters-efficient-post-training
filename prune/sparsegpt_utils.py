import math

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


class SparseGPT:
    def __init__(self, layer, hessian_chunk_size=8192):
        self.layer = layer
        self.dev = self.layer.weight.device
        weight = self._weight_2d()
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.hessian_chunk_size = int(hessian_chunk_size)

    def _weight_2d(self):
        weight = self.layer.weight.data
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, Conv1D):
            weight = weight.t()
        return weight

    def add_batch(self, inp, _):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, (nn.Linear, Conv1D)):
            if inp.dim() == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        scale = math.sqrt(2 / self.nsamples)
        chunk_size = self.hessian_chunk_size
        if chunk_size <= 0:
            inp = scale * inp.float()
            self.H += inp.matmul(inp.t())
            return
        for start in range(0, inp.shape[1], chunk_size):
            inp_chunk = scale * inp[:, start:start + chunk_size].float()
            self.H += inp_chunk.matmul(inp_chunk.t())
            del inp_chunk

    def _stable_cholesky(self, hessian, base_damp, max_tries=8):
        diag = torch.arange(self.columns, device=self.dev)
        eye_increment = torch.as_tensor(base_damp, device=self.dev, dtype=hessian.dtype)
        if not torch.isfinite(eye_increment) or eye_increment <= 0:
            eye_increment = torch.tensor(1e-6, device=self.dev, dtype=hessian.dtype)
        for attempt in range(max_tries):
            try:
                return torch.linalg.cholesky(hessian)
            except torch._C._LinAlgError as error:
                if attempt == max_tries - 1:
                    raise
                hessian[diag, diag] += eye_increment
                print(
                    f"SparseGPT Hessian Cholesky failed ({error}); "
                    f"adding diagonal jitter={float(eye_increment):.6e} and retrying",
                    flush=True,
                )
                eye_increment *= 10

    def score(self, percdamp=0.01):
        weight = self._weight_2d().clone().float()
        hessian = self.H.clone().float()
        hessian = 0.5 * (hessian + hessian.t())
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        diag_values = torch.diag(hessian)
        diag_mean = torch.mean(diag_values[torch.isfinite(diag_values)])
        if not torch.isfinite(diag_mean) or diag_mean <= 0:
            diag_mean = torch.tensor(1.0, device=self.dev)
        damp = float(percdamp) * diag_mean
        diag = torch.arange(self.columns, device=self.dev)
        hessian[diag, diag] += damp
        hessian = self._stable_cholesky(hessian, damp)
        hessian = torch.cholesky_inverse(hessian)
        hessian = 0.5 * (hessian + hessian.t())
        hessian = self._stable_cholesky(hessian, damp, max_tries=4).t()

        diag_hinv = torch.diag(hessian).reshape((1, -1))
        metric = weight.pow(2) / diag_hinv.pow(2)
        if isinstance(self.layer, Conv1D):
            metric = metric.t()
        return metric.reshape(self.layer.weight.shape).detach().cpu()

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
