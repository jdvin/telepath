import torch
from torch import Tensor
from torch import nn


class Shift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x - x.mean(dim=-1, keepdim=True)


class Scale(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)


class Affine(nn.Module):
    def __init__(self, size, bias=True):
        super().__init__()
        self.size = size
        self.weight = torch.nn.Parameter(torch.ones(self.size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.size))
        else:
            self.register_buffer("bias", torch.zeros(self.size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor):
        # Root Mean Square Layer Normalization https://arxiv.org/abs/1910.07467
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x.to(self.weight.dtype)
