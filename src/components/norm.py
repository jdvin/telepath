import torch
from torch import nn


class Shift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x - x.mean(dim=-1, keepdim=True)


class Scale(nn.Module):
    def __init__(self, eps=1e-6):
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


class LayerNorm(nn.Module):
    """Layernorm with composability."""

    def __init__(
        self, size: int, shift=True, scale=True, eps=1e-5, affine=True, bias=True
    ):
        super().__init__()
        graph = []
        if shift:
            graph.append(Shift())
        if scale:
            graph.append(Scale(eps))
        if affine:
            graph.append(Affine(size, bias))

        self.graph = nn.Sequential(*graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Taken from formula at https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html."""
        return self.graph.forward(x.float()).type(x.dtype)
