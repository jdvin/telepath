from torch import nn, Tensor
from torch.nn import functional as F


class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=bias)
        self.V = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.W(x)) * self.V(x)
