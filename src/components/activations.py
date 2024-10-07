import math
from torch import nn, Tensor, tanh, pow
from torch.nn import functional as F


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * pow(input, 3.0)))
            )
        )


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
        self.gelu = NewGELUActivation()

    def forward(self, x: Tensor) -> Tensor:
        return self.gelu(self.W(x)) * self.V(x)
