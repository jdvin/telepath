import typing as _t

import torch
import numpy as np


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, n_heads: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = torch.nn.Parameter(torch.empty((n_heads, d_model, d_k)))
        self.W_K = torch.nn.Parameter(torch.empty((n_heads, d_model, d_k)))
        self.W_V = torch.nn.Parameter(torch.empty((n_heads, d_model, d_k)))
        self.W_O = torch.nn.Parameter(torch.empty((n_heads * d_v, d_model)))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> _t.Tuple[torch.tensor]:
        """Forward pass on all attention heads.

        Args:
            q: A (b x t x d) tensor to generate the query matrices of the attention heads.
            k: A (b x t x d) tensor to generate the key matrices of the attention heads.
            v: A (b x t x d) tensor to generate the value matrices of the attention heads.

        Note:
            b = Batch size.
            t = Sequence length.
            d = Model depth.
        """

        # Get the batch size from the input.
        b = q.shape[0]

        # Nest the dimension [1] of the q, k, and v tensors (resulting: b x 1 x t x d tensor)
        # and then repeat them for each head (resulting: b x h x t x d tensor).
        Q = torch.unsqueeze(q, 1).repeat(1, self.h, 1, 1)
        K = torch.unsqueeze(k, 1).repeat(1, self.h, 1, 1)
        V = torch.unsqueeze(v, 1).repeat(1, self.h, 1, 1)

        # Linearly transform each tensor query and key tensor -> (b x h x t x d_k).
        Q = torch.matmul(Q, self.W_Q)
        K = torch.matmul(K, self.W_K)

        # Linearly transform each value tensor -> (b x h x t x d_v).
        V = torch.matmul(V, self.W_V)

        # Calculate dot-product self attention for each head.
        energies = torch.divide(
            torch.matmul(Q, K.transpose(-1, -2)),
            np.sqrt(self.d_k),
        )  # b x h x t x t

        attention_weights = torch.nn.functional.softmax(energies, -1)  # b x h x t x d_k
        scores = torch.matmul(attention_weights, V)  # b x h x t x d_v

        # Transpose and concatenate to get a (b x h*d_v x t) tensor.
        concat = scores.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_v)

        # Run the output through the final linear transformation.
        output = torch.matmul(concat, self.W_O)

        # Return output and attention weights.
        return output, attention_weights
