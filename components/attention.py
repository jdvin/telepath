import torch
import math


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        self.d_k = d_k
        self.q_linear = torch.nn.Linear(d_model, d_k)
        self.k_linear = torch.nn.Linear(d_model, d_k)
        self.v_linear = torch.nn.Linear(d_model, d_v)

    def forward(self, q, k, v):

        # batch_size = q.size(0)

        q = self.q_linear(q)  # .view(batch_size, -1, self.d_k)
        k = self.k_linear(k)
        v = self.v_linear(v)

        energies = torch.matmul(self.q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        self.attention_weights = torch.nn.functional.softmax(energies, -1)
        output = torch.matmul(self.attention_weights, v)

        return output


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, n_heads: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.h = n_heads

        self.heads = torch.nn.ModuleList(
            [SelfAttention(d_model=d_model, d_k=d_k, d_v=d_v) for _ in n_heads]
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k, v):
        """Forward pass on all attention heads.

        Args:
            q: A (b x t x d) tensor to generate the query matrices of the attention heads.
            k: A (b x t x d) tensor to generate the key matrices of the attention heads.
            v: A (b x t x d) tensor to generate the value matrices of the attention heads.

        Note:
            b = Batch size.
            t = Sequence length.
            d = Model depth.

        Returns:
            _type_: _description_
        """

        # Nest the dimension [1] of the q, k, and v tensors (resulting: b x 1 x t x d tensor) and then repeat them for each head (resulting: b x h x t d tensor).
        Q = torch.unsqueeze(q, 1).repeat(1, self.h, 1, 1)
        K = torch.unsqueeze(k, 1).repeat(1, self.h, 1, 1)
        V = torch.unsqueeze(v, 1).repeat(1, self.h, 1, 1)

        # Linearly transform each tensor.

        # Calculate dot-product self attention for each head.

        # Concatenate heads and put through final linear layer
        # concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        # output = self.out(concat)

        # return output
