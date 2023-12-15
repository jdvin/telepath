import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        proj_bias: bool,
        block_size: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        flash: bool = True,
    ):
        super().__init__()
        # Model embedding is split across heads.
        # Functionally equivalent to having a smaller `d_model` which is duplicated across heads.
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.is_causal = is_causal
        # Batched projection from input (B x T x D)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=proj_bias)
        # Output projection.
        self.out_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.flash = flash
        if not self.flash:
            bias = torch.ones(block_size, block_size)
            if self.is_causal:
                bias = torch.tril(bias)
            self.register_buffer("bias", bias.view(1, 1, block_size, block_size))

    def split_heads(self, x: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
        """Split matrices into heads and reshape to have heads as child ranks."""
        return x.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, qkv_expert: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.size()  # Batch size, sequence length, model dimension.

        # Project x into the query, key and value matrices, then split along the extended dimension.
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)

        # Concatenate the core and expert query, key and value matrices.
        if qkv_expert is not None:
            q_expert, k_expert, v_expert = qkv_expert.split(self.d_model, dim=-1)
            q = torch.cat([q, q_expert], dim=1)
            k = torch.cat([k, k_expert], dim=1)
            v = torch.cat([v, v_expert], dim=1)

        q = self.split_heads(q, B, T, D)
        k = self.split_heads(k, B, T, D)
        v = self.split_heads(v, B, T, D)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
            )
        else:
            # (B, nhead, T, D_head) x (B, nhead, D_head, T) -> (B, nhead, T, T).
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Fill the upper triangle.
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            # (B, nhead, T, T) x (B, nhead, T, D_head) -> (B, nhead, T, D_head).
            y = attn @ v
        # Flatten heads.
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        y = self.out_proj(y)

        y = self.resid_dropout(y)

        return y


class ExpertAttention(MultiheadAttention):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        core_proj_bias: bool,
        expert_proj_bias: bool,
        core_block_size: int,
        expert_block_size: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        flash: bool = True,
    ):
        super().__init__(
            n_heads,
            d_model,
            core_proj_bias,
            expert_block_size + core_block_size,
            dropout,
            is_causal,
            flash,
        )
        self.expert_block_size = expert_block_size
        self.non_expert_block_size = non_expert_block_size
        self.expert_qkv_proj = nn.Linear(d_model, 3 * d_model, bias=proj_bias)

    def forward(x: torch.Tensor):
        # Perform the expert emebdding projection separately.
        qkv_expert = self.expert_qkv_proj(x[:, :expert_block_size, :])
        # Concatenate with what will become the core embedding projection.
        return super().forward(x[:, expert_block_size:, :], qkv_expert)
