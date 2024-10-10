import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .pos import RelativePositionBias


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        source_seq_len: int,
        target_seq_len: int,
        q_bias: bool = True,
        k_bias: bool = False,
        v_bias: bool = True,
        out_bias: bool = True,
        scale: float = 0.0,
        dropout: float = 0.1,
        is_causal: bool = False,
        flash: bool = True,
    ):
        super().__init__()
        # Model embedding is split across heads.
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.is_causal = is_causal
        assert scale
        self.scale = scale
        # The projectons are scalled differently in the original whisper implementation
        self.q_proj = nn.Linear(d_model, d_model, bias=q_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=k_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=v_bias)
        # Output projection.
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = flash
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        if self.is_causal:
            bias = torch.triu(
                torch.full(
                    (1, 1, target_seq_len, source_seq_len),
                    torch.finfo(self.q_proj.weight.dtype).min,
                ),
                diagonal=1,
            )
            self.register_buffer("bias", bias)

    def split_heads(self, x: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
        """Split matrices into heads and reshape to have heads as child ranks."""
        return x.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """
        Args:
            q: Tensor[float] (B, nhead, T_q, D_head)
            k: Tensor[float] (B, nhead, T_kv, D_head)
            v: Tensor[float] (B, nhead, T_kv, D_head)
            attention_mask: Tensor[float] (B, 1, T_q, T_kv)
        """
        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0,
                scale=self.scale,
            )

        else:
            # (B, nhead, T_q, D_head) x (B, nhead, D_head, T_kv) -> (B, nhead, T_q, T_kv).
            qk = (q @ k.transpose(-2, -1)) * self.scale
            # Add attention bias (input masking, causal masking, relative pos, etc).
            qk = qk + attention_mask if attention_mask is not None else qk
            attn = F.softmax(qk, dim=-1, dtype=torch.float32).type_as(qk)
            attn = self.attn_dropout(attn) if self.training else attn
            # (B, nhead, T, T) x (B, nhead, T, D_head) -> (B, nhead, T, D_head).
            y = attn @ v
        return y

    def forward(
        self,
        x: Tensor,
        xc: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor]:
        B, T_q, D = x.size()  # Batch size, sequence length, model dimension.
        T_kv = xc.size(1) if xc is not None else T_q
        # Instantiate a 'dummy' kv cache to make the logic simpler.
        if kv_cache is None:
            kv_cache = {}

        k = kv_cache.get(hash(self.k_proj), None)
        v = kv_cache.get(hash(self.v_proj), None)
        T_cached = k.size(1) if k is not None else 0
        if k is None or k.size(1) != T_kv:
            k_new = self.k_proj((x if xc is None else xc)[:, T_cached:, :])
            v_new = self.v_proj((x if xc is None else xc)[:, T_cached:, :])
            kv_cache[self.k_proj] = (
                k_new if k is None else torch.concat([k, k_new], dim=1)
            )
            kv_cache[self.v_proj] = (
                v_new if v is None else torch.concat([v, v_new], dim=1)
            )

        q = self.q_proj(x)
        k = kv_cache[self.k_proj]
        v = kv_cache[self.v_proj]
        q = self.split_heads(q, B, T_q, D)
        k = self.split_heads(k, B, T_kv, D)
        v = self.split_heads(v, B, T_kv, D)
        bias = (
            self.bias[:, :, T_cached : T_cached + T_q, :T_kv]
            if self.is_causal
            else None
        )
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(
                B, self.n_heads, T_kv, T_q
            )
            if bias is not None:
                attention_mask = attention_mask + bias
        y = self.qkv_attention(
            q,
            k,
            v,
            bias if attention_mask is None else attention_mask,
        )
        # Flatten heads.
        y = y.transpose(1, 2).contiguous().view(B, T_q, D)
        y = self.out_proj(y)
        return self.resid_dropout(y) if self.training else y


class RelativePositionMultiHeadAttention(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rp_bias = RelativePositionBias(bidirectional=False, n_heads=self.n_heads)
        assert (
            self.source_seq_len == self.target_seq_len
        ), "Relative position MHA can only be used in self-attention!"
        if not self.is_causal:
            raise NotImplementedError(
                "This implementation won't work with non-causal attention."
            )
        self.compute_bias()
        self.register_load_state_dict_post_hook(self._post_load_hook)

    def compute_bias(self):
        bias = self.rp_bias(self.target_seq_len, self.source_seq_len)
        if self.is_causal:
            bias = bias + torch.triu(
                torch.full(
                    (1, 1, self.target_seq_len, self.source_seq_len),
                    torch.finfo(self.rp_bias.relative_attention_bias.weight.dtype).min,
                ),
                diagonal=1,
            )
        self.bias = bias

    def _post_load_hook(self, module, incompatible_keys):
        # This method will be called after the state dict is loaded
        self.compute_bias()
