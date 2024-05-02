from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import yaml

from .components.attention import MultiHeadAttention
from .components.norm import LayerNorm

from transformers import WhisperModel


@dataclass
class TelepathConfig:
    pretrained_whisper: str
    pretrained_gpt: str
    freeze_gpt: bool
    gpt_start_token: int
    gpt_stop_token: int
    gpt_block_size: int
    gpt_dropout: float

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


def sinusoids(length: int, channels: int, max_timescale: int = 1000):
    """Returns sinusoids for positional embedding

    Taken from Whisper implementation: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        block_size: int,
        d_model: int,
        n_heads: int,
        cross_attn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_heads=n_heads,
            d_model=d_model,
            scale=(d_model // n_heads) ** -0.25,
            k_bias=True,
            block_size=block_size,
            dropout=dropout,
        )
        self.attn_ln = LayerNorm(d_model)

        self.cross_attn = (
            MultiHeadAttention(
                n_heads, d_model, k_bias=True, block_size=block_size, dropout=dropout
            )
            if cross_attn
            else None
        )
        self.cross_attn_ln = LayerNorm(d_model) if cross_attn else None

        d_mlp = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
        self.mlp_ln = LayerNorm(d_model)

    def forward(self, x: Tensor, kv_cache: dict[int, Tensor] | None = None) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        if self.cross_attn and self.cross_attn_ln:
            x = x + self.cross_attn(self.cross_attn_ln(x), kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class NeuralEncoder(nn.Module):
    def __init__(
        self,
        n_freqs: int,
        block_size: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        n_layers: int,
    ):
        super().__init__()

        # We want the convolutions to be performed separately on each eletrode channel.
        # The channels will be stacked across the height dimension.
        self.conv1 = nn.Conv2d(
            in_channels=n_freqs,
            out_channels=d_model,
            kernel_size=(1, 3),
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=(1, 3),
            stride=(1, 2),
            padding=1,
        )
        self.register_buffer("embed_positions", sinusoids(block_size, d_model))
        self.embed_electrodes = nn.Embedding(n_freqs, d_model)

        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(block_size, d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        self.ln_post = LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, n_ee_channels, n_freqs, sequence_length) -> (batch_size, n_freqs, n_eeg_channels, sequence_length).
        # We want the convolutions to be performed separately on each eletrode channel.
        # The inputs to a convolution 2d are of the shape (N, C_in, H, W).
        B, N_C, N_F, T = x.size()
        x = x.reshape(B, N_F, N_C, T)

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = (x + self.embed_positions).to(x.dtype)
        x = x + self.embed_electrodes.weight
        # Stack the electrode embeddings across the time dimension.
        x = x.reshape(B, N_F, N_C * T)
        for block in self.blocks:
            x = block(x)
        return x

    def optim_groups(self, weight_decay: float = 1e-1) -> list[dict[str, str]]:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        return optim_groups

    @classmethod
    def from_pretrained(cls, model_id: str):
        pretrained_model = WhisperModel.from_pretrained(model_id)
        assert isinstance(pretrained_model, WhisperModel)
        pretrained_model = pretrained_model.encoder


class Telepath(nn.Module):
    def __init__(self, config: TelepathConfig):
        super().__init__()
        self.config = config

        self.pre_norm = norm.LayerNorm(
            size=config.n_channels,
            shift=config.pre_norm_shift,
            scale=config.pre_norm_scale,
            affine=config.pre_norm_affine,
            bias=config.pre_norm_bias,
        )

        self.encoder = NeuralWhisperEncoder.from_pretrained()

        self.decoder = ExpertGPT.from_pretrained(
            model_type=config.pretrained_gpt,
            expert_block_size=config.expert_encoder_block_size,
        )
        self.decoder.crop_block_size(config.gpt_block_size)

        self.start_token = config.gpt_start_token
        self.stop_token = config.gpt_stop_token

    def forward(self, eeg: Tensor, input_ids: Tensor) -> Tensor:
        """Forward pass through the Telepath model.

        Attributes:
            eeg_signal: EEG signal of shape (batch_size, n_samples, n_channels).
            input_ids: Input token ids of shape (batch_size, n_tokens).
        """
        enc = self.pre_norm(eeg)
        enc = self.encoder_proj(enc)
        enc = self.encoder(enc)
        return self.decoder.forward(input_ids, embed=enc)

    @torch.no_grad()
    def generate(
        self, eeg: Tensor, device: str, stop_token: int | None = None
    ) -> list[list[int]]:
        """Generate a sequence of tokens given an EEG signal.
        Attributes:
            eeg_signal: EEG signal of shape (batch_size, n_samples, n_channels).
            stop_token: Token id to stop generation at.
        """
        assert len(eeg.size()) == 3
        batch_size = eeg.size(0)
        eeg = eeg.to(device)
        enc = self.pre_norm(eeg)
        enc = self.encoder_proj(enc)
        enc = self.encoder(enc)
        return self.decoder.generate(
            input_ids=torch.full((batch_size, 1), self.start_token).to(device),
            embed=enc,
            stop_token=stop_token or self.stop_token,
        )
