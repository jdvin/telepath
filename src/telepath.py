from dataclasses import dataclass
from json import encoder

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import yaml

from .gpt import GPT, ExpertGPT
from .components import attention, norm


@dataclass
class TelepathConfig:
    n_channels: int
    pre_norm_shift: bool
    pre_norm_scale: bool
    pre_norm_affine: bool
    pre_norm_bias: bool

    expert_encoder_block_size: int
    expert_encoder_d_model: int
    expert_encoder_n_heads: int
    expert_encoder_n_layers: int
    expert_encoder_bias: bool
    expert_encoder_dropout: float

    tokenizer_path: str
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


class WaveNetEncoder(nn.Module):
    pass


def sinusoids(length: int, channels: int, max_timescale: int):
    """Returns sinusoids for positional embedding

    Taken from Whisper implementation: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AttentionEncoderBlock(nn.Module):
    def __init__(
        self, block_size: int, d_model: int, n_heads: int, bias: bool, dropout: float
    ):
        super().__init__()
        self.ln_1 = norm.LayerNorm(d_model, affine=True, bias=bias)
        self.attn = attention.MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model,
            proj_bias=bias,
            block_size=block_size,
            dropout=dropout,
            is_causal=False,
            flash=True,
        )
        self.ln_2 = norm.LayerNorm(
            d_model,
            affine=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NeuralEncoder(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        block_size: int,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        n_layers: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_input_channels, out_channels=d_model, kernel_size=3
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=3, dilation=2
        )

        self.blocks = nn.ModuleList(
            AttentionEncoderBlock(block_size, d_model, n_heads, bias, dropout)
            for _ in range(n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
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

        self.encoder = NeuralEncoder(
            n_input_channels=config.n_channels,
            block_size=config.expert_encoder_block_size,
            d_model=config.expert_encoder_d_model,
            n_heads=config.expert_encoder_n_heads,
            bias=config.expert_encoder_bias,
            dropout=config.expert_encoder_dropout,
            n_layers=config.expert_encoder_n_layers,
        )

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
