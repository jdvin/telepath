from dataclasses import dataclass
from json import encoder

import torch
from torch import nn
import yaml

from .gpt import GPT
from .components import attention, norm


@dataclass
class TelepathConfig:
    n_channels: int
    pre_norm_shift: bool
    pre_norm_scale: bool
    pre_norm_affine: bool
    pre_norm_bias: bool

    encoder_block_size: int
    encoder_d_model: int
    encoder_n_heads: int
    encoder_n_layers: int
    encoder_bias: bool
    encoder_dropout: float

    pretrained_gpt: GPT | None = None
    freeze_gpt: bool = True
    gpt_start_token: int = 3784
    gpt_stop_token: int = 50256
    gpt_n_heads: int = 12
    gpt_d_model: int = 768
    gpt_bias: bool = True
    gpt_n_layers: int = 12
    gpt_vocab_size: int = 50257
    gpt_block_size: int = 110
    gpt_dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


class WaveNetEncoder(nn.Module):
    pass


class AttentionEncoderBlock(nn.Module):
    # TODO: Should probably be recurrent to take advantage of EEG signal from prior objects..
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        block_size: int,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        n_layers: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            AttentionEncoderBlock(block_size, d_model, n_heads, bias, dropout)
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
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

        self.encoder_proj = nn.Linear(
            config.n_channels, config.encoder_d_model, bias=config.encoder_bias
        )

        self.encoder = AttentionEncoder(
            block_size=config.encoder_block_size,
            d_model=config.encoder_d_model,
            n_heads=config.encoder_n_heads,
            bias=config.encoder_bias,
            dropout=config.encoder_dropout,
            n_layers=config.encoder_n_layers,
        )

        if config.pretrained_gpt:
            self.decoder = GPT.from_pretrained(model_type=config.pretrained_gpt)
            self.decoder.crop_block_size(config.gpt_block_size)
        else:
            self.decoder = GPT(
                n_heads=config.gpt_n_heads,
                d_model=config.gpt_d_model,
                bias=config.gpt_bias,
                n_layers=config.gpt_n_layers,
                vocab_size=config.gpt_vocab_size,
                block_size=config.gpt_block_size,
                dropout=config.gpt_dropout,
            )

        self.start_token = config.gpt_start_token
        self.stop_token = config.gpt_stop_token

    def forward(self, eeg: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Telepath model.

        Attributes:
            eeg_signal: EEG signal of shape (batch_size, n_samples, n_channels).
            input_ids: Input token ids of shape (batch_size, n_tokens).
        """
        enc = self.pre_norm(eeg)
        enc = self.encoder_proj(enc)
        enc = self.encoder(enc)
        return self.decoder.forward(input_ids, concat_embed=enc)

    @torch.no_grad()
    def generate(self, eeg: torch.Tensor, device: str, stop_token: int | None = None) -> list[list[int]]:
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
            concat_embed=enc,
            stop_token=stop_token or self.stop_token,
        )
