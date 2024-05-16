from dataclasses import dataclass
from typing import List, Sequence, Iterable

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import yaml

from .components.attention import MultiHeadAttention
from .components.norm import LayerNorm

from transformers import WhisperModel, WhisperConfig


@dataclass
class TelepathConfig:
    n_eeg_channels: int
    pretrained_whisper: str
    decoder_start_sequence: Tensor
    decoder_stop_token: int
    decoder_special_tokens_start: int
    decoder_vocab_size: int
    encoder_block_size: int
    decoder_block_size: int
    n_freqs: int
    d_model: int
    n_heads: int
    encoder_n_layers: int
    decoder_n_layers: int
    dropout = 0.1

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def __post_init__(self):
        if not self.pretrained_whisper:
            assert (
                self.n_freqs
                and self.d_model
                and self.n_heads
                and self.decoder_vocab_size
                and self.encoder_n_layers
                and self.decoder_n_layers
                and self.decoder_vocab_size
                and self.decoder_start_sequence
                and self.decoder_stop_token
                and self.decoder_special_tokens_start
            )
            return
        pt_whisper_config = WhisperConfig.from_pretrained(self.pretrained_whisper)
        tokenizer = WhisperModel.from_pretrained(
            self.pretrained_whisper, task="translate", language="english"
        )
        assert isinstance(tokenizer, WhisperModel)
        self.decoder_vocab_size = pt_whisper_config.vocab_size
        self.n_freqs = pt_whisper_config.n_freqs
        self.d_model = pt_whisper_config.d_model
        self.n_heads = self.n_heads or pt_whisper_config.n_heads
        self.encoder_n_layers = (
            self.encoder_n_layers or pt_whisper_config.encoder_n_layers
        )
        self.decoder_n_layers = (
            self.decoder_n_layers or pt_whisper_config.decoder_n_layers
        )
        assert isinstance(tokenizer.prefix_tokens, list)
        self.decoder_start_sequence = torch.tensor(tokenizer.prefix_tokens)
        assert isinstance(tokenizer.eos_token_id, int)
        self.decoder_stop_token = tokenizer.eos_token_id
        self.decoder_special_tokens_start = min(tokenizer.added_tokens_decoder.keys())


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


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, d_model: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(n_vocab, d_model)
        self.embed_positions = nn.Parameter(torch.zeros(n_ctx, d_model))
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(n_ctx, d_model, n_head, cross_attn=True)
            for _ in range(n_layer)
        )
        assert isinstance(self.blocks[0], ResidualAttentionBlock)
        assert isinstance(self.blocks[0].attn, MultiHeadAttention)
        assert isinstance(self.blocks[0].attn.k_proj, nn.Linear)
        self.ln_post = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        xc: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        inference: bool = False,
    ) -> Tensor:
        offset = kv_cache[next(iter(kv_cache.values()))].size(0) if kv_cache else 0
        x = self.embed_tokens(x) + self.embed_positions[offset : offset + x.size(1)]
        for block in self.blocks:
            x = block(x, xc, kv_cache=kv_cache)
        x = self.ln_post(x)
        if inference:
            x = x[:, -1, :]
        return x @ self.embed_tokens.weight.t()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        embed: torch.Tensor | None = None,
        max_length: int = 10,
        stop_token: int = 50256,
    ) -> list[list[int]]:
        """Generate a sequence of tokens using argmax sampling.

        Optimised for batch generation; pops generations off the inference stack as they complete.
        Attributes:
            input_ids: Input token ids of shape (batch_size, n_tokens).
            concat_embed: Additional embeddings used by the model forward method. Must be of size (batch_size, k, d_model).
            max_length: Maximum length of the generated sequence.
            stop_token: Token id of the stop token.
        """
        kv_cache = {}
        batch_size = input_ids.size(0) if len(input_ids.size()) == 2 else 1
        # Preallocate generations so that we can save them by order.
        generations = [[] for _ in range(batch_size)]
        # Used to track the indexes of the running generations.
        generating_batch_indexes = list(range(batch_size))
        for _ in range(max_length):
            logits = self.forward(input_ids, embed, inference=True, kv_cache=kv_cache)
            input_ids = torch.cat([input_ids, logits.argmax(dim=-1)], dim=1)
            # Get the index of every generation that has generated the stop token.
            stop_indexes = (
                (input_ids[:, -1] == stop_token).nonzero(as_tuple=True)[0].tolist()
            )
            while stop_indexes:
                # Remove the running generations that have generated the stop token.
                i = stop_indexes.pop()
                # Get the batch position of the current generation and remove it from the list of those currently generating.
                batch_index = generating_batch_indexes.pop(i)
                # Map from current relative index to batch index.
                generations[batch_index] = input_ids[i, :].tolist()
                input_ids = torch.cat([input_ids[:i, :], input_ids[i + 1 :, :]], dim=0)
                if embed is not None:
                    embed = torch.cat([embed[:i, :, :], embed[i + 1 :, :, :]], dim=0)
                # Shift the indexes after the one just removed.
                stop_indexes = [(j - 1) if j > i else j for j in stop_indexes]

            if len(generating_batch_indexes) == 0:
                break
        # If there are still running generations, add them to the list.
        for i, batch_index in enumerate(generating_batch_indexes):
            generations[batch_index] = input_ids[i, :].tolist()
        return generations


class Telepath(nn.Module):
    def __init__(self, config: TelepathConfig):
        super().__init__()
        self.config = config

        self.encoder = NeuralEncoder(
            n_freqs=config.n_freqs,
            block_size=config.decoder_block_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_n_layers,
            dropout=config.dropout,
        )

        self.decoder = TextDecoder(
            n_vocab=config.decoder_vocab_size,
            n_ctx=config.decoder_block_size,
            d_model=config.d_model,
            n_head=config.n_heads,
            n_layer=config.decoder_n_layers,
        )

        self.start_sequence = config.decoder_start_sequence
        self.stop_token = config.decoder_stop_token

    def forward(self, eeg: Tensor, input_ids: Tensor) -> Tensor:
        """Forward pass through the Telepath model.

        Attributes:
            eeg_signal: EEG signal of shape (batch_size, n_samples, n_channels).
            input_ids: Input token ids of shape (batch_size, n_tokens).
        """
        return self.decoder(input_ids, xc=self.encoder(eeg))

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
        B = eeg.size(0)
        eeg = eeg.to(device)
        enc = self.encoder(eeg)
        return self.decoder.generate(
            input_ids=self.start_sequence.repeat(B, 1).detach().to(device),
            embed=enc,
            stop_token=stop_token or self.stop_token,
        )

    @classmethod
    def from_pretrained(cls, config: TelepathConfig):
        ptw = WhisperModel.from_pretrained(config.pretrained_whisper)
        assert isinstance(ptw, WhisperModel)
        param_map = {
            "layers": "blocks",
            "self_attn": "attn",
            "self_attn_layer_norm": "attn_ln.graph.2",
            "fc1": "mlp.0",
            "fc2": "mlp.2",
            "final_layer_norm": "mlp_ln.graph.2",
            "layer_norm": "ln_post.graph.2",
        }
        map_params = lambda pn: ".".join(
            [param_map.get(seg, seg) for seg in pn.split(".")]
        )
        nw = cls(config)
        nw_sd = nw.state_dict()
        for key, param in ptw.state_dict().items():
            new_key = map_params(key)
            # We are moving from a 1D conv to a 2D conv, but we want the conv to be the same for each eletrode.
            # NOTE: Do we want to have unique params for each electrode?
            # Could then initialize from the same weights but have them learn different filters.
            if "conv" in key:
                param = param.unsqueeze(2)

            nw_sd[new_key] = param
        nw.load_state_dict(nw_sd)
        return nw
