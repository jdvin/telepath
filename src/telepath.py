from dataclasses import dataclass
from typing import List, Sequence, Iterable

import numpy as np
import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml

from src.components.norm import RMSNorm

from .components.attention import MultiHeadAttention, RelativePositionMultiHeadAttention
from .components.activations import GEGLU

from transformers import (
    AutoTokenizer,
    WhisperModel,
    WhisperConfig,
    T5ForConditionalGeneration,
    T5Config,
)


@dataclass
class ParamMap:
    map: dict[str, str | None]
    carry_forward_indices: set[int]

    def __getitem__(self, key: str) -> str:
        out = self.map.get(key, key)
        assert out is not None
        return out

    def map_param(self, pn: str) -> str:
        out_pns = []
        carry = ""
        for i, seg in enumerate(pn.split(".")):
            carry += seg
            if i in self.carry_forward_indices:
                continue
            if mapped_seg := self[carry]:
                # print(carry, mapped_seg)
                out_pns.append(mapped_seg)
            carry = ""

        return ".".join(out_pns)


WHISPER_PARAM_MAP = ParamMap(
    {
        "encoder": "",
        "layers": "blocks",
        "self_attn": "attn",
        "self_attn_layer_norm": "attn_ln",  # .graph.2",
        "fc1": "mlp.0",
        "fc2": "mlp.2",
        "final_layer_norm": "mlp_ln",  # .graph.2",
        "layer_norm": "ln_post",  # .graph.2",
        "encoder_attn": "cross_attn",
        "encoder_attn_layer_norm": "cross_attn_ln",  # .graph.2",
    },
    set(),
)

T5_PARAM_MAP = ParamMap(
    {
        "decoder": "",
        "block": "blocks",
        "layer": "",
        "0SelfAttention": "attn",
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "out_proj",
        "relative_attention_bias": "rp_bias.relative_attention_bias",
        "0layer_norm": "attn_ln",
        "1EncDecAttention": "cross_attn",
        "1layer_norm": "cross_attn_ln",
        "2DenseReluDense": "mlp",
        "2layer_norm": "mlp_ln",
        "wi_0": "0.W",
        "wi_1": "0.V",
        "wo": "1",
        "final_layer_norm": "ln_post",
    },
    {3},
)


@dataclass
class TelepathConfig:
    n_eeg_channels: int
    text_encoder_pretrained_model: str | None
    text_encoder_start_sequence: Tensor = tensor([1484, 9709, 7314, 10])
    text_encoder_stop_token: int = 0
    text_encoder_vocab_size: int = 0
    neural_encoder_block_size: int = 0
    text_encoder_block_size: int = 0
    neural_encoder_spectrogram: bool = False
    n_freqs: int | None = None
    fft_hop_length: int | None = None
    d_model: int = 0
    neural_encoder_d_mlp: int = 0
    neural_encoder_activation: str = ""
    text_encoder_d_mlp: int = 0
    text_encoder_activation: str = ""
    n_heads: int = 0
    neural_encoder_n_layers: int = 0
    text_encoder_n_layers: int = 0
    dropout: float = 0.1
    neural_encoder_scale_exponent: float = -0.25
    train_text_encoder: bool = False

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
        source_seq_len: int,
        target_seq_len: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        is_causal: bool,
        activation: nn.Module = nn.GELU,
        cross_attn: bool = False,
        dropout: float = 0.0,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_heads=n_heads,
            d_model=d_model,
            scale=(d_model // n_heads) ** scale_exponent,
            source_seq_len=source_seq_len,
            target_seq_len=source_seq_len,
            dropout=dropout,
            is_causal=is_causal,
        )
        self.attn_ln = nn.LayerNorm(d_model)

        self.cross_attn = (
            MultiHeadAttention(
                n_heads,
                d_model,
                source_seq_len=source_seq_len,
                target_seq_len=target_seq_len,
                scale=(d_model // n_heads) ** scale_exponent,
                k_bias=True,
                dropout=dropout,
            )
            if cross_attn
            else None
        )
        self.cross_attn_ln = nn.LayerNorm(d_model) if cross_attn else None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            activation(),
            nn.Linear(d_mlp, d_model),
        )
        self.mlp_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        xc: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
    ) -> Tensor:
        x = x + self.attn(
            x=self.attn_ln(x), attention_mask=attention_mask, kv_cache=kv_cache
        )
        if self.cross_attn and self.cross_attn_ln:
            x = x + self.cross_attn(
                self.cross_attn_ln(x),
                xc,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
        x = x + self.mlp(self.mlp_ln(x))
        return x


class RelativePositionResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(
        self,
        source_seq_len: int,
        target_seq_len: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        is_causal: bool,
        activation: nn.Module = nn.GELU,
        cross_attn: bool = False,
        dropout: float = 0.1,
        scale_exponent: float = 0,
    ):
        super().__init__(
            source_seq_len=source_seq_len,
            target_seq_len=target_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            d_mlp=d_mlp,
            is_causal=is_causal,
            activation=activation,
            cross_attn=cross_attn,
            dropout=dropout,
            scale_exponent=scale_exponent,
        )
        self.attn = RelativePositionMultiHeadAttention(
            n_heads=n_heads,
            d_model=d_model,
            source_seq_len=target_seq_len,
            target_seq_len=target_seq_len,
            q_bias=False,
            k_bias=False,
            v_bias=False,
            out_bias=False,
            scale=1,
            dropout=dropout,
            is_causal=is_causal,
            flash=False,
        )
        self.attn_ln = RMSNorm(d_model)

        if cross_attn:
            self.cross_attn = MultiHeadAttention(
                n_heads,
                d_model,
                source_seq_len=source_seq_len,
                target_seq_len=target_seq_len,
                scale=1,
                q_bias=False,
                k_bias=False,
                v_bias=False,
                out_bias=False,
                dropout=dropout,
                flash=False,
            )

            self.cross_attn_ln = RMSNorm(d_model)

        self.mlp = nn.Sequential(
            GEGLU(d_model, d_mlp, bias=False),
            nn.Linear(d_mlp, d_model, bias=False),
        )
        self.mlp_ln = RMSNorm(d_model)


class NeuralEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        block_size: int,
        d_model: int,
        d_mlp: int,
        n_heads: int,
        dropout: float,
        n_layers: int,
        scale_exponent: float,
        checkpoint_activations: bool = False,
    ):
        super().__init__()

        self.embed_positions = nn.Embedding(block_size, d_model)
        self.embed_positions.weight = nn.Parameter(sinusoids(block_size, d_model))
        self.sample_proj = nn.Linear(n_channels, d_model)

        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(
                block_size,
                block_size,
                d_model,
                n_heads,
                d_mlp=d_mlp,
                dropout=dropout,
                scale_exponent=scale_exponent,
                is_causal=False,
            )
            for _ in range(n_layers)
        )
        self.ln_post = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.checkpoint_activations = checkpoint_activations

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, n_eeg_channels, sequence_length)
        B, N_C, T = x.size()
        x = self.sample_proj(x.transpose(-1, -2))
        x = (x + self.embed_positions.weight[None, :, :]).to(x.dtype)
        if self.checkpoint_activations:
            x = checkpoint_sequential(
                self.blocks, len(self.blocks), x, use_reentrant=False
            )
        else:
            for block in self.blocks:
                x = block(x=x)
        x = self.ln_post(x)
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
    def from_pretrained(cls, config: TelepathConfig):
        e_pt = WhisperModel.from_pretrained(config.encoder_pretrained_model)
        assert isinstance(e_pt, WhisperModel), type(e_pt)

        e_n = cls(
            n_channels=config.n_eeg_channels,
            block_size=config.encoder_block_size,
            d_model=config.d_model,
            d_mlp=config.encoder_d_mlp,
            n_heads=config.n_heads,
            dropout=config.dropout,
            n_layers=config.encoder_n_layers,
            scale_exponent=config.encoder_scale_exponent,
        )
        # breakpoint()

        new_keys = list(e_n.state_dict().keys())
        nw_sd = e_n.state_dict()
        for key, param in e_pt.get_encoder().state_dict().items():
            param: Tensor
            new_key = ENCODER_PARAM_MAP.map_param(key)
            assert new_key in new_keys, f"{new_key} not in {new_keys}."
            if "conv" in key:
                continue

            if key == "encoder.embed_positions.weight":
                param = param[: config.encoder_block_size, :]

            nw_sd[new_key] = param.clone()
        e_n.load_state_dict(nw_sd)
        return e_n


class RelativePositionTransformer(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        source_n_ctx: int,
        target_n_ctx: int,
        d_model: int,
        d_mlp: int,
        n_head: int,
        n_layer: int,
        cross_attn: bool,
        is_causal: bool,
        checkpoint_activations: bool = False,
        scale_exponent: float = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(n_vocab, d_model)
        self.blocks = nn.ModuleList(
            RelativePositionResidualAttentionBlock(
                source_n_ctx,
                target_n_ctx,
                d_model,
                n_head,
                d_mlp=d_mlp,
                cross_attn=cross_attn,
                scale_exponent=scale_exponent,
                is_causal=is_causal,
                dropout=dropout,
            )
            for _ in range(n_layer)
        )
        self.ln_post = RMSNorm(d_model)
        self.checkpoint_activations = checkpoint_activations

    def forward(
        self,
        x: Tensor,
        return_hidden_states: bool,
        xc: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        inference: bool = False,
    ) -> Tensor:
        offset = kv_cache[next(iter(kv_cache.keys()))].size(1) if kv_cache else 0
        x = self.embed_tokens(x[:, offset:])
        if self.checkpoint_activations:
            x = checkpoint_sequential(
                self.blocks, len(self.blocks), x, use_reentrant=False
            )
        else:
            for block in self.blocks:
                x = block(x, xc, attention_mask=attention_mask, kv_cache=kv_cache)
        x = self.ln_post(x)
        if inference:
            x = x[:, -1, :]
        if return_hidden_states:
            return x
        logits = x @ torch.transpose(self.embed_tokens.weight.to(x.dtype), 0, 1)
        return logits

    @classmethod
    def from_pretrained(cls, config: TelepathConfig, cross_attn: bool, is_causal: bool):
        d_pt = T5ForConditionalGeneration.from_pretrained(
            config.decoder_pretrained_model
        )
        assert isinstance(d_pt, T5ForConditionalGeneration), type(d_pt)

        d_n = cls(
            n_vocab=config.decoder_vocab_size,
            source_n_ctx=config.encoder_block_size,
            target_n_ctx=config.decoder_block_size,
            d_model=config.d_model,
            d_mlp=config.decoder_d_mlp,
            n_head=config.n_heads,
            n_layer=config.decoder_n_layers,
            dropout=config.dropout,
            cross_attn=cross_attn,
            is_causal=is_causal,
        )

        new_keys = list(d_n.state_dict().keys())
        nw_sd = d_n.state_dict()
        for key, param in d_pt.get_decoder().state_dict().items():
            param: Tensor
            new_key = T5_PARAM_MAP.map_param(key)
            assert new_key in new_keys, f"{new_key} not in {new_keys}."

            nw_sd[new_key] = param.clone()
            if "rp_bias.relative_attention_bias" not in new_key:
                continue
            for i in range(1, len(d_n.blocks)):
                new_key = new_key.replace(f".{i-1}.", f".{i}.")
                nw_sd[new_key] = param.clone()
        d_n.load_state_dict(nw_sd)
        return d_n


class TextEncoder(RelativePositionTransformer):
    def __init__(
        self,
        n_vocab: int,
        source_n_ctx: int,
        target_n_ctx: int,
        d_model: int,
        d_mlp: int,
        n_head: int,
        n_layer: int,
        checkpoint_activations: bool = False,
        scale_exponent: float = 0,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_vocab=n_vocab,
            source_n_ctx=source_n_ctx,
            target_n_ctx=target_n_ctx,
            d_model=d_model,
            d_mlp=d_mlp,
            n_head=n_head,
            n_layer=n_layer,
            checkpoint_activations=checkpoint_activations,
            scale_exponent=scale_exponent,
            dropout=dropout,
            cross_attn=False,
            is_causal=False,
        )

    def forward(
        self,
        x: Tensor,
        return_hidden_states: bool = True,
        xc: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        inference: bool = False,
    ) -> Tensor:
        return super().forward(
            x, return_hidden_states, xc, attention_mask, kv_cache, inference
        )


class TextDecoder(RelativePositionTransformer):
    def __init__(
        self,
        n_vocab: int,
        source_n_ctx: int,
        target_n_ctx: int,
        d_model: int,
        d_mlp: int,
        n_head: int,
        n_layer: int,
        checkpoint_activations: bool = False,
        scale_exponent: float = 0,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_vocab=n_vocab,
            source_n_ctx=source_n_ctx,
            target_n_ctx=target_n_ctx,
            d_model=d_model,
            d_mlp=d_mlp,
            n_head=n_head,
            n_layer=n_layer,
            checkpoint_activations=checkpoint_activations,
            scale_exponent=scale_exponent,
            dropout=dropout,
            cross_attn=True,
            is_causal=True,
        )

    def forward(
        self,
        x: Tensor,
        return_hidden_states: bool = False,
        xc: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        inference: bool = False,
    ) -> Tensor:
        return super().forward(
            x, return_hidden_states, xc, attention_mask, kv_cache, inference
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        embed: torch.Tensor | None,
        max_length: int,
        stop_token: int,
    ) -> list[list[int]]:
        """Generate a sequence of tokens using argmax sampling.

        Optimised for batch generation; pops generations off the inference stack as they complete.
        Attributes:
            input_ids: Input token ids of shape (batch_size, n_tokens).
            embed: Additional embeddings used by the model forward method. Must be of size (batch_size, k, d_model).
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
            logits = self.forward(
                x=input_ids, xc=embed, inference=True, kv_cache=kv_cache
            )
            input_ids = torch.cat(
                [input_ids, logits.argmax(dim=-1).unsqueeze(0).t()], dim=1
            )
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


class TelepathGenerator(nn.Module):
    def __init__(self, config: TelepathConfig):
        super().__init__()
        self.config = config

        self.encoder = NeuralEncoder(
            n_channels=config.n_eeg_channels,
            block_size=config.neural_encoder_block_size,
            d_model=config.d_model,
            d_mlp=config.neural_encoder_d_mlp,
            n_heads=config.n_heads,
            n_layers=config.neural_encoder_n_layers,
            dropout=config.dropout,
            scale_exponent=config.neural_encoder_scale_exponent,
        )

        self.decoder = TextDecoder(
            n_vocab=config.decoder_vocab_size,
            encoder_n_ctx=config.encoder_block_size,
            decoder_n_ctx=config.decoder_block_size,
            d_model=config.d_model,
            d_mlp=config.decoder_d_mlp,
            n_head=config.n_heads,
            n_layer=config.decoder_n_layers,
            dropout=config.dropout,
        )

        if not config.train_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.start_sequence = config.decoder_start_sequence
        self.stop_token = config.decoder_stop_token

        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_pretrained_model)

    def forward(
        self, eeg: Tensor, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through the Telepath model.

        Attributes:
            eeg_signal: EEG signal of shape (batch_size, n_samples, n_channels).
            input_ids: Input token ids of shape (batch_size, n_tokens).
        """
        enc = self.encoder(eeg)
        return enc, self.decoder(input_ids, xc=enc, attention_mask=attention_mask)

    def step(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        (
            eeg,
            token_ids,
        ) = (batch["input_features"], batch["input_ids"])
        eeg = eeg.to(dtype=torch.bfloat16)
        # Remove the last token from the logits, as we don't need to predict the padding token.
        enc, logits = self.forward(eeg, token_ids)
        logits = logits[:, :-1, :].contiguous()
        # Flatten logits tensor (B x T-1 x V) to 2D tensor ((B T-1) x V) for loss calculation.
        logits = logits.view(-1, logits.size(-1))
        # Mask all padding tokens except the first which are being used as stop tokens.
        loss_mask = token_ids == self.config.decoder_stop_token
        # HACK: Exploits the fact that argmax breaks ties by returning the lowest index.
        stop_token_indices = (loss_mask == 1).to(torch.long).argmax(dim=1)
        batch_indices = torch.arange(loss_mask.shape[0])
        loss_mask[batch_indices, stop_token_indices] = 0
        loss_mask[batch_indices, : len(self.start_sequence)] = 1
        labels = token_ids.clone()
        labels[loss_mask] = -100
        # Shift and flatten labels (B x T) to 1D tensor (B T-1).
        labels = labels[:, 1:].contiguous().view(-1)
        # Mask special tokens in the loss function, except for the first EOS token.
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return enc, logits, loss

    def configure_optimizers(
        self, num_batches: int, max_lr: float, weight_decay: float, warmup_frac: float
    ):
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=max_lr,
            weight_decay=weight_decay,
        )
        warmup_batches = int(num_batches * warmup_frac)
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-7, end_factor=1, total_iters=warmup_batches
        )
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=num_batches)
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, decay_scheduler], milestones=[warmup_batches]
        )
        return optimizer, scheduler

    @property
    def module(self):
        """For interoperability with DDP"""
        return self

    @torch.no_grad()
    def generate(
        self,
        eeg: Tensor | None = None,
        enc: Tensor | None = None,
        device: str = "cuda",
        stop_token: int | None = None,
    ) -> list[list[int]]:
        """Generate a sequence of tokens given an EEG signal.
        Attributes:
            eeg_signal: EEG signal of shape (batch_size, n_samples, n_channels).
            stop_token: Token id to stop generation at.
        """
        if eeg is not None:
            assert len(eeg.size()) == 3
            eeg = eeg.to(device)
            enc = self.encoder(eeg)
        assert enc is not None
        B = enc.size(0)
        return self.decoder.generate(
            input_ids=self.start_sequence.repeat(B, 1).detach().to(device),
            embed=enc,
            stop_token=stop_token or self.stop_token,
            max_length=self.config.
        )

    @classmethod
    def from_pretrained(cls, config: TelepathConfig):
        """Initialize the model from pretrained Whisper."""
        model = cls(config)
        model.encoder = NeuralEncoder.from_pretrained(config)
        model.decoder = TextDecoder.from_pretrained(config)
        return model
