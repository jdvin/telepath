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

from .components.attention import MultiHeadAttention, RelativePositionMultiHeadAttention
from .activations import GEGLU

from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration


@dataclass
class TelepathConfig:
    n_eeg_channels: int
    encoder_pretrained_model: str | None
    encoder_sd_prefix: str | None
    decoder_pretrained_model: str | None
    decoder_sd_prefix: str | None
    decoder_start_sequence: Tensor = tensor([])
    decoder_stop_token: int = 0
    decoder_vocab_size: int = 0
    encoder_block_size: int = 0
    decoder_block_size: int = 0
    n_freqs: int = 0
    fft_hop_length: int = 0
    d_model: int = 0
    encoder_d_mlp: int = 0
    encoder_activation: str = ""
    decoder_d_mlp: int = 0
    decoder_activation: str = ""
    n_heads: int = 0
    encoder_n_layers: int = 0
    decoder_n_layers: int = 0
    dropout: float = 0.1
    scale_exponent: float = -0.25
    train_decoder: bool = True

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def __post_init__(self):
        """There is definitely a better way."""
        if self.pretrained_whisper:
            pt_whisper_config = WhisperConfig.from_pretrained(self.pretrained_whisper)
            # Translate from the neural code to english, please.
            tokenizer = WhisperTokenizer.from_pretrained(
                self.pretrained_whisper, task="translate", language="english"
            )
            assert isinstance(tokenizer, WhisperTokenizer)
            self.decoder_vocab_size = pt_whisper_config.vocab_size
            # TODO:
            # We want to the channel dimension of the spectrogram to align with the channel dimension of the whisper feature extractor.
            # It is an open question whether or not doing a straight exrtraction of the N equidistant frequencies is the best approach,
            # or whether we should instead construct a neural equivalent of the mel scale.
            self.n_freqs = pt_whisper_config.num_mel_bins
            self.d_model = pt_whisper_config.d_model
            assert (
                pt_whisper_config.decoder_attention_heads
                == pt_whisper_config.encoder_attention_heads
            )
            self.n_heads = pt_whisper_config.encoder_attention_heads
            self.encoder_n_layers = (
                self.encoder_n_layers or pt_whisper_config.encoder_layers
            )
            self.decoder_n_layers = (
                self.decoder_n_layers or pt_whisper_config.decoder_layers
            )
            assert isinstance(tokenizer.prefix_tokens, list)
            self.decoder_start_sequence = torch.tensor(tokenizer.prefix_tokens)
            assert isinstance(tokenizer.eos_token_id, int)
            self.decoder_stop_token = tokenizer.eos_token_id
            self.decoder_special_tokens_start = min(
                tokenizer.added_tokens_decoder.keys()
            )


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


class RelativePositionResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        source_seq_len: int,
        target_seq_len: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        is_causal: bool,
        activation: nn.Module = GEGLU,
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
            source_seq_len=source_seq_len,
            target_seq_len=target_seq_len,
            q_bias=False,
            k_bias=False,
            v_bias=False,
            out_bias=False,
            scale=1,
            dropout=0.1,
            is_causal=is_causal,
        )

        self.cross_attn = (
            MultiHeadAttention(
                n_heads,
                d_model,
                source_seq_len=source_seq_len,
                target_seq_len=target_seq_len,
                scale=1,
                q_bias=False,
                k_bias=False,
                v_bias=False,
                dropout=dropout,
            )
            if cross_attn
            else None
        )


class NeuralEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_freqs: int,
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

        # We want the convolutions to be performed separately on each eletrode channel.
        # The channels will be stacked across the height dimension.
        self.conv1 = nn.Conv1d(
            in_channels=n_freqs * n_channels,
            out_channels=d_model * n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=n_channels,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_model * n_channels,
            out_channels=d_model * n_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=n_channels,
        )
        self.embed_positions = nn.Embedding(block_size, d_model)
        self.embed_positions.weight = nn.Parameter(sinusoids(block_size, d_model))
        self.embed_electrodes = nn.Embedding(n_channels, d_model)

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
        # (batch_size, n_ee_channels, n_freqs, sequence_length) -> (batch_size, n_freqs, n_eeg_channels, sequence_length).
        # We want the convolutions to be performed separately on each eletrode channel.
        # The inputs to a convolution 2d are of the shape (N, C_in, H, W).
        B, N_C, N_F, T = x.size()
        x = x.reshape(B, N_C * N_F, T)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.reshape(B, N_C, self.d_model, T // 2)
        # -> (batch_size, sequence_length, n_eeg_channels, d_model).
        x = x.permute(0, 3, 1, 2)
        x = (x + self.embed_positions.weight[None, :, None, :]).to(x.dtype)
        x = (x + self.embed_electrodes.weight[None, None, ...]).to(x.dtype)
        # Stack the electrode embeddings across the time dimension.
        x = x.reshape(B, N_C * (T // 2), self.d_model)
        if self.checkpoint_activations:
            x = checkpoint_sequential(
                self.blocks, len(self.blocks), x, use_reentrant=False
            )
        else:
            for block in self.blocks:
                x = block(x=x)
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
        self,
        n_vocab: int,
        n_ctx: int,
        d_model: int,
        d_mlp: int,
        n_head: int,
        n_layer: int,
        checkpoint_activations: bool = False,
        scale_exponent: float = -0.25,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(n_vocab, d_model)
        self.embed_positions = nn.Embedding(n_ctx, d_model)
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(
                n_ctx,
                d_model,
                n_head,
                d_mlp=d_mlp,
                cross_attn=True,
                scale_exponent=scale_exponent,
                is_causal=True,
            )
            for _ in range(n_layer)
        )
        assert isinstance(self.blocks[0], ResidualAttentionBlock)
        assert isinstance(self.blocks[0].attn, MultiHeadAttention)
        assert isinstance(self.blocks[0].attn.k_proj, nn.Linear)
        self.ln_post = nn.LayerNorm(d_model)
        self.checkpoint_activations = checkpoint_activations

    def forward(
        self,
        x: Tensor,
        xc: Tensor | None = None,
        attention_mask: Tensor | None = None,
        kv_cache: dict[int, Tensor] | None = None,
        inference: bool = False,
    ) -> Tensor:
        offset = kv_cache[next(iter(kv_cache.keys()))].size(1) if kv_cache else 0
        x = self.embed_tokens(x)
        x = x + self.embed_positions.weight[: x.shape[1]]
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
        logits = x @ torch.transpose(self.embed_tokens.weight.to(x.dtype), 0, 1)

        return logits

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
            logits = self.forward(input_ids, embed, inference=True, kv_cache=kv_cache)
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


class Telepath(nn.Module):
    def __init__(self, config: TelepathConfig):
        super().__init__()
        self.config = config

        self.encoder = NeuralEncoder(
            n_channels=config.n_eeg_channels,
            n_freqs=config.n_freqs,
            block_size=config.encoder_block_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.encoder_n_layers,
            dropout=config.dropout,
            scale_exponent=config.scale_exponent,
        )

        self.decoder = TextDecoder(
            n_vocab=config.decoder_vocab_size,
            n_ctx=config.decoder_block_size,
            d_model=config.d_model,
            n_head=config.n_heads,
            n_layer=config.decoder_n_layers,
            scale_exponent=config.scale_exponent,
        )

        if not config.train_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.start_sequence = config.decoder_start_sequence
        self.stop_token = config.decoder_stop_token

        # Translate from neural code to english please.
        assert isinstance(self.config.pretrained_whisper, str)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.config.pretrained_whisper, task="translation", language="en"
        )

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
        stop_token_indices = (loss_mask == 1).to(torch.long).argmax(dim=1)
        batch_indices = torch.arange(loss_mask.shape[0])
        loss_mask[batch_indices, stop_token_indices] = 0
        loss_mask[batch_indices, self.start_sequence :] = 1
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
        )

    @classmethod
    def from_pretrained(cls, config: TelepathConfig):
        """Initialize the model from pretrained Whisper."""
        ptw = WhisperModel.from_pretrained(config.pretrained_whisper)
        assert isinstance(ptw, WhisperModel)
        param_map = {
            "layers": "blocks",
            "self_attn": "attn",
            "self_attn_layer_norm": "attn_ln",  # .graph.2",
            "fc1": "mlp.0",
            "fc2": "mlp.2",
            "final_layer_norm": "mlp_ln",  # .graph.2",
            "layer_norm": "ln_post",  # .graph.2",
            "encoder_attn": "cross_attn",
            "encoder_attn_layer_norm": "cross_attn_ln",  # .graph.2",
        }
        map_params = lambda pn: ".".join(
            [param_map.get(seg, seg) for seg in pn.split(".")]
        )
        nw = cls(config)
        new_keys = list(nw.state_dict().keys())
        nw_sd = nw.state_dict()
        for key, param in ptw.state_dict().items():
            param: Tensor
            new_key = map_params(key)
            assert new_key in new_keys, f"{new_key}"
            if "conv" in key and "weight" in key:
                # Reshape the Whisper 1D conv weights to our 2D grouped conv weights
                out_channels, in_channels, kernel_size = param.shape
                param = param.unsqueeze(1).repeat(1, config.n_eeg_channels, 1, 1)
                param = param.reshape(
                    out_channels * config.n_eeg_channels, in_channels, kernel_size
                )

            if key == "decoder.embed_positions.weight":
                param = param[: config.decoder_block_size, :]

            if key == "encoder.embed_positions.weight":
                param = param[: config.encoder_block_size, :]

            nw_sd[new_key] = param.clone()
        nw.load_state_dict(nw_sd)
        return nw
