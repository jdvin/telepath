"""All credit for this GPT implementation goes to Karpathy; this is a direct copy of nanoGPT."""
import math
import re

import torch
import torch.nn as nn
from torch.nn import functional as F

from .components.norm import LayerNorm
from .components.attention import MultiheadAttention


class Block(nn.Module):
    def __init__(
        self, n_heads: int, d_model: int, bias: bool, block_size: int, dropout: float
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, affine=True, bias=bias)
        self.attn = MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model,
            proj_bias=bias,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln_2 = LayerNorm(d_model, affine=True, bias=bias)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        bias: bool,
        n_layers: int,
        vocab_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, d_model),
                wpe=nn.Embedding(block_size, d_model),
                drop=nn.Dropout(dropout),
                blocks=nn.ModuleList(
                    [
                        Block(n_heads, d_model, bias, block_size, dropout)
                        for _ in range(n_layers)
                    ]
                ),
                ln_f=LayerNorm(d_model, affine=True, bias=bias),
            ),
        )
        assert isinstance(self.transformer.wpe, nn.Embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)
        self.block_size = block_size

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt((2 * n_layers)))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # import pdb

        # pdb.set_trace()
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def forward(
        self,
        x: torch.Tensor,
        concat_embed: torch.Tensor | None = None,
        inference: bool = False,
    ) -> torch.Tensor:
        # TODO: KV cache.
        # B x T x D.
        tok_emb = self.transformer.wte(x)  # type: ignore
        if concat_embed is not None:
            tok_emb = torch.cat([concat_embed, tok_emb], dim=1)
        T = tok_emb.size(1)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore
        for block in self.transformer.blocks:  # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x)  # type: ignore

        if not inference:
            logits: torch.Tensor = self.lm_head(x)
        else:
            # Return logits for final token of each sequence. Nesting the last dim in a list ensures that it is not flattened.
            logits: torch.Tensor = self.lm_head(x[:, [-1]])
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        concat_embed: torch.Tensor,
        max_length: int = 10,
        stop_token: int = 50256,
    ) -> torch.Tensor:
        """Generate a sequence of tokens using argmax sampling.
        Attributes:
            input_ids: Input token ids of shape (batch_size, n_tokens).
            concat_embed: Concatenated embeddings of shape (batch_size, k, d_model).
            max_length: Maximum length of the generated sequence.
            stop_token: Token id of the stop token.
        """
        for _ in range(max_length):
            logits = self.forward(input_ids, concat_embed, inference=True)
            input_ids = torch.cat([input_ids, logits.argmax(dim=-1)], dim=1)
            if (input_ids == stop_token).any():
                break
        return input_ids

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.block_size
        self.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])  # type: ignore
        for block in self.transformer.blocks:  # type: ignore
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        PARAM_MAP = (
            (r"transformer.wte.weight", r"transformer.wte.weight"),
            (r"transformer.wpe.weight", r"transformer.wpe.weight"),
            (
                r"transformer.h.[0-9]+.ln_1.weight",
                r"transformer.blocks.[0-9]+.ln_1.graph.2.weights",
            ),
            (
                r"transformer.h.[0-9]+.ln_1.bias",
                r"transformer.blocks.[0-9]+.ln_1.graph.2.bias",
            ),
            (
                r"transformer.h.[0-9]+.attn.c_attn.weight",
                r"transformer.blocks.[0-9]+.attn.qkv_proj.weight",
            ),
            (
                r"transformer.h.[0-9]+.attn.c_attn.bias",
                r"transformer.blocks.[0-9]+.attn.qkv_proj.bias",
            ),
            (
                r"transformer.h.[0-9]+.attn.c_proj.weight",
                r"transformer.blocks.[0-9]+.attn.out_proj.weight",
            ),
            (
                r"transformer.h.[0-9]+.attn.c_proj.bias",
                r"transformer.blocks.[0-9]+.attn.out_proj.bias",
            ),
            (
                r"transformer.h.[0-9]+.ln_2.weight",
                r"transformer.blocks.[0-9]+.ln_2.graph.2.weights",
            ),
            (
                r"transformer.h.[0-9]+.ln_2.bias",
                r"transformer.blocks.[0-9]+.ln_2.graph.2.bias",
            ),
            (
                r"transformer.h.[0-9]+.mlp.c_fc.weight",
                r"transformer.blocks.[0-9]+.mlp.0.weight",
            ),
            (
                r"transformer.h.[0-9]+.mlp.c_fc.bias",
                r"transformer.blocks.[0-9]+.mlp.0.bias",
            ),
            (
                r"transformer.h.[0-9]+.mlp.c_proj.weight",
                r"transformer.blocks.[0-9]+.mlp.2.weight",
            ),
            (
                r"transformer.h.[0-9]+.mlp.c_proj.bias",
                r"transformer.blocks.[0-9]+.mlp.2.bias",
            ),
            (r"transformer.ln_f.weight", r"transformer.ln_f.graph.2.weights"),
            (r"transformer.ln_f.bias", r"transformer.ln_f.graph.2.bias"),
            (r"lm_head.weight", r"lm_head.weight"),
        )
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layers=12, n_heads=12, d_model=768),  # 124M params
            "gpt2-medium": dict(n_layers=24, n_heads=16, d_model=1024),  # 350M params
            "gpt2-large": dict(n_layers=36, n_heads=20, d_model=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_heads=25, d_model=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = bool(True)  # always True for GPT model checkpoints
        config_args["dropout"] = 0.1
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized GPT model
        model = GPT(**config_args)  # type: ignore
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()  # type: ignore

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k
            for k in sd_keys_hf
            if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))
        ]  # ignore these, just a buffer
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        for k in sd_keys_hf:
            # This disgusting HACK is required because I insist on being able to name things whatever I want.
            try:
                mapped_reg = [
                    loc_reg for hf_reg, loc_reg in PARAM_MAP if re.match(hf_reg, k)
                ][0]
                mapped_key = [
                    loc_key for loc_key in sd_keys if re.match(mapped_reg, loc_key)
                ][0]
            except IndexError as e:
                print(f"Unmatched key: {k}.")
                raise e
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[mapped_key].shape
                with torch.no_grad():
                    sd[mapped_key].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[mapped_key].shape
                with torch.no_grad():
                    sd[mapped_key].copy_(sd_hf[k])

        return model

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
