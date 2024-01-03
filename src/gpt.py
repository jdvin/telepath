"""All credit for this GPT implementation goes to Karpathy; this is a direct copy of nanoGPT."""
import math
import re

import torch
import torch.nn as nn
from torch.nn import functional as F

from .components.norm import LayerNorm
from .components.attention import MultiheadAttention, ExpertAttention


class Block(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        bias: bool,
        block_size: int,
        dropout: float,
        flash: bool = True,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, affine=True, bias=bias)
        self.attn = MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model,
            proj_bias=bias,
            block_size=block_size,
            dropout=dropout,
            is_causal=True,
            flash=flash,
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


class ExpertBlock(Block):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        bias: bool,
        core_block_size: int,
        expert_block_size: int,
        dropout: float,
        is_causal: bool = True,
        flash: bool = True,
    ):
        super().__init__(
            n_heads, d_model, bias, core_block_size + expert_block_size, dropout
        )
        self.attn = ExpertAttention(
            n_heads=n_heads,
            d_model=d_model,
            core_proj_bias=bias,
            expert_proj_bias=bias,
            core_block_size=core_block_size,
            expert_block_size=expert_block_size,
            dropout=dropout,
            is_causal=is_causal,
            flash=flash,
        )
        self.expert_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
            nn.Dropout(dropout),
        )

        self.expert_block_size = expert_block_size
        self.core_block_size = core_block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x_resid = x.clone()
        x = self.ln_2(x)
        x = x_resid + torch.cat(
            (
                self.expert_mlp(x[:, : self.expert_block_size, :]),
                self.mlp(x[:, self.expert_block_size :, :]),
            ),
            dim=1,
        )

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
        flash: bool = True,
    ):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, d_model),
                wpe=nn.Embedding(block_size, d_model),
                drop=nn.Dropout(dropout),
                blocks=nn.ModuleList(
                    [
                        Block(n_heads, d_model, bias, block_size, dropout, flash)
                        for _ in range(n_layers)
                    ]
                ),
                ln_f=LayerNorm(d_model, affine=True, bias=bias),
            ),
        )
        assert isinstance(self.transformer.wpe, nn.Embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)
        self.block_size = block_size
        self.vocab_size = vocab_size

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
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

    def forward(
        self,
        x: torch.Tensor,
        embed: torch.Tensor | None = None,
        inference: bool = False,
    ) -> torch.Tensor:
        # TODO: KV cache.
        # B x T x D.
        tok_emb = self.transformer.wte(x)  # type: ignore
        if embed is not None:
            tok_emb = torch.cat([embed, tok_emb], dim=1)
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
        embed: torch.Tensor | None = None,
        max_length: int = 10,
        stop_token: int = 50256,
    ) -> list[list[int]]:
        """Generate a sequence of tokens using argmax sampling.

        Pops generations off the inference stack as they complete.
        Attributes:
            input_ids: Input token ids of shape (batch_size, n_tokens).
            concat_embed: Additional embeddings used by the model forward method. Must be of size (batch_size, k, d_model).
            max_length: Maximum length of the generated sequence.
            stop_token: Token id of the stop token.
        """
        batch_size = input_ids.size(0) if len(input_ids.size()) == 2 else 1
        # Preallocate generations so that we can save them by order.
        generations = [[] for _ in range(batch_size)]
        # Used to track the indexes of the running generations.
        generating_batch_indexes = list(range(batch_size))
        for _ in range(max_length):
            logits = self.forward(input_ids, embed, inference=True)
            input_ids = torch.cat([input_ids, logits.argmax(dim=-1)], dim=1)
            # Get the index of every generation that has generated the stop token.
            stop_indexes = (
                (input_ids[:, -1] == stop_token).nonzero(as_tuple=True)[0].tolist()
            )
            while True:
                if not stop_indexes:
                    break
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
        model = cls(**config_args)  # type: ignore
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
                # import pdb

                # pdb.set_trace()
                sd_keys.remove(mapped_key)
            except IndexError as e:
                print(f"Unmatched key: {k}.")
                raise e
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[mapped_key].shape
                with torch.no_grad():
                    sd[mapped_key].copy_(sd_hf.pop(k).t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[mapped_key].shape
                with torch.no_grad():
                    sd[mapped_key].copy_(sd_hf.pop(k))

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


class ExpertGPT(GPT):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        bias: bool,
        n_layers: int,
        vocab_size: int,
        core_block_size: int,
        expert_block_size: int,
        freeze_core: bool,
        dropout: float,
        flash: bool = True,
    ):
        super().__init__(
            n_heads,
            d_model,
            bias,
            n_layers,
            vocab_size,
            core_block_size,
            dropout,
            flash,
        )
        for i in range(len(self.transformer.blocks)):
            self.transformer.blocks[i] = ExpertBlock(
                n_heads,
                d_model,
                bias,
                core_block_size,
                expert_block_size,
                dropout,
                flash,
            )

        self.freeze_core = freeze_core

    def forward(
        self,
        input_ids: torch.Tensor,
        embed: torch.Tensor,
        inference: bool = False,
    ) -> torch.Tensor:
        """Subtly different forward method."""
        tok_emb = self.transformer.wte(input_ids)
        T = tok_emb.size(1)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.transformer.wpe(pos)
        x = torch.cat((embed, tok_emb + pos_emb), dim=1)
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        if not inference:
            logits: torch.Tensor = self.lm_head(x[:, self.expert_block_size :, :])
        else:
            # Return logits for final token of each sequence. Nesting the last dim in a list ensures that it is not flattened.
            logits: torch.Tensor = self.lm_head(x[:, [-1]])
        return logits

    def optim_groups(self, weight_decay: float = 1e-1):
        # Filter out those that do not require grad
        # If freeze_core, then only optimize expert parameters.
        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if p.requires_grad and not (self.freeze_core and "expert" not in pn)
        }
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
