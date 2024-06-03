from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import AutoTokenizer, WhisperTokenizerFast

from .telepath import Telepath, TelepathConfig


def parse_notation(params: dict[str, Any]) -> dict[str, Any]:
    for k, v in params.items():
        if isinstance(v, str):
            try:
                params[k] = float(v)
            except ValueError:
                pass
    return params


class Trainer:
    def __init__(self, model_config_path: str, device: str | int):
        super().__init__()
        self.config = TelepathConfig.from_yaml(model_config_path)
        self.model = Telepath(self.config).to(device)
        self.device = device
        # Translate from neural code to english please.
        self.tokenizer = WhisperTokenizerFast.from_pretrained(
            self.config.pretrained_whisper, task="translation", language="en"
        )

    def step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        (
            eeg,
            token_ids,
            decoder_attention_mask,
        ) = (batch["input_features"], batch["input_ids"], batch["attention_mask"])

        # Remove the last token from the logits, as we don't need to predict the padding token.
        logits = self.model(eeg, token_ids, attention_mask=decoder_attention_mask)[
            :, :-1, :
        ].contiguous()
        # Flatten logits tensor (B x T-1 x V) to 2D tensor ((B T-1) x V) for loss calculation.
        logits = logits.view(-1, logits.size(-1))
        # Shift and flatten labels (B x T) to 1D tensor (B T-1).
        labels = token_ids[:, 1:].contiguous().view(-1)
        # Mask special tokens.
        labels[labels >= self.config.decoder_special_tokens_start] = -100
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def configure_optimizers(
        self, num_batches: int, max_lr: float, weight_decay: float, warmup_frac: float
    ):
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=max_lr,
            weight_decay=weight_decay,
        )
        warmup_batches = int(num_batches * warmup_frac)
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1, total_iters=warmup_batches
        )
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=num_batches)
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, decay_scheduler], milestones=[warmup_batches]
        )
        return optimizer, scheduler
