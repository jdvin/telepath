from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
import yaml

from .telepath import Telepath, TelepathConfig

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
LR_SCHEDULERS = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "linear": torch.optim.lr_scheduler.StepLR,
}


def parse_notation(params: dict[str, Any]) -> dict[str, Any]:
    for k, v in params.items():
        if isinstance(v, str):
            try:
                params[k] = float(v)
            except ValueError:
                pass
    return params


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


@dataclass
class OptimizerConfig:
    optim: torch.optim.Optimizer
    optim_params: dict[str, Any]
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    lr_scheduler_params: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        config["optim_params"] = parse_notation(config["optim_params"])
        config["lr_scheduler_params"] = parse_notation(
            config.get("lr_scheduler_params", {})
        )
        optim = OPTIMIZERS[config.pop("optim")]
        lr_scheduler = LR_SCHEDULERS[config.pop("lr_scheduler")]
        return cls(optim=optim, lr_scheduler=lr_scheduler, **config)


class TelepathWrapper:
    def __init__(self, model_config_path: str, optimizer_config_path: str, device: str):
        super().__init__()
        self.config = TelepathConfig.from_yaml(model_config_path)
        self.model = Telepath(self.config).to(device)
        self.optimizer_config = OptimizerConfig.from_yaml(optimizer_config_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.config.freeze_gpt:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        eeg, tokens = batch["eeg"], batch["input_ids"]
        logits = self.model(eeg, tokens).clone()
        loss = self.model.decoder.loss(logits, tokens)
        return loss

    def configure_optimizers(self, num_batches: int) -> tuple:
        param_groups = self.model.encoder.optim_groups(
            self.optimizer_config.optim_params.pop("weight_decay")
        )
        if not self.config.freeze_gpt:
            param_groups.extend(
                self.model.decoder.optim_groups(
                    self.optimizer_config.optim_params.pop("weight_decay")
                )
            )

        optim = self.optimizer_config.optim(
            param_groups, **self.optimizer_config.optim_params
        )
        lr_scheduler = self.optimizer_config.lr_scheduler(
            optim, **self.optimizer_config.lr_scheduler_params, T_max=num_batches
        )

        return optim, lr_scheduler
