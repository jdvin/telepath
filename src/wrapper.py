from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
import yaml

from .telepath import Telepath, TelepathConfig

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
LR_SCHEDULERS = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "linear": torch.optim.lr_scheduler.StepLR,
}


def pass_notation(params: dict[str, Any]) -> dict[str, Any]:
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
        config["optim_params"] = pass_notation(config["optim_params"])
        config["lr_scheduler_params"] = pass_notation(config["lr_scheduler_params"])
        optim = OPTIMIZERS[config.pop("optim")]
        lr_scheduler = LR_SCHEDULERS[config.pop("lr_scheduler")]
        return cls(optim=optim, lr_scheduler=lr_scheduler, **config)


class TelepathWrapper:
    def __init__(self, model_config_path: str, optimizer_config_path: str):
        super().__init__()
        self.config = TelepathConfig.from_yaml(model_config_path)
        self.model = Telepath(self.config)
        self.optimizer_config = OptimizerConfig.from_yaml(optimizer_config_path)

        if self.config.freeze_gpt:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def step(
        self,
        step_type: str,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        eeg, tokens = batch["eeg"], batch["input_ids"]
        logits = self.model(eeg, tokens)[:, eeg.size(-2) :, :].clone()
        loss = self.model.decoder.loss(logits, tokens)
        return loss

    def configure_optimizers(self) -> tuple:
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
            optim, **self.optimizer_config.lr_scheduler_params
        )

        return optim, lr_scheduler
