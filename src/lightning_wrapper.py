from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
import yaml

from telepath import Telepath, TelepathConfig

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


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    lr_scheduler: str
    lr_scheduler_interval: str
    lr_scheduler_params: dict[str, Any]

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        return cls(**config)


@dataclass
class OptimizerConfig:
    optim: torch.optim.Optimizer
    optim_params: dict[str, Any]
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    lr_scheduler_params: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        optim = OPTIMIZERS[config.pop("name")]
        lr_scheduler = LR_SCHEDULERS[config.pop("lr_scheduler")]
        return cls(optim=optim, lr_scheduler=lr_scheduler, **config)


class TelepathLightningWrapper(pl.LightningModule):
    def __init__(self, model_config_path: str, optimizer_config_path: str):
        super().__init__()
        self.config = TelepathConfig.from_yaml(model_config_path)
        self.model = Telepath(self.config)
        self.optimizer_config = OptimizerConfig.from_yaml(optimizer_config_path)

        if self.config.freeze_gpt:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        eeg, tokens = batch
        pred_tokens = self.model(eeg, tokens)
        loss = self.model.decoder.loss(pred_tokens, tokens)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg, tokens = batch
        pred_tokens = self.model(eeg, tokens)
        loss = self.model.decoder.loss(pred_tokens, tokens)
        self.log("eval_loss", loss)
        return loss

    def configure_optimizers(self):
        param_groups = self.model.encoder.optim_groups(
            self.optimzer_config.optim_params.pop("weight_decay")
        )
        if not self.config.freeze_gpt:
            param_groups.extend(
                self.model.decoder.optim_groups(
                    self.optimizer_config.optim_params.pop("weight_decay")
                )
            )

        return self.optimizer_config.optim(
            param_groups, **self.optimizer_config.optim_params
        )
