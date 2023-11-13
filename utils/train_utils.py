from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import yaml
import wandb


@dataclass
class TrainMetrics:
    train_loss: float = 0
    val_loss: float = 0
    microstep: int = 0
    step: int = 0
    epoch: int = 0
    lr: float = 0
    opt: dict[str, Any] = field(default_factory=dict)

    def log(self):
        metrics = asdict(self)
        opt = metrics.pop("opt")
        metrics = {
            key.replace("_", "/"): value for key, value in {**metrics, **opt}.items()
        }

        wandb.log(**metrics)
        self.train_loss = 0
        self.val_loss = 0


def load_yaml(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_batch(dataloader, device: str) -> dict[str, torch.Tensor]:
    batch = next(iter(dataloader))

    for key, value in batch:
        if device == "cpu":
            batch[key] = value.to(device)
        else:
            batch[key] = value.pin_memory().to(device, non_blocking=True)
    return batch
