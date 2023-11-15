from dataclasses import dataclass, field, asdict
import math
from typing import Any
import random

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

        wandb.log(metrics)
        self.train_loss = 0
        self.val_loss = 0


class DataLoader:
    def __init__(self, dataset, batch_size: int, device: str, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.num_batches = math.ceil(len(dataset) / batch_size)
        self.i = 0

    def reset(self):
        self.i = 0
        self.batch_indexes = list(range(self.num_batches))
        if self.shuffle:
            random.shuffle(self.batch_indexes)

    def get_batch(self):
        if self.i % self.num_batches == 0:
            self.reset()
        batch = self.dataset[self.i * self.batch_size : (self.i + 1) * self.batch_size]
        self.i += 1

        for key, value in batch.items():
            if "cuda" in self.device:
                batch[key] = value.pin_memory().to(self.device, non_blocking=True)
            else:
                batch[key] = value.to(self.device)
        return batch

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.get_batch()

    def __len__(self):
        return self.num_batches


def load_yaml(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
