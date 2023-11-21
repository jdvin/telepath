from dataclasses import dataclass, field, asdict
import math
from typing import Any
import random

import yaml
import wandb


@dataclass
class TrainMetrics:
    train_loss: float = 0
    val_loss: float = -1
    microstep: int = 1
    step: int = 1
    epoch: int = 1
    lr: float = 0
    _past: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._past = {key: -1 for key in asdict(self).keys()}

    def log(self):
        metrics = asdict(self)
        metrics.pop("_past")
        # Only log values that have changed.
        # TODO: Check if this is necessary.
        metrics = {
            key.replace("_", "/"): value
            for key, value in metrics.items()
            if self._past[key] != value
        }

        wandb.log(metrics)
        self._past = metrics


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
