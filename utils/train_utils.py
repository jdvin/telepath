from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import Any
import random

import yaml
import wandb
from tqdm import tqdm
import torch

from src.wrapper import TelepathWrapper


class MetricLogRule(Enum):
    EVERY_STEP = "every_step"
    ON_CHANGE = "on_change"
    MANUAL = "manual"


@dataclass
class Metric:
    """I'm in the arena, trying things."""

    value: Any
    _update_rule: MetricLogRule
    _log: bool = False
    _past: Any | None = None


@dataclass
class TrainMetrics:
    train_loss = Metric(0, MetricLogRule.EVERY_STEP)
    val_loss = Metric(-1, MetricLogRule.ON_CHANGE)
    val_accuracy = Metric(-1, MetricLogRule.ON_CHANGE)
    microstep = Metric(1, MetricLogRule.EVERY_STEP)
    step = Metric(1, MetricLogRule.EVERY_STEP)
    epoch = Metric(1, MetricLogRule.EVERY_STEP)
    lr = Metric(0, MetricLogRule.EVERY_STEP)
    generations = Metric(
        wandb.Table(columns=["target", "output"]), MetricLogRule.MANUAL
    )

    def log(self):
        log_metrics = {}
        for key, metric in asdict(self).items():
            if (
                metric._update_rule == MetricLogRule.EVERY_STEP
                or (
                    metric.update_rule == MetricLogRule.ON_CHANGE
                    and metric.value != metric._past[key]
                )
                or (metric.update_rule == MetricLogRule.MANUAL and metric._log)
            ):
                metric._log = False
                metric._past = metric.value
                key = key.replace("_", "/")
                log_metrics[key] = metric.value

        wandb.log(log_metrics)


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
            # TODO: This loop pattern probably stops us from being able to take advantage of the non-blocking device movement.
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


@torch.no_grad()
def get_accuracy(
    batch: dict[str, torch.Tensor], wmodel: TelepathWrapper, metrics: TrainMetrics
) -> float:
    """Naiive accuracy calculation.

    Does not take into account the fact that the model may have predicted a synonym of the correct word.
    """

    pred_tokens: list[list[int]] = wmodel.model.generate(batch["eeg"], wmodel.device)
    pred_text = wmodel.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
    true_text = wmodel.tokenizer.batch_decode(
        batch["input_ids"], skip_special_tokens=True
    )
    for pred, true in zip(pred_text, true_text):
        metrics.generations.value.add_data(true, pred)
    assert len(pred_tokens) == len(batch["input_ids"])
    accuracy = 0
    for pred, true in zip(pred_tokens, batch["input_ids"]):
        # True values are padded for training.
        if pred == true[: len(pred)]:
            accuracy += 1 / len(pred_tokens)
    return accuracy


@torch.no_grad()
def run_eval(
    wmodel: TelepathWrapper, val_dataloader: DataLoader, metrics: TrainMetrics
):
    metrics.val_loss.value = 0
    metrics.val_accuracy.value = 0
    val_pbar = tqdm(total=len(val_dataloader), desc="Running validation")
    for k, micro_batch in enumerate(val_dataloader):
        metrics.val_loss.value += wmodel.step(micro_batch).item() / len(val_dataloader)
        metrics.val_accuracy.value += get_accuracy(micro_batch, wmodel, metrics) / len(
            val_dataloader
        )
        val_pbar.update()
    metrics.generations._log = True
