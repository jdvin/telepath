from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import Any
import random

import yaml
import wandb
from tqdm import tqdm
import torch
import pandas as pd

from src.wrapper import TelepathWrapper

THINGS_CONCEPTS_PATH = "data/things_concepts.csv"
SYNONYM_MAP = {
    object_word.lower().strip(): [
        synonym.lower().replace("_", " ").strip()
        for synonym in synonyms.split(",")
        if synonym.lower().replace("_", " ").strip() != object_word.lower().strip()
    ]
    for object_word, synonyms in pd.read_csv(THINGS_CONCEPTS_PATH)[
        ["Word", "WordNet Synonyms"]
    ].values
}


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
    reset: bool = False
    _reset_value: Any | None = None
    suffixes: list[str] = field(default_factory=list)
    hidden: bool = False

    def __post_init__(self):
        if self.reset:
            self._reset_value = self.value


def log_metrics(metrics: dict[str, Metric]):
    log_metrics = {}
    for key, metric in metrics.items():
        if (
            metric._update_rule == MetricLogRule.EVERY_STEP
            or (
                metric._update_rule == MetricLogRule.ON_CHANGE
                and metric.value != metric._past
            )
            or (metric._update_rule == MetricLogRule.MANUAL and metric._log)
        ):
            metric._log = False
            metric._past = metric.value
            key = key.replace("_", "/")
            # Wandb location for metrics that are not shown by default.
            # https://github.com/wandb/wandb/issues/3967
            if metric.hidden:
                key = f"hidden_panel/{key}"
            # If we want they key to change dynamically each time it gets logged (e.g., if we want to keep track of how a table will change over time).
            if metric.suffixes:
                key = f"{key}_{'_'.join([suffix + metrics[suffix].value for suffix in metric.suffixes])})"
            log_metrics[key] = metric.value
            if metric.reset:
                metric.value = metric._reset_value

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
    batch: dict[str, torch.Tensor], wmodel: TelepathWrapper, metrics: dict[str, Metric]
) -> float:
    """Slightly less naiive accuracy calculation."""

    pred_tokens: list[list[int]] = wmodel.model.generate(batch["eeg"], wmodel.device)
    batch_pred_text = wmodel.tokenizer.batch_decode(
        pred_tokens, skip_special_tokens=True
    )
    batch_true_text = wmodel.tokenizer.batch_decode(
        batch["input_ids"], skip_special_tokens=True
    )
    accuracy = 0
    for pred_text, true_text in zip(batch_pred_text, batch_true_text):
        metrics["generations"].value.add_data(true_text, pred_text)
        # TODO: This should be done in the data processing stage - check!.
        true_text = true_text.lower().strip()
        pred_text = pred_text.lower().strip()
        # Using `in` allows to account for noise in the generation at the expense of speed.
        pred_text_is_synonym = any(
            [synonym in pred_text for synonym in SYNONYM_MAP.get("true_text") or []]
        )
        if true_text in pred_text or pred_text_is_synonym:
            accuracy += 1 / len(batch_pred_text)
    return accuracy


@torch.no_grad()
def run_eval(
    wmodel: TelepathWrapper, val_dataloader: DataLoader, metrics: dict[str, Metric]
):
    metrics["val_loss"].value = 0
    metrics["val_accuracy"].value = 0
    val_pbar = tqdm(total=len(val_dataloader), desc="Running validation")
    for k, micro_batch in enumerate(val_dataloader):
        metrics["val_loss"].value += wmodel.step(micro_batch).item() / len(
            val_dataloader
        )
        metrics["val_accuracy"].value += get_accuracy(
            micro_batch, wmodel, metrics
        ) / len(val_dataloader)
        val_pbar.update()
    metrics["val_loss"]._log = True
    metrics["val_accuracy"]._log = True
    metrics["generations"]._log = True
