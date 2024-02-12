from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import Any, Iterator
import os
import random

from datasets import DatasetDict
import torch
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from torch.distributed import ReduceOp, all_reduce, barrier
from tqdm import tqdm
import pandas as pd
import wandb
import yaml

from ..src.wrapper import TelepathWrapper

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


def get_microbatch(
    dataloader_iterator: Iterator,
    device: str | int,
) -> dict[str, torch.Tensor]:
    micro_batch = next(dataloader_iterator)
    return {
        k: v.pin_memory().to(device, non_blocking=True) for k, v in micro_batch.items()
    }


class MetricLogRule(Enum):
    EVERY_STEP = "every_step"
    ON_CHANGE = "on_change"
    MANUAL = "manual"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


@dataclass
class Metric:
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
            self._reset_value = deepcopy(self.value)


def setup(
    rank: int,
    world_size: int,
    logger: Any,
    run_group: str,
    run_name: str,
    config: Any,
):
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    if rank != 0:
        logger.remove()
    else:
        if not os.path.isdir("checkpoints"):
            os.makedirs("checkpoints")
        assert not os.path.isdir(f"checkpoints/{run_name}")
        os.makedirs(f"checkpoints/{run_name}")
        wandb.init(
            project="paraverbal-whisper",
            group=run_group,
            name=run_name,
            config=asdict(config),
        )
    if world_size > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup(world_size: int):
    if world_size > 1:
        torch.distributed.destroy_process_group()


def format_number(number: int) -> str:
    if number < 1_000:
        return str(number)
    if number < 1_000_000:
        return f"{number / 1_000:.2f}K"
    if number < 1_000_000_000:
        return f"{number / 1_000_000:.2f}M"
    return f"{number / 1_000_000_000:.2f}B"


def count_params(model: torch.nn.Module) -> dict[str, str]:
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrained = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "Trained": format_number(trained),
        "Untrained": format_number(untrained),
        "Total": format_number(trained + untrained),
    }


def get_validation_step_indexes(
    validation_interval: float, steps_per_epoch: int
) -> set[int]:
    assert 1 >= validation_interval > 0
    steps_per_validation = math.ceil(validation_interval * steps_per_epoch)
    validation_step_indexes = set(
        range(steps_per_validation, steps_per_epoch + 1, steps_per_validation)
    )
    if steps_per_validation % steps_per_epoch == 0:
        validation_step_indexes.add(steps_per_epoch)
    return validation_step_indexes


def log_metrics(metrics: dict[str, Metric], rank: int, is_distributed: bool) -> None:
    log_metrics = {}
    for key, metric in metrics.items():
        if (
            # If we log the metric every step.
            metric._update_rule == MetricLogRule.EVERY_STEP
            # Or, if log on change and the metric has changed.
            or (
                metric._update_rule == MetricLogRule.ON_CHANGE
                and metric.value != metric._past
            )
            # Or, if we manually want to log the metric and the flag is set.
            or (metric._update_rule == MetricLogRule.MANUAL and metric._log)
        ):
            metric._log = False
            metric._past = metric.value
            key = key.replace("_", "/")
            # If we want they key to change dynamically each time it gets logged (e.g., if we want to keep track of how a table will change over time).
            if metric.suffixes:
                key = f"{key}_{'_'.join([suffix + str(metrics[suffix].value) for suffix in metric.suffixes])}"
            # Tensor values are needed to simplify logging for distributed training, but we do not log them.
            if isinstance(metric.value, torch.Tensor):
                if is_distributed:
                    all_reduce(metric.value, op=ReduceOp.AVG)
                metric.value = metric.value.item()

            log_metrics[key] = metric.value
            if metric.reset:
                metric.value = deepcopy(metric._reset_value)
    if rank != 0:
        return
    wandb.log(log_metrics)


def get_dataloaders(
    dataset: DatasetDict, microbatch_size: int, rank: int, world_size: int
) -> tuple[DataLoader, Sampler | None, DataLoader, Sampler | None]:
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset["train"], num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset["test"], num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler, val_sampler = None, None
    train_dataloader = DataLoader(
        dataset["train"],  # type: ignore
        batch_size=microbatch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        dataset["test"],  # type: ignore
        batch_size=microbatch_size,
        shuffle=val_sampler is None,
        sampler=val_sampler,
    )
    return train_dataloader, train_sampler, val_dataloader, val_sampler


def get_dataloader_iterator(
    dataloader: DataLoader, sampler: Sampler | None, epoch: int
) -> Iterator:
    if isinstance(sampler, DistributedSampler):
        # Required to ensure that the order is different each epoch.
        sampler.set_epoch(epoch)
    return iter(dataloader)


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    val_sampler: Sampler | None,
    metrics: dict[str, Metric],
    device: str | int,
):
    val_pbar = tqdm(
        total=len(val_dataloader),
        desc="Running validation",
        leave=False,
        disable=device not in {0, "cuda:0"},
    )
    val_dataloader_iterator = iter(val_dataloader)
    val_dataloader_iterator = get_dataloader_iterator(
        val_dataloader, val_sampler, metrics["epoch"].value
    )

    for _ in range(len(val_dataloader)):
        micro_batch = get_microbatch(val_dataloader_iterator, device)
        loss, logits = model.step(micro_batch)
        # TODO: This is slightly mathematically incorrect for batches that are not the same size re: simpsons paradox.
        # If the val dataloader shuffles each eval cycle, then it should average out lol.
        metrics["val_loss"].value += loss / len(val_dataloader)
        metrics["val_accuracy"].value += model.classifier_head.get_accuracy(
            logits, micro_batch["labels"]
        ) / len(val_dataloader)
        val_pbar.update()

    metrics["val_loss"]._log = True
    metrics["val_accuracy"]._log = True


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
