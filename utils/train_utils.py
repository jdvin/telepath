from dataclasses import asdict, dataclass
from collections import defaultdict
import math
from typing import Any, Callable, Iterator
import os
import random

from loguru import logger
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from tqdm import tqdm
import numpy as np
import wandb
import yaml

from src.telepath import Telepath, TelepathConfig

from utils.metrics import Metric, MetricType


@dataclass
class TrainingConfig:
    """Configuration for training a model.

    Attributes:
        model_key: The key for the target model class in the `MODEL_REGISTRY`.
        dataset_path: The path to the dataset.
        num_epochs: The number of epochs to train the model for.
        batch_size: The total batch size. Must be divisible by `micro_batch_size`. Gradients are accumulated over `batch_size` / `micro_batch_size` steps.
        micro_batch_size: The number of examples inference is performed over per micro step.
        validation_interval: The interval at which validation is performed. Measured in fractions of an epoch. Rounded up to the nearest multiple of the number of steps in an epoch.
        log_interval: The interval at which metrics are logged. Measured in steps.
        max_lr: The maximum learning rate for the optimizer.
        weight_decay: The weight decay for the optimizer.
    """

    world_size: int
    run_name: str
    run_group: str
    run_project: str
    eval_first: bool
    device: str
    dtype: str
    training_config_path: str
    model_config_path: str
    checkpoints: bool
    dataset_path: str
    subjects: list[int]
    things_metadata_path: str
    num_epochs: int
    batch_size: int
    micro_batch_size: int
    validation_interval: float
    log_interval: float  # Measured in steps.
    max_lr: float
    weight_decay: float
    warmup_frac: float
    grad_clip: float

    def __post_init__(self):
        assert self.batch_size % self.micro_batch_size == 0
        assert self.micro_batch_size * self.world_size <= self.batch_size


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_microbatch(
    dataloader_iterator: Iterator,
    device: str | int,
) -> dict[str, torch.Tensor]:
    micro_batch = next(dataloader_iterator)
    return {
        k: v.pin_memory().to(device=device, non_blocking=True)
        if isinstance(device, int)
        else v.to(device=device)
        for k, v in micro_batch.items()
        if isinstance(v, torch.Tensor)
    }


def setup(
    rank: int,
    world_size: int,
    logger: Any,
    run_project: str,
    run_group: str,
    run_name: str,
    checkpoints: bool,
    training_config: TrainingConfig,
    model_config: TelepathConfig,
):
    """Setup the environment for training."""
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    if rank != 0:
        # Suppress output from all ranks except rank 0.
        logger.remove()
        pass
    else:
        # Initialize checkpoints directory and wandb logging for the first rank.
        if checkpoints:
            if not os.path.isdir("checkpoints"):
                os.makedirs("checkpoints")
            assert not os.path.isdir(f"checkpoints/{run_name}")
            os.makedirs(f"checkpoints/{run_name}")
        wandb.init(
            project=run_project,
            group=run_group,
            name=run_name,
            config=dict(
                training_config=asdict(training_config),
                model_config=asdict(model_config),
            ),
        )
    if world_size > 1:
        torch.cuda.set_device(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup(world_size: int):
    if world_size > 1:
        torch.distributed.destroy_process_group()


def format_number(number: int) -> str:
    """Format a number as a string with K, M, or B suffixes."""
    if number < 1_000:
        return str(number)
    if number < 1_000_000:
        return f"{number / 1_000:.2f}K"
    if number < 1_000_000_000:
        return f"{number / 1_000_000:.2f}M"
    return f"{number / 1_000_000_000:.2f}B"


def count_params(model: torch.nn.Module) -> dict[str, str]:
    """Count the number of trainable and untrainable parameters in a model."""
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrained = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "Trained": format_number(trained),
        "Untrained": format_number(untrained),
        "Total": format_number(trained + untrained),
    }


def log_model_details(model: torch.nn.Module) -> None:
    """Log the architecture and parameter counts of a model."""
    logger.info(f"Architecture:\n{model}")
    param_counts = count_params(model)
    logger.info(
        "| "
        + " | ".join(
            [f"{key} Parameters: {value}" for key, value in param_counts.items()]
        )
    )


def get_validation_step_indexes(
    validation_interval: float, steps_per_epoch: int
) -> set[int]:
    """Get the indexes of the steps at which validation should be performed.

    Validation interval is measured in fractions of an epoch."""
    assert 1 >= validation_interval > 0
    steps_per_validation = math.ceil(validation_interval * steps_per_epoch)
    validation_step_indexes = set(
        range(steps_per_validation, steps_per_epoch + 1, steps_per_validation)
    )
    if steps_per_validation % steps_per_epoch != 0:
        validation_step_indexes.add(steps_per_epoch)
    return validation_step_indexes


def get_dataloaders(
    dataset: dict[str, np.memmap],
    microbatch_size: int,
    rank: int,
    world_size: int,
    collate_fn: Callable[[list], dict],
) -> tuple[DataLoader, Sampler | None, DataLoader, Sampler | None]:
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset["train"], num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            dataset["test"], num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler, val_sampler = None, None
    train_dataloader = DataLoader(
        dataset["train"],  # type: ignore
        batch_size=microbatch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset["test"],  # type: ignore
        batch_size=microbatch_size,
        shuffle=val_sampler is None,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, train_sampler, val_dataloader, val_sampler


def get_dataloader_iterator(
    dataloader: DataLoader, sampler: Sampler | None, epoch: int
) -> Iterator:
    """Get an iterator for a dataloader, with the sampler set to the correct epoch."""
    if isinstance(sampler, DistributedSampler):
        # Required to ensure that the order is different each epoch.
        sampler.set_epoch(epoch)
    return iter(dataloader)


@torch.no_grad()
def run_eval(
    model: Telepath,
    val_dataloader: DataLoader,
    val_sampler: Sampler | None,
    metrics: dict[str, Metric],
    device: str | int,
):
    """Run evaluation on the validation sets."""
    model.eval()
    val_pbar = tqdm(
        total=len(val_dataloader),
        desc=f"Running validation.",
        leave=False,
        disable=device not in {0, "cuda:0", "cuda"},
    )
    val_dataloader_iterator = get_dataloader_iterator(
        val_dataloader, val_sampler, metrics["epoch"].value
    )
    # Accumulate losses, logits, and labels for all batches in the validation set.
    # More memory efficient to accumulate as we go, but if we perform the calculation at the end,
    # then it is more mathematically correct and we can more easily calculate metrics across all labels in the val set.
    losses = []
    generations = defaultdict(list)
    for _ in range(len(val_dataloader)):
        micro_batch = get_microbatch(val_dataloader_iterator, device)

        enc, _, loss = model.step(micro_batch)
        generated_ids: list[list[int]] = model.module.generate(enc=enc, device=device)
        generations["predictions"].extend(
            model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )
        generations["targets"].extend(
            model.tokenizer.batch_decode(
                micro_batch["input_ids"], skip_special_tokens=True
            )
        )

        losses.append(loss)

        val_pbar.update()

    losses = torch.tensor(losses)
    metrics["val_loss"].update(losses.mean().item())
    metrics["val_loss"].log("val_loss")
    model.train()
