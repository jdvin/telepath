import argparse
from contextlib import nullcontext
from dataclasses import asdict
import math

from datasets import load_from_disk, DatasetDict
from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as torch_mp
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from .utils.train_utils import (
    load_yaml,
    run_eval,
    setup,
    cleanup,
    log_model_details,
    Metric,
    MetricLogRule,
    log_metrics,
    get_microbatch,
    get_dataloaders,
    get_validation_step_indexes,
    get_dataloader_iterator,
)

from .src.telepath import Telepath, TelepathConfig

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--run_group", type=str, required=True)
parser.add_argument("--model_config_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--eval_first", action="store_true")
parser.add_argument("--device", type=str)
parser.add_argument(
    "--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp32"
)
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--checkpoints", action="store_true", default=False)

NUM_EPOCHS = 100
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
VALIDATION_INTERVAL = 1  # Measured in fractions of an epoch. Rounded up to the nearest multiple of the number of steps in an epoch.
LOG_INTERVAL = 1  # Measured in steps.
MAX_LR = 1e-5
WEIGHT_DECAY = 0  # 1e-6


def main(
    rank: int,
    world_size: int,
    run_name: str,
    run_group: str,
    dataset_path: str,
    eval_first: bool,
    device: str,
    dtype: str,
    model_config_path: str,
    checkpoints: bool,
):
    assert BATCH_SIZE % MICRO_BATCH_SIZE == 0
    assert MICRO_BATCH_SIZE * world_size <= BATCH_SIZE
    grad_accum_steps = BATCH_SIZE // (MICRO_BATCH_SIZE * world_size)

    config = TelepathConfig(**load_yaml(model_config_path))

    setup(rank, world_size, logger, run_group, run_name, config)

    logger.info("Loading dataset.")
    ds = load_from_disk(dataset_path)
    assert isinstance(ds, DatasetDict)

    logger.info("Creating data loaders.")
    # Create data loaders.
    (
        train_dataloader,
        train_sampler,
        val_dataloader,
        val_sampler,
    ) = get_dataloaders(ds, MICRO_BATCH_SIZE, rank, world_size)
    # Full steps per epoch is the number of batches in the training set.
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    validation_step_indexes = get_validation_step_indexes(
        VALIDATION_INTERVAL, steps_per_epoch
    )

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]
    device_type = "cuda" if "cuda" in device else "cpu"
    scaler_context = (
        nullcontext()
        if dtype == "fp32"
        else autocast(device_type=device_type, dtype=torch_dtype)
    )
    scaler = GradScaler(enabled=dtype != "fp32")

    logger.info("Creating model instance.")
    # Create model.
    model = WhisperClassifier(config).to(rank)
    logger.info(f"|Max LR: {MAX_LR} | Weight Decay: {WEIGHT_DECAY}|")
    log_model_details(model)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    logger.info("Creating optimizer.")

    optim, lr_scheduler = model.module.configure_optimizers(
        num_batches=steps_per_epoch * NUM_EPOCHS,
        max_lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    metrics = dict(
        train_loss=Metric(
            torch.tensor([0.0], device=rank), MetricLogRule.EVERY_STEP, reset=True
        ),
        val_loss=Metric(
            torch.tensor([0.0], device=rank), MetricLogRule.MANUAL, reset=True
        ),
        val_accuracy=Metric(
            torch.tensor([0.0], device=rank), MetricLogRule.MANUAL, reset=True
        ),
        microstep=Metric(1, MetricLogRule.EVERY_STEP),
        step=Metric(1, MetricLogRule.EVERY_STEP),
        epochstep=Metric(1, MetricLogRule.EVERY_STEP),
        epochmicrostep=Metric(1, MetricLogRule.EVERY_STEP),
        lr=Metric(0, MetricLogRule.EVERY_STEP),
        epoch=Metric(1, MetricLogRule.EVERY_STEP),
    )
    metrics["lr"].value = lr_scheduler.get_last_lr()[0]
    logger.info("Spinning Dataloader.")
    train_dataloader_iterator = get_dataloader_iterator(
        train_dataloader, train_sampler, metrics["epoch"].value
    )
    micro_batch = get_microbatch(train_dataloader_iterator, rank)
    logger.info("Beginning Training.")
    train_pbar = tqdm(
        total=steps_per_epoch,
        desc=f"Epoch {metrics['epoch'].value}/{NUM_EPOCHS}.",
        leave=False,
        disable=rank != 0,
    )

    if eval_first:
        run_eval(
            model=model.module,
            val_dataloader=val_dataloader,
            val_sampler=val_sampler,
            metrics=metrics,
            device=rank,
        )
    while True:
        # Forward and backward pass.
        with scaler_context:
            loss, _ = model.module.step(micro_batch)
            loss = loss / grad_accum_steps
        metrics["train_loss"].value += loss.item()
        # Get the next batch straight away without blocking whilst we compute the backward pass, unless we are at the end of the epoch.
        if metrics["epochmicrostep"].value < len(train_dataloader):
            micro_batch = get_microbatch(train_dataloader_iterator, rank)
        scaler.scale(loss).backward()  # type: ignore

        # If we are still accumulating gradients then skip gradient application and logging.
        if metrics["microstep"].value % grad_accum_steps != 0:
            metrics["microstep"].value += 1
            metrics["epochmicrostep"].value = metrics["microstep"].value - (
                len(train_dataloader) * (metrics["epoch"].value - 1)
            )
            continue

        # Gradient application and logging.
        scaler.step(optim)
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step()
        train_pbar.update()
        if metrics["epochstep"].value in validation_step_indexes:
            run_eval(
                model=model.module,
                val_dataloader=val_dataloader,
                val_sampler=val_sampler,
                metrics=metrics,
                device=rank,
            )

        if metrics["step"].value % LOG_INTERVAL == 0:
            log_metrics(metrics, rank, is_distributed=world_size > 1)

        if metrics["step"].value % steps_per_epoch == 0:
            if rank == 0 and checkpoints:
                torch.save(
                    model.module.state_dict(),
                    f"checkpoints/{run_name}/whisper_classifier_ep{metrics['epoch'].value}.pt",
                )
            metrics["epoch"].value += 1
            train_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch: {metrics['epoch'].value}/{NUM_EPOCHS}.",
                leave=False,
                disable=rank != 0,
            )
            train_dataloader_iterator = get_dataloader_iterator(
                train_dataloader, train_sampler, metrics["epoch"].value
            )
            # We have to do this again because this gets skipped at the end of the last epoch.
            micro_batch = get_microbatch(train_dataloader_iterator, rank)

        if metrics["epoch"].value == NUM_EPOCHS + 1:
            break

        metrics["step"].value += 1
        metrics["epochstep"].value = metrics["step"].value - (
            steps_per_epoch * (metrics["epoch"].value - 1)
        )
        metrics["microstep"].value += 1
        metrics["epochmicrostep"].value = metrics["microstep"].value - (
            len(train_dataloader) * (metrics["epoch"].value - 1)
        )
        metrics["lr"].value = lr_scheduler.get_last_lr()[0]

    cleanup(world_size)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.world_size == 1:
        main(rank=0, **vars(args))
    else:
        torch_mp.spawn(
            main,
            args=(
                args.world_size,
                args.run_name,
                args.run_group,
                args.dataset_path,
                args.eval_first,
                args.device,
                args.dtype,
                args.model_config_path,
                args.checkpoints,
            ),
            nprocs=args.world_size,
            join=True,
        )
