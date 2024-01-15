import argparse
from contextlib import nullcontext
import os
import random

from loguru import logger
import torch
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import wandb

from utils.data_utils import get_dataset
from utils.train_utils import (
    load_yaml,
    run_eval,
    Metric,
    MetricLogRule,
    log_metrics,
    DataLoader,
)
from src.wrapper import (
    TelepathWrapper,
)

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--run_group", type=str, required=True)
parser.add_argument("--model_config_path", type=str, required=True)
parser.add_argument("--optimizer_config_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--add_transform", action="store_true")
parser.add_argument("--eval_first", action="store_true")
parser.add_argument("--max_length", type=int)
parser.add_argument("--num_channels", type=int)
parser.add_argument("--num_samples", type=int)
parser.add_argument("--device", type=str)
parser.add_argument(
    "--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16"
)
parser.add_argument("--compile", action="store_true")

NUM_EPOCHS = 5
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 1
VALIDATION_INTERVAL = 0.1
LOG_INTERVAL = 1


def main(
    run_name: str,
    run_group: str,
    dataset_path: str,
    add_transform: bool,
    eval_first: bool,
    max_length: int,
    num_channels: int,
    num_samples: int,
    model_config_path: str,
    optimizer_config_path: str,
    device: str,
    dtype: str,
    compile: bool,
):
    torch.manual_seed(42)
    random.seed(42)

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
    wmodel = TelepathWrapper(
        model_config_path=model_config_path,
        optimizer_config_path=optimizer_config_path,
        device=device,
    )

    if compile:
        logger.info("Compiling model.")
        wmodel.model = torch.compile(wmodel.model)

    if add_transform:
        logger.info("Adding transform to dataset.")
        assert max_length is not None
        assert num_channels is not None
        assert num_samples is not None
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            wmodel.config.tokenizer_path
        )
        assert isinstance(
            tokenizer, PreTrainedTokenizerFast
        ), f"Expected PreTrainedTokenizer, got {type(tokenizer)}."
    else:
        tokenizer = None  # type: ignore
    logger.info("Loading dataset.")
    ds = get_dataset(
        path=dataset_path,
        add_transform=add_transform,
        tokenizer=tokenizer,
        start_token_id=wmodel.config.gpt_start_token,
        stop_token_id=wmodel.config.gpt_stop_token,
        pad_token_id=wmodel.config.gpt_stop_token,
        max_length=max_length,
        num_channels=num_channels,
        num_samples=num_samples,
    )
    config = {
        "model": load_yaml(model_config_path),
        "optim": load_yaml(model_config_path),
    }
    logger.info("Creating data loaders.")
    # Create data loaders.
    train_dataloader = DataLoader(
        ds["train"], batch_size=MICRO_BATCH_SIZE, device=device, shuffle=True
    )
    val_dataloader = DataLoader(
        ds["test"], batch_size=MICRO_BATCH_SIZE, device=device, shuffle=True
    )

    logger.info("Creating optimizer.")
    grad_accum_steps = BATCH_SIZE // MICRO_BATCH_SIZE
    optim, lr_scheduler = wmodel.configure_optimizers(
        num_batches=len(train_dataloader) * NUM_EPOCHS * grad_accum_steps
    )
    metrics = dict(
        train_loss=Metric(0, MetricLogRule.EVERY_STEP, reset=True),
        val_loss=Metric(0, MetricLogRule.MANUAL, reset=True),
        val_accuracy=Metric(0, MetricLogRule.MANUAL, reset=True),
        microstep=Metric(1, MetricLogRule.EVERY_STEP),
        step=Metric(1, MetricLogRule.EVERY_STEP),
        lr=Metric(0, MetricLogRule.EVERY_STEP),
        epoch=Metric(1, MetricLogRule.EVERY_STEP),
        generations=Metric(
            wandb.Table(columns=["target", "output"]),
            MetricLogRule.MANUAL,
            reset=True,
            suffixes=["step"],
        ),
    )
    metrics["lr"].value = lr_scheduler.get_last_lr()[0]
    logger.info("Spinning Dataloader.")
    micro_batch = train_dataloader.get_batch()
    logger.info("Beginning Training.")
    train_pbar = tqdm(
        total=len(train_dataloader) * grad_accum_steps,
        desc=f"Epoch {metrics['epoch'].value}/{NUM_EPOCHS}.",
    )
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")

    assert not os.path.isdir(f"checkpoints/{run_name}")
    os.makedirs(f"checkpoints/{run_name}")

    if eval_first:
        run_eval(wmodel=wmodel, val_dataloader=val_dataloader, metrics=metrics)

    wandb.init(project="telepath", group=run_group, name=run_name, config=config)
    while True:
        # Forward and backward pass.
        with scaler_context:
            loss = wmodel.step(micro_batch)
            loss = loss / grad_accum_steps
        metrics["train_loss"].value += loss.item()
        # Get the batch straight away without blocking whilst we compute the backward pass.
        micro_batch = train_dataloader.get_batch()
        scaler.scale(loss).backward()

        # If we are still accumulating gradients, then skip gradient application and logging.
        if metrics["microstep"].value % grad_accum_steps != 0:
            metrics["microstep"].value += 1
            continue

        # Greidnt application and logging.
        scaler.step(optim)
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step()
        train_pbar.update()
        if (
            metrics["step"].value % int(len(train_dataloader) * VALIDATION_INTERVAL)
            == 0
        ):
            run_eval(wmodel=wmodel, val_dataloader=val_dataloader, metrics=metrics)
        if metrics["step"].value % LOG_INTERVAL == 0:
            log_metrics(metrics)

        metrics["step"].value += 1
        metrics["microstep"].value += 1
        metrics["lr"].value = lr_scheduler.get_last_lr()[0]

        if metrics["step"].value % len(train_dataloader) == 0:
            torch.save(
                wmodel.model.state_dict(),
                f"checkpoints/{run_name}/telepath_ep{metrics['epoch'].value}.pt",
            )
            metrics["epoch"].value += 1
            train_pbar = tqdm(
                total=len(train_dataloader) * grad_accum_steps,
                desc=f"Epoch: {metrics['epoch'].value}/{NUM_EPOCHS}.",
            )

        if metrics["epoch"].value == NUM_EPOCHS + 1:
            break


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
