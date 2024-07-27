import argparse
from contextlib import nullcontext
import math

from loguru import logger
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as torch_mp
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import wandb
from utils.data_utils import extract_things_100ms_ds, get_collate_fn, get_dataset_dict

from utils.train_utils import (
    run_eval,
    setup,
    cleanup,
    load_yaml,
    log_model_details,
    get_microbatch,
    get_dataloaders,
    get_validation_step_indexes,
    get_dataloader_iterator,
    TrainingConfig,
)

from utils.metrics import (
    MetricKey,
    get_metrics,
    Metric,
)

from src.telepath import TelepathConfig, Telepath

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument("--run-project", type=str, required=True)
parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--run-group", type=str, required=True)
parser.add_argument("--eval-first", action="store_true")
parser.add_argument("--device", type=str)
parser.add_argument("--training-config-path", type=str, default=None)
parser.add_argument("--model-config-path", type=str, default=None)
parser.add_argument("--world-size", type=int, default=1)
parser.add_argument("--checkpoints", action="store_true", default=False)
parser.add_argument("--reset-data-cache", action="store_true", default=False)


def main(
    rank: int,
    world_size: int,
    training_config_path: str,
    model_config_path: str,
    run_project: str,
    run_group: str,
    run_name: str,
    eval_first: bool,
    device: str,
    checkpoints: bool,
    reset_data_cache: bool,
):
    cfg = TrainingConfig(
        **load_yaml(training_config_path),
        training_config_path=training_config_path,
        model_config_path=model_config_path,
        world_size=world_size,
        run_project=run_project,
        run_name=run_name,
        run_group=run_group,
        eval_first=eval_first,
        device=device,
        checkpoints=checkpoints,
    )
    grad_accum_steps = cfg.batch_size // (cfg.micro_batch_size * cfg.world_size)
    model_config = TelepathConfig(**load_yaml(model_config_path))

    setup(
        rank=rank,
        world_size=world_size,
        logger=logger,
        run_project=run_project,
        run_group=run_group,
        run_name=run_name,
        checkpoints=checkpoints,
        training_config=cfg,
        model_config=model_config,
    )
    rank = rank if world_size > 1 else device
    is_main_process = rank in {"cuda:0", "cuda", 0, "cpu", "mps"}
    logger.info("Creating model instance.")
    # Create model.
    model_config = TelepathConfig(**load_yaml(model_config_path))
    model: Telepath = Telepath(model_config)

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[cfg.dtype]
    model.to(rank, dtype=torch_dtype)
    assert isinstance(model, Telepath)
    assert not isinstance(model.module.configure_optimizers, torch.Tensor)
    assert isinstance(model.module, nn.Module)
    log_model_details(model.module)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])  # type: ignore
    logger.info(f"Loading dataset from {cfg.dataset_path}.")
    # The first rank goes ahead to create the dataset if it does not already exist, before the other ranks then load it.
    # This is probably quite a strange pattern, but it is the simplest way to implement this behaviour.
    # TODO: Distributed dataset creation.
    if is_main_process:
        ds = extract_things_100ms_ds(
            root_dir=cfg.dataset_path,
            subjects=cfg.subjects,
            reset_cache=reset_data_cache,
        )
    if world_size > 1:
        dist.barrier()
        if rank != 0:
            ds = extract_things_100ms_ds(
                root_dir=cfg.dataset_path, subjects=cfg.subjects
            )
    collate_fn = get_collate_fn(
        model.tokenizer,
        model.start_sequence,
        model.stop_token,
        model.stop_token,
        model_config.n_freqs * 2,
        model.config.fft_hop_length,
    )
    logger.info("Creating data loaders.")
    # Create data loaders.
    (
        train_dataloader,
        train_sampler,
        val_dataloader,
        val_sampler,
    ) = get_dataloaders(ds, cfg.micro_batch_size, rank, world_size, collate_fn)
    # Steps per epoch is the number of batches in the training set.
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    validation_step_indexes = get_validation_step_indexes(
        cfg.validation_interval, steps_per_epoch
    )

    device_type = "cuda" if "cuda" in device else "cpu"
    scaler_context = (
        nullcontext()
        if cfg.dtype == "fp32"
        else autocast(device_type=device_type, dtype=torch_dtype)
    )
    scaler = GradScaler(enabled=cfg.dtype == "fp16")

    logger.info("Creating optimizer.")

    optim, lr_scheduler = model.module.configure_optimizers(
        num_batches=steps_per_epoch * cfg.num_epochs,
        max_lr=cfg.max_lr,
        weight_decay=cfg.weight_decay,
        warmup_frac=cfg.warmup_frac,
    )
    metrics: dict[str, Metric] = get_metrics(
        [
            MetricKey.TRAIN_LOSS,
            MetricKey.TRAIN_GRADNORM,
            MetricKey.MICROSTEP,
            MetricKey.STEP,
            MetricKey.EPOCHSTEP,
            MetricKey.EPOCHMICROSTEP,
            MetricKey.LR,
            MetricKey.EPOCH,
            MetricKey.VAL_LOSS,
            MetricKey.VAL_ACCURACY,
            MetricKey.VAL_GENERATIONS,
        ],
        device=rank,
        world_size=world_size,
    )

    metrics["lr"].update(lr_scheduler.get_last_lr()[0])
    logger.info("Spinning Dataloader.")
    train_dataloader_iterator = get_dataloader_iterator(
        train_dataloader, train_sampler, metrics["epoch"].value  # type: ignore
    )
    micro_batch = get_microbatch(train_dataloader_iterator, rank)
    logger.info("Beginning Training.")
    train_pbar = tqdm(
        total=steps_per_epoch,
        desc=f"Epoch {metrics['epoch'].value}/{cfg.num_epochs}.",
        leave=False,
        disable=rank not in {0, "cuda:0", "cuda"},
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
        # Do no sync gradients whilst accumulating.
        ddp_context = (
            nullcontext()
            if world_size == 1 or metrics["microstep"].value % grad_accum_steps != 0
            else model.no_sync()
        )
        # torch.cuda.memory._record_memory_history()
        with ddp_context:
            with scaler_context:
                _, _, loss = model.module.step(micro_batch)
                loss = loss / grad_accum_steps
            metrics["train_loss"].update(loss.item())
            # Get the next batch straight away without blocking whilst we compute the backward pass,
            # unless we are at the end of the epoch.
            if metrics["epochmicrostep"].value < len(train_dataloader) - 1:
                micro_batch = get_microbatch(train_dataloader_iterator, rank)
            scaler.scale(loss).backward()  # type: ignore

        # torch.cuda.memory._dump_snapshot("my_snapshot_checkpointed.pickle")
        # break
        # If we are still accumulating gradients then skip gradient application and logging.
        if metrics["microstep"].value % grad_accum_steps != 0:
            metrics["microstep"].update(1)
            metrics["epochmicrostep"].update(
                (metrics["microstep"].value, len(train_dataloader))
            )
            continue
        if cfg.grad_clip > 0:
            scaler.unscale_(optim)
            metrics["train_gradnorm"].update(
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            )
        # Gradient application and logging.
        scaler.step(optim)
        scaler.update()
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

        if metrics["step"].value % cfg.log_interval == 0:
            for key, metric in metrics.items():
                if metric.log_every_step:
                    metric.log(key)
            if is_main_process:
                wandb.log({}, commit=True)
        if metrics["step"].value % steps_per_epoch == 0:
            if is_main_process and checkpoints:
                torch.save(
                    model.module,
                    f"checkpoints/{run_name}/{cfg.run_project}_{cfg.run_group}_{cfg.run_name}_ep{metrics['epoch'].value}.pt",
                )
            metrics["epoch"].update(1)
            train_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch: {metrics['epoch'].value}/{cfg.num_epochs}.",
                leave=False,
                disable=rank not in {0, "cuda:0", "cuda"},
            )
            train_dataloader_iterator = get_dataloader_iterator(
                train_dataloader, train_sampler, metrics["epoch"].value
            )
            # Get the first microbatch of the new epoch.
            micro_batch = get_microbatch(train_dataloader_iterator, rank)

        if metrics["epoch"].value == cfg.num_epochs + 1:
            logger.info("Training complete.")
            break

        metrics["step"].update(1)
        metrics["epochstep"].update((metrics["step"].value, steps_per_epoch))
        metrics["microstep"].update(1)
        metrics["epochmicrostep"].update(
            (metrics["microstep"].value, len(train_dataloader))
        )
        metrics["lr"].update(lr_scheduler.get_last_lr()[0])
    cleanup(world_size)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.world_size == 1:
        main(rank=0, **vars(args))
    else:
        torch_mp.spawn(  # type: ignore
            main,
            args=(
                args.world_size,
                args.training_config_path,
                args.model_config_path,
                args.run_project,
                args.run_group,
                args.run_name,
                args.eval_first,
                args.device,
                args.dtype,
                args.checkpoints,
                args.reset_data_cache,
            ),
            nprocs=args.world_size,
            join=True,
        )
