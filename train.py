import argparse
from contextlib import nullcontext
import os
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
import pandas as pd
from utils.data_utils import extract_things_100ms_ds, get_collate_fn

from pytorch_memlab import MemReporter

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
    MetricManager,
)

from src.telepath import TelepathConfig, TelepathTrainer, configure_optimizers


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
    is_test_run: bool,
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
    grad_accum_steps = cfg.batch_size // (cfg.train_micro_batch_size * cfg.world_size)
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
    is_main_process = rank == 0
    logger.info("Creating model instance.")
    # Create model.
    model_config = TelepathConfig(**load_yaml(model_config_path))
    model: TelepathTrainer = TelepathTrainer(model_config, rank, world_size)

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[cfg.dtype]

    model.to(rank, dtype=torch_dtype)
    log_model_details(model)
    # reporter = MemReporter(model)
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
            is_test_run=is_test_run,
        )
        dist.barrier()
    else:
        dist.barrier()
        ds = extract_things_100ms_ds(root_dir=cfg.dataset_path, subjects=cfg.subjects)
    collate_fn = get_collate_fn(
        model.module.tokenizer,
        model_config.text_encoder_stop_token,
        model_config.text_encoder_stop_token,
        get_spectrogram=False,
    )
    logger.info("Creating data loaders.")
    # Create data loaders.
    (
        train_dataloader,
        train_sampler,
        val_dataloader,
        val_sampler,
    ) = get_dataloaders(
        ds,
        cfg.train_micro_batch_size,
        cfg.val_micro_batch_size,
        rank,
        world_size,
        collate_fn,
    )
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

    optim, lr_scheduler = configure_optimizers(
        model.module.parameters,
        num_batches=steps_per_epoch * cfg.num_epochs,
        max_lr=cfg.max_lr,
        weight_decay=cfg.weight_decay,
        warmup_frac=cfg.warmup_frac,
    )
    metrics = MetricManager(
        device=rank,
        world_size=world_size,
        is_main_process=is_main_process,
        log_interval=cfg.log_interval,
        batch_size=cfg.batch_size,
    )

    metrics.lr.update(lr_scheduler.get_last_lr()[0])
    logger.info("Spinning Dataloader.")
    train_dataloader_iterator = get_dataloader_iterator(
        train_dataloader, train_sampler, metrics.epoch.value  # type: ignore
    )
    micro_batch = get_microbatch(train_dataloader_iterator, rank, torch_dtype)
    logger.info("Beginning Training.")
    train_pbar = tqdm(
        total=steps_per_epoch,
        desc=f"Epoch {metrics.epoch.value}/{cfg.num_epochs}.",
        leave=False,
        disable=rank not in {0, "cuda:0", "cuda"},
    )
    # logger.debug("====Post Init====")
    # reporter.report(device=rank)
    dist.barrier()
    if eval_first:
        run_eval(
            model=model.module,
            val_dataloader=val_dataloader,
            val_sampler=val_sampler,
            metrics=metrics,
            device=rank,
            dtype=torch_dtype,
        )
    while True:
        is_accumulating = (
            metrics.microstep.value % grad_accum_steps != 0
            and metrics.epoch_microstep.value != len(train_dataloader)
        )
        # Forward and backward pass.
        # Do no sync gradients whilst accumulating.
        ddp_context = (
            nullcontext() if world_size == 1 or is_accumulating else model.no_sync()
        )
        # torch.cuda.memory._record_memory_history()
        with ddp_context:
            with scaler_context:
                loss, _, _ = model.module.step(micro_batch)
                # logger.debug("====Forward Pass====")
                # reporter.report()
                loss = loss / grad_accum_steps
            metrics.train_loss.update(loss.item())
            # Get the next batch straight away without blocking whilst we compute the backward pass,
            # unless we are at the end of the epoch.
            if metrics.epoch_microstep.value < len(train_dataloader) - 1:
                micro_batch = get_microbatch(
                    train_dataloader_iterator, rank, torch_dtype
                )
            scaler.scale(loss).backward(retain_graph=True)  # type: ignore

        # torch.cuda.memory._dump_snapshot("new_snapshot.pickle")
        # break
        # If we are still accumulating gradients then skip gradient application and logging.
        if is_accumulating:
            metrics.microstep.update(1)
            metrics.epoch_microstep.update(
                (metrics.microstep.value, len(train_dataloader))
            )
            continue
        if cfg.grad_clip > 0:
            scaler.unscale_(optim)
            metrics.train_gradnorm.update(
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            )
        # Gradient application and logging.
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step()
        train_pbar.update()
        if metrics.epoch_step.value in validation_step_indexes:
            run_eval(
                model=model.module,
                val_dataloader=val_dataloader,
                val_sampler=val_sampler,
                metrics=metrics,
                device=rank,
                dtype=torch_dtype,
            )
        metrics.log()
        if metrics.epoch_microstep.value == len(train_dataloader):
            if is_main_process and checkpoints:
                torch.save(
                    model.module,
                    f"checkpoints/{run_name}/{cfg.run_project}_{cfg.run_group}_{cfg.run_name}_ep{metrics['epoch'].value}.pt",
                )
            metrics.epoch.update(1)
            train_pbar = tqdm(
                total=steps_per_epoch,
                desc=f"Epoch: {metrics.epoch.value}/{cfg.num_epochs}.",
                leave=False,
                disable=not is_main_process,
            )
            train_dataloader_iterator = get_dataloader_iterator(
                train_dataloader, train_sampler, metrics.epoch.value
            )
            # Get the first microbatch of the new epoch.
            micro_batch = get_microbatch(train_dataloader_iterator, rank, torch_dtype)

        if metrics.epoch.value == cfg.num_epochs + 1:
            logger.info("Training complete.")
            break
        metrics.step.update(1)
        metrics.microstep.update(1)
        metrics.temperature.update(model.module.t.item())
        metrics.bias.update(model.module.b.item())
        metrics.epoch_step.update((metrics.step.value, steps_per_epoch))
        metrics.epoch_microstep.update((metrics.microstep.value, len(train_dataloader)))
        metrics.lr.update(lr_scheduler.get_last_lr()[0])
    cleanup(world_size)


if __name__ == "__main__":
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
    parser.add_argument("--is-test-run", action="store_true", default=False)

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
                args.checkpoints,
                args.reset_data_cache,
                args.is_test_run,
            ),
            nprocs=args.world_size,
            join=True,
        )
