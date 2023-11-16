import argparse

from loguru import logger
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import wandb

from utils.data_utils import get_dataset
from utils.train_utils import load_yaml, TrainMetrics, DataLoader
from src.wrapper import (
    TelepathWrapper,
)

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--model_config_path", type=str, required=True)
parser.add_argument("--optimizer_config_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--add_transform", action="store_true")
parser.add_argument("--tokenizer_path", type=str)
parser.add_argument("--max_length", type=int)
parser.add_argument("--num_channels", type=int)
parser.add_argument("--num_samples", type=int)
parser.add_argument("--device", type=str, default="mps")

NUM_EPOCHS = 5
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
VALIDATION_INTERVAL = 0.1
LOG_INTERVAL = 1


def main(
    run_name: str,
    dataset_path: str,
    add_transform: bool,
    tokenizer_path: str,
    max_length: int,
    num_channels: int,
    num_samples: int,
    model_config_path: str,
    optimizer_config_path: str,
    device: str,
):
    torch.manual_seed(42)

    if add_transform:
        logger.info("Adding transform to dataset.")
        assert tokenizer_path is not None
        assert max_length is not None
        assert num_channels is not None
        assert num_samples is not None
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            tokenizer_path
        )
        assert isinstance(
            tokenizer, PreTrainedTokenizerFast
        ), f"Expected PreTrainedTokenizer, got {type(tokenizer)}."
    else:
        tokenizer = None  # type: ignore
    logger.info("Creating model instance.")
    # Create model.
    wmodel = TelepathWrapper(
        model_config_path=model_config_path,
        optimizer_config_path=optimizer_config_path,
        device=device,
    )
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
    wandb.init(project="telepath", name=run_name, config=config)
    logger.info("Creating data loaders.")
    # Create data loaders.
    train_dataloader = DataLoader(
        ds["train"], batch_size=MICRO_BATCH_SIZE, device=device, shuffle=True
    )
    val_dataloader = DataLoader(
        ds["test"], batch_size=MICRO_BATCH_SIZE, device=device, shuffle=True
    )

    logger.info("Creating optimizer.")
    optim, lr_scheduler = wmodel.configure_optimizers(num_batches=len(train_dataloader))
    grad_accum_steps = BATCH_SIZE // MICRO_BATCH_SIZE
    metrics = TrainMetrics()
    metrics.lr = lr_scheduler.get_last_lr()[0]
    logger.info("Spinning dataloader.")
    micro_batch = train_dataloader.get_batch()
    logger.info("Beginning Training.")
    train_pbar = tqdm(
        total=len(train_dataloader), desc=f"Epoch {metrics.epoch}/{NUM_EPOCHS}."
    )
    while True:
        loss = wmodel.step(micro_batch)
        loss = loss / grad_accum_steps
        metrics.train_loss += loss.item()
        # Get the batch straight away without blocking whilst we compute the backward pass.
        micro_batch = train_dataloader.get_batch()
        loss.backward()

        # If we are still accumulating gradients, then skip gradient application and logging.
        # First term is a HACK to stop the step logic from being run on the first microstep.
        if metrics.microstep and metrics.microstep % grad_accum_steps != 0:
            metrics.microstep += 1
            continue

        optim.step()
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step()
        train_pbar.update()
        if metrics.step % int(len(train_dataloader) * VALIDATION_INTERVAL) == 0:
            metrics.val_loss = 0
            val_pbar = tqdm(total=len(val_dataloader), desc="Running validation")
            for micro_batch in val_dataloader:
                val_pbar.update()
                metrics.val_loss += wmodel.step(micro_batch).item()
            metrics.val_loss = metrics.val_loss / len(val_dataloader)

        if metrics.step % LOG_INTERVAL == 0:
            metrics.log()

        metrics.train_loss = 0
        metrics.step += 1
        metrics.microstep += 1
        metrics.lr = lr_scheduler.get_last_lr()[0]

        if metrics.step % len(train_dataloader) == 0:
            metrics.epoch += 1
            train_pbar = tqdm(
                total=len(train_dataloader),
                description=f"Epoch: {metrics.epoch}/{NUM_EPOCHS}.",
            )

        if metrics.epoch == NUM_EPOCHS:
            break

    # save
    torch.save(wmodel.model.state_dict(), f"chekpoints/{run_name}_model.pt")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        run_name=args.run_name,
        dataset_path=args.dataset_path,
        add_transform=args.add_transform,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        num_channels=args.num_channels,
        num_samples=args.num_samples,
        model_config_path=args.model_config_path,
        optimizer_config_path=args.optimizer_config_path,
        device=args.device,
    )
