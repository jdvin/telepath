import argparse

from datasets import load_dataset
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import wandb

from utils.data_utils import get_dataset
from utils.train_utils import load_yaml, get_batch, TrainMetrics
from src.wrapper import (
    TelepathWrapper,
    TrainingConfig,
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
parser.add_argument("--accelerator", type=str, default="mps")

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
):
    torch.manual_seed(42)

    if add_transform:
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

    # Create model.
    wmodel = TelepathWrapper(
        model_config_path=model_config_path, optimizer_config_path=optimizer_config_path
    )
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

    # Create data loaders.
    train_dataloader = DataLoader(ds["train"], batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=8)  # type: ignore
    val_dataloader = DataLoader(ds["test"], batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=8)  # type: ignore

    optim, lr_scheduler = wmodel.configure_optimizers()
    epoch = 0
    micro_step = 0
    grad_accum_steps = BATCH_SIZE // MICRO_BATCH_SIZE
    metrics = TrainMetrics()
    micro_batch = get_batch(train_dataloader, "mps")
    while True:
        loss = wmodel.step("train", micro_batch)
        loss = loss / grad_accum_steps
        metrics.train_loss = loss.item()
        micro_batch = get_batch(train_dataloader, "mps")
        loss.backward()
        micro_step += 1

        # If we are still accumulating gradients, then skip gradient application and logging.
        if micro_step % grad_accum_steps != 0:
            continue

        optim.step()
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step()
        metrics["train_loss"] = 0
        # log other stuff
        # check if we should eval
        # reset stuff

        for micro_batch in val_dataloader:
            loss = wmodel.step("val", batch)
        if epoch == NUM_EPOCHS:
            break

    # save
    torch.save(lmodel.model.state_dict(), f"chekpoints/{run_name}_model.pt")


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
    )
