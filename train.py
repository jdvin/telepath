import argparse
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from utils.data_utils import get_dataset
from src.lightning_wrapper import (
    TelepathLightningWrapper,
    TrainingConfig,
    OptimizerConfig,
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
    lmodel = TelepathLightningWrapper(
        model_config_path=model_config_path, optimizer_config_path=optimizer_config_path
    )
    ds = get_dataset(
        path=dataset_path,
        add_transform=add_transform,
        tokenizer=tokenizer,
        start_token_id=lmodel.config.gpt_start_token,
        stop_token_id=lmodel.config.gpt_stop_token,
        pad_token_id=lmodel.config.gpt_stop_token,
        max_length=max_length,
        num_channels=num_channels,
        num_samples=num_samples,
    )

    # Create logger.
    logger = WandbLogger(project="telepath")

    # Create data loaders.
    train_dataloader = DataLoader(ds["train"], batch_size=4)  # type: ignore
    val_dataloader = DataLoader(ds["test"], batch_size=4)  # type: ignore

    # Create trainer.
    trainer = pl.Trainer(
        accelerator="mps",
        max_epochs=5,
        val_check_interval=0.1,
        logger=logger,
        log_every_n_steps=8,
        accumulate_grad_batches=8,
    )

    # train
    trainer.fit(
        lmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

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
