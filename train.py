import argparse
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
import lighting.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utils.data_utils import get_dataset
from .src.telepath import TelepathConfig, Telepath
from .src.lightning_wrapper import (
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
        assert isinstance(tokenizer, PreTrainedTokenizer)

    ds = get_dataset(
        path=dataset_path,
        add_transform=add_transform,
        tokenizer=tokenizer,
        max_length=max_length,
        num_channels=num_channels,
        num_samples=num_samples,
    )

    # Create model.
    lmodel = TelepathLightningWrapper(
        model_config_path=model_config_path, optimizer_config_path=optimizer_config_path
    )

    # Create logger.
    logger = WandbLogger(project="telepath")

    # Create data loaders.
    train_dataloader = DataLoader(ds["train"], batch_size=32)
    val_dataloader = DataLoader(ds["test"], batch_size=32)

    # Create trainer.
    trainer = pl.Trainer(accelerator="mps", val_check_interval=0.1, logger=logger)

    # train
    trainer.fit(lmodel, train_dataloader, val_dataloader)

    # save
    torch.save(lmodel.model.state_dict(), f"chekpoints/{run_name}_model.pt")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        dataset_path=args.dataset_path,
        add_transform=args.add_transform,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        num_channels=args.num_channels,
        num_samples=args.num_samples,
        model_config_path=args.model_config_path,
        optimizer_config_path=args.optimizer_config_path,
    )
