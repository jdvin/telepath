import argparse
import enum
import random

import datasets
import transformers
import torch

from typing import Callable, Any


class ValidationType(enum.Enum):
    RANDOM = "random"
    SUBJECT = "subject"
    OBJECT = "object"


parser = argparse.ArgumentParser()
parser.add_argument("--datafiles", type=str, nargs="+", required=True)
parser.add_argument("--validation_type", type=ValidationType, required=True)
parser.add_argument("--validation_objects", type=str, nargs="+")
parser.add_argument("--pre_transform", action="store_true")
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--max_length", type=int)
parser.add_argument("--num_channels", type=int)
parser.add_argument("--num_samples", type=int)
parser.add_argument("--output_path", type=str)


def get_transform(
    tokenizer: transformers.PreTrainedTokenizerFast,
    max_length: int,
    num_channels: int,
    num_samples: int,
) -> Callable[[dict[str, Any]], dict[str, torch.Tensor]]:
    def transform(batch) -> dict[str, torch.Tensor]:
        batch["eeg"] = torch.tensor(batch["eeg"]).reshape(-1, num_channels, num_samples)
        batch["input_ids"] = tokenizer(
            batch["object"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]

        return batch

    return transform


def create_dataset(
    datafiles: list[str],
    validation_type: ValidationType,
    validation_objects: list[str] | None = None,
    pre_transform: bool = False,
    tokenizer: transformers.PreTrainedTokenizerFast | None = None,
    max_length: int | None = None,
    num_channels: int | None = None,
    num_samples: int | None = None,
) -> datasets.DatasetDict:
    if validation_type == ValidationType.SUBJECT:
        assert (
            len(datafiles) > 1
        ), "Must provide more than one datafile for subject-held-out validation."
        held_out_subject = random.choice(datafiles)
        datafile_map = {
            "train": [df for df in datafiles if df != held_out_subject],
            "test": [held_out_subject],
        }
    else:
        datafile_map = {"train": datafiles}

    dataset = datasets.load_dataset("json", data_files=datafile_map)
    assert isinstance(dataset, datasets.DatasetDict)
    if validation_type == ValidationType.RANDOM:
        dataset_splits: datasets.DatasetDict = dataset["train"].train_test_split(
            test_size=0.1, seed=42
        )
    elif validation_type == ValidationType.OBJECT:
        train_dataset = dataset["train"].filter(
            lambda row: row["object"] not in validation_objects
        )
        test_dataset = dataset["train"].filter(
            lambda row: row["object"] in validation_objects
        )
        dataset_splits: datasets.DatasetDict = datasets.DatasetDict(
            {"train": train_dataset, "test": test_dataset}
        )
    # Load bearing hack to shut up pyright.
    assert isinstance(dataset_splits, datasets.DatasetDict)  # type: ignore
    if pre_transform:
        assert tokenizer is not None
        assert max_length is not None
        assert num_channels is not None
        assert num_samples is not None
        dataset_splits = dataset_splits.map(
            get_transform(tokenizer, max_length, num_channels, num_samples),
            batched=True,
            batch_size=None,
            remove_columns=["object"],
        )
    return dataset_splits


def get_dataset(
    path: str,
    add_transform: bool = True,
    tokenizer: transformers.PreTrainedTokenizerFast | None = None,
    max_length: int | None = None,
    num_channels: int | None = None,
    num_samples: int | None = None,
) -> datasets.DatasetDict:
    dataset = datasets.load_from_disk(path)
    assert isinstance(dataset, datasets.DatasetDict)

    if add_transform:
        assert tokenizer is not None
        assert max_length is not None
        assert num_channels is not None
        assert num_samples is not None
        transform_fn = get_transform(tokenizer, max_length, num_channels, num_samples)
        dataset.set_transform(transform_fn)

    return dataset


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = create_dataset(
        datafiles=args.datafiles,
        validation_type=args.validation_type,
        validation_objects=args.validation_objects,
        pre_transform=args.pre_transform,
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        num_channels=args.num_channels,
        num_samples=args.num_samples,
    )
    dataset.save_to_disk(args.output_path)
