import argparse
import enum
import random
from typing import Callable, Any

import datasets
import pandas as pd
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class ValidationType(enum.Enum):
    RANDOM = "random"
    SUBJECT = "subject"
    OBJECT = "object"


parser = argparse.ArgumentParser()
parser.add_argument("--datafiles", type=str, nargs="+", required=True)
parser.add_argument("--validation_type", type=ValidationType, required=True)
parser.add_argument("--pre_transform", action="store_true")
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--max_length", type=int)
parser.add_argument("--num_channels", type=int)
parser.add_argument("--num_samples", type=int)
parser.add_argument("--output_path", type=str)


def get_transform(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    start_token_id: int,
    stop_token_id: int,
    pad_token_id: int,
    max_length: int,
    num_channels: int,
    num_samples: int,
    things_concepts_path: str = "data/things_concepts.csv",
) -> Callable[[dict[str, Any]], dict[str, torch.Tensor]]:
    # Load the map from object ID to word.
    things_concepts = pd.read_csv(things_concepts_path)
    object_id_to_word = dict(zip(things_concepts["uniqueID"], things_concepts["Word"]))
    tokenizer.pad_token_id = pad_token_id

    # Define transformation function with parameters.
    def transform(batch) -> dict[str, torch.Tensor]:
        batch_size = len(batch["object"])
        transformed_batch = {}
        transformed_batch["eeg"] = (
            torch.tensor(batch["eeg"])
            .view(batch_size, num_channels, num_samples)
            .transpose(1, 2)
            .to(torch.float32)
        )
        objects = [" " + object_id_to_word[object_id] for object_id in batch["object"]]
        transformed_batch["input_ids"] = tokenizer.batch_encode_plus(
            objects,
            padding="max_length",
            truncation=True,
            max_length=max_length - 2,
            return_tensors="pt",
        )["input_ids"]
        transformed_batch["input_ids"] = torch.cat(
            (  # type: ignore
                torch.full((batch_size, 1), start_token_id),
                transformed_batch["input_ids"],
                torch.full((batch_size, 1), stop_token_id),
            ),
            dim=1,
        )

        return transformed_batch

    return transform


def create_dataset(
    datafiles: list[str],
    validation_type: ValidationType,
    pre_transform: bool = False,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    start_token_id: int | None = None,
    stop_token_id: int | None = None,
    pad_token_id: int | None = None,
    max_length: int | None = None,
    num_channels: int | None = None,
    num_samples: int | None = None,
    things_concepts_path: str = "data/things_concepts.csv",
) -> datasets.DatasetDict:
    # If we are using subject-held-out validation, randomly choose one subject data file to hold.
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
        # Otherwise, all data is put in the training set for now.
        datafile_map = {"train": datafiles}

    dataset = datasets.load_dataset("json", data_files=datafile_map)
    assert isinstance(dataset, datasets.DatasetDict)
    if validation_type == ValidationType.RANDOM:
        dataset_splits: datasets.DatasetDict = dataset["train"].train_test_split(
            test_size=0.1, seed=42
        )
    elif validation_type == ValidationType.OBJECT:
        things_concepts = pd.read_csv(things_concepts_path)
        validation_objects = list(
            things_concepts["uniqueID"].sample(frac=0.1, random_state=42)
        )
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
        assert start_token_id is not None
        assert stop_token_id is not None
        assert pad_token_id is not None
        assert max_length is not None
        assert num_channels is not None
        assert num_samples is not None
        dataset_splits = dataset_splits.map(
            get_transform(
                tokenizer,
                start_token_id=start_token_id,
                stop_token_id=stop_token_id,
                pad_token_id=pad_token_id,
                max_length=max_length,
                num_channels=num_channels,
                num_samples=num_samples,
            ),
            batched=True,
            batch_size=None,
            remove_columns=["object"],
        )
    print(dataset_splits)
    return dataset_splits


def get_dataset(
    path: str,
    add_transform: bool = True,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    start_token_id: int | None = None,
    stop_token_id: int | None = None,
    pad_token_id: int | None = None,
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
        assert start_token_id is not None
        assert stop_token_id is not None
        assert pad_token_id is not None
        transform_fn = get_transform(
            tokenizer=tokenizer,
            max_length=max_length,
            start_token_id=start_token_id,
            stop_token_id=stop_token_id,
            pad_token_id=pad_token_id,
            num_channels=num_channels,
            num_samples=num_samples,
        )
        dataset.set_transform(transform_fn)

    return dataset


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = create_dataset(
        datafiles=args.datafiles,
        validation_type=args.validation_type,
        pre_transform=args.pre_transform,
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        num_channels=args.num_channels,
        num_samples=args.num_samples,
    )
    dataset.save_to_disk(args.output_path)
