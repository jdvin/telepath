import argparse
from enum import Enum
import os
from typing import Callable, Any
import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class ValidationType(Enum):
    DEFAULT = "default"
    RANDOM = "random"
    SUBJECT = "subject"
    OBJECT = "object"


ELECTRODE_ORDER = np.array(
    [
        "Fp1",
        "F3",
        "F7",
        "FT9",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "TP9",
        "CP5",
        "CP1",
        "Pz",
        "P3",
        "P7",
        "O1",
        "Oz",
        "O2",
        "P4",
        "P8",
        "TP10",
        "CP6",
        "CP2",
        "Cz",
        "C4",
        "T8",
        "FT10",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "F1",
        "F5",
        "FT7",
        "FC3",
        "FCz",
        "C1",
        "C5",
        "TP7",
        "CP3",
        "P1",
        "P5",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "P6",
        "P2",
        "CPz",
        "CP4",
        "TP8",
        "C6",
        "C2",
        "FC4",
        "FT8",
        "F6",
        "F2",
        "AF4",
        "AF8",
    ]
)

SESSIONS_PER_SUBJECT = 4


SESSION_EPOCHS = {"train": 16800, "test": 4080}


def get_spectrogram(signal: torch.Tensor, n_fft: int, hop_length: int):
    window = torch.hann_window(n_fft).to(signal.device)
    stft = torch.stft(signal, n_fft, hop_length, window=window, return_complex=True)
    return stft.abs() ** 2


class ThingsDataset(Dataset):
    def __init__(self, ds: np.memmap, things_metadata_path: str):
        self.ds = ds
        self.things_metadata = pd.read_csv(things_metadata_path)

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, idx):
        return {
            # Slice out the stimulus channels.
            "eeg": self.ds[idx, 1:, :],
            # Get the object id from the stimulus channel.
            "object": self.things_metadata["Word"][self.ds[idx, 0, 0].to(int)],
        }


def get_dataset_dict(
    root_dir: str, ds_key: str, things_metadata_path
) -> dict[str, ThingsDataset]:
    train_data = np.load(f"{root_dir}/{ds_key}_train.npy", allow_pickle=True)
    test_data = np.load(f"{root_dir}/{ds_key}_test.npy", allow_pickle=True)
    return {
        "train": ThingsDataset(train_data, things_metadata_path),
        "test": ThingsDataset(test_data, things_metadata_path),
    }


def extract_things_100ms_ds(
    root_dir: str,
    subjects: list[int] | range,
    validation_type: ValidationType = ValidationType.DEFAULT,
    epoch_start: int = -200,
    epoch_end: int = 200,
    reset_cache: bool = False,
) -> dict[str, np.memmap]:
    """This is going to be a doozy.

    This may look unecessarily verbose, but the goal here is to be able to go straight from the
    the raw files as they were given to a dataset of desired structure.
    That way, if the structure changes, it can just be reflected here, instead of having to do preprocessing each time.

    Side Note: I have no idea why the dude set the dataset up like this, would it not have made so much more sense to just
    use the common indexing scheme of THINGS between the training and test sets to begin with?
    """
    ds_str = "".join([str(sub) for sub in subjects]) + str(epoch_start) + str(epoch_end)
    if validation_type != ValidationType.DEFAULT:
        raise NotImplementedError("Only default validation type is supported atm.")
    # Keys are coded: `{split_type}_img_concepts_THINGS`.
    # Values are coded arrays of strings each coded: `{index}_{object_id}`.
    # The index is offset by +1 relative to THINGS probably because `0` is used as padding in the stim channel.
    eeg_img_metadata = {
        key.split("_")[0]: [float(obj.split("_")[0]) - 1 for obj in value]
        for key, value in np.load(f"{root_dir}/image_metadata.npy", allow_pickle=True)
        .all()
        .items()
        if "THINGS" in key
    }

    training_file_path = f"{root_dir}/{ds_str}_train.npy"
    test_file_path = f"{root_dir}/{ds_str}_test.npy"
    cached = os.path.exists(training_file_path) and os.path.exists(test_file_path)
    split_shape = lambda epochs_per_session: (
        len(subjects) * SESSIONS_PER_SUBJECT * epochs_per_session,
        len(ELECTRODE_ORDER),
        epoch_end - epoch_start,
    )

    ds = {
        "train": np.memmap(
            filename=training_file_path,
            mode="r" if cached else "w+",
            shape=split_shape(SESSION_EPOCHS["train"]),
        ),
        "test": np.memmap(
            filename=test_file_path,
            mode="r" if cached else "w+",
            shape=split_shape(SESSION_EPOCHS["test"]),
        ),
    }
    if cached and not reset_cache:
        return ds

    for i, sub in enumerate(subjects):
        for j, ses in enumerate(range(1, SESSIONS_PER_SUBJECT + 1)):
            for split_type in ["train", "test"]:
                path = os.path.join(
                    root_dir,
                    f"sub-{'0' if sub < 9 else ''}{sub}",
                    f"ses-0{ses}",
                    f"raw_eeg_{split_type}.npy",
                )
                data = np.load(path, allow_pickle=True).all()
                stim_index = data["ch_types"].index("stim")
                ch_names = np.array(
                    [name for name in data["ch_names"] if name != "stim"]
                )
                # Ensure the order of the electrode order is consistent.
                # This may be overkill but it is very important, so worth being sure about.
                _, ordered_electrode_indexes = np.where(
                    ELECTRODE_ORDER[:, None] == ch_names
                )
                # Get the true THINGS id...
                stims = np.array(
                    [
                        (
                            eeg_img_metadata[split_type][int(i) - 1]
                            if i not in {0.0, 99999.0}
                            else i
                        )
                        for i in data["raw_eeg_data"][stim_index, :]
                    ],
                    dtype=data["raw_eeg_data"].dtype,
                )
                # Get ordered electrode data.
                data = data["raw_eeg_data"][ordered_electrode_indexes, :]
                # Stack with stimulus data.
                data = np.vstack((stims, data))
                # Get the index of each stimulus onset.
                epoch_indexes = data[0, :].nonzero()[0]
                for k in epoch_indexes:
                    # Get the absolute index of the current epoch.
                    n = (
                        i * SESSIONS_PER_SUBJECT * SESSION_EPOCHS[split_type]
                        + j * SESSION_EPOCHS[split_type]
                        + k
                    )
                    # Slice the current epoch out of the data stream.
                    # rows x ch x time <- ch x time.
                    ds[split_type][n, :, :] = data[:, k + epoch_start : k + epoch_end]
                    # Label the stimulus channel at the start of the epoch.
                    ds[split_type][n, 0, 0] = np.max(data[split_type][n, 0, :])

    assert all([isinstance(value, np.memmap) for value in ds.values()])
    return ds


def get_collate_fn(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    start_token_id_sequence: Tensor,
    stop_token_id: int,
    pad_token_id: int,
    n_fft: int,
    fft_hop_length: int,
    things_concepts_path: str = "data/things_concepts.csv",
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    # Load the map from object ID to word.
    things_concepts = pd.read_csv(things_concepts_path)
    object_id_to_word = dict(zip(things_concepts["uniqueID"], things_concepts["Word"]))
    tokenizer.pad_token_id = pad_token_id
    stop_token = tokenizer.decode([stop_token_id])

    # Define transformation function with parameters.
    def collate_fn(
        samples: list[dict[str, torch.Tensor | str]]
    ) -> dict[str, torch.Tensor]:
        batch_size = len(samples)
        batch = {}
        eegs: list[Tensor] = [
            sample["eeg"] for sample in samples if isinstance(sample["eeg"], Tensor)
        ]  # fuck you pyright.
        batch["eeg"] = get_spectrogram(torch.stack(eegs), n_fft, fft_hop_length)
        # We are doing all of the special tokens manually because (1) We do not trust HF, and (2) we have more control.
        # Here we add the stop token manually because it will then be included in the _attended to region_ of the attenton mask.
        # Which is not the default behaviour if we have `add_special_tokens=False`, because then all added tokens are treated like padding (i.e., not attended to).
        objects = [
            " " + object_word.lower().strip() + stop_token
            for object_word in batch["object"]
        ]
        tokenizer_out = tokenizer.batch_encode_plus(
            objects,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        batch["attention_mask"] = torch.cat(
            (tokenizer_out["input_ids"], torch.zeros(batch_size, 1))
        )
        batch["input_ids"] = torch.cat(
            (  # type: ignore
                torch.tile(start_token_id_sequence, (batch_size, 1)),
                tokenizer_out["input_ids"],
                torch.full((batch_size, 1), stop_token_id),
            ),
            dim=1,
        )

        return batch

    return collate_fn
