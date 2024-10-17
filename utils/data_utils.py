from enum import Enum
import os
from typing import Callable, Any

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm


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


def extract_things_100ms_ds(
    root_dir: str,
    subjects: list[int] | range,
    validation_type: ValidationType = ValidationType.DEFAULT,
    epoch_start: int = -200,
    epoch_end: int = 200,
    is_test_run: bool = False,
    reset_cache: bool = False,
) -> dict[str, np.memmap]:
    """
    Extract the EEG data from the raw files and save it to a memmap file.

    Args:
        ...
        is_test_run: If True, construct a testing dataset consisting of only `SESSION_PER_SUBJECT` epochs, for the given subjects. Have the `test` set be the same as the training set.
                    Useful for debugging.
        ...

    This is going to be a doozy.

    This may look unecessarily verbose, but the goal here is to be able to go straight from the
    the raw files as they were given to a dataset of desired structure.
    That way, if the structure changes, it can just be reflected here, instead of having to do preprocessing each time.

    Side Note: I have no idea why the dude set the dataset up like this, would it not have made so much more sense to just
    use the common indexing scheme of THINGS between the training and test sets to begin with?
    """
    ds_str = "".join([str(sub) for sub in subjects]) + str(epoch_start) + str(epoch_end)
    if is_test_run:
        sessions_per_subject = 1
        session_epochs = {"train": 120, "test": 120}
        ds_str += "_test"
    else:
        sessions_per_subject = 4
        session_epochs = {"train": 16710, "test": 4080}

    epoch_length = epoch_end - epoch_start
    target_obj_onset_idx = -1 * epoch_start
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
    cached = (
        os.path.exists(training_file_path) and os.path.exists(test_file_path)
    ) and not reset_cache
    logger.info(f"Using cached dataset: {cached}.")

    def split_shape(epochs_per_session: int) -> tuple[int, int, int]:
        return (
            len(subjects) * sessions_per_subject * epochs_per_session,
            len(ELECTRODE_ORDER) + 1,  # +1 for the stimulus channel.
            epoch_end - epoch_start,
        )

    train_split_shape = split_shape(session_epochs["train"])
    test_split_shape = split_shape(session_epochs["test"])
    ds = {
        "train": np.memmap(
            dtype=np.float32,
            filename=training_file_path,
            mode="r" if cached else "w+",
            shape=train_split_shape,
        ),
        "test": np.memmap(
            dtype=np.float32,
            filename=test_file_path,
            mode="r" if cached else "w+",
            shape=test_split_shape,
        ),
    }
    if cached and not reset_cache:
        return ds
    total_rows = sum(
        [
            session_epochs[split_type] * sessions_per_subject * len(subjects)
            for split_type in ds.keys()
        ]
    )
    split_types = ["train", "test"] if not is_test_run else ["train"]
    pbar = tqdm(total=total_rows, desc="Extracting EEG Data.")
    n = 0
    for sub_i, sub in enumerate(subjects):
        for ses_i, ses in enumerate(range(1, sessions_per_subject + 1)):
            for split_type in split_types:
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
                # TODO: Why do we do this first, why not as we are iterating through the epoch indexes?
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
                epoch_i = 0
                for epoch_loc in epoch_indexes:
                    target_obj = data[0, epoch_loc]
                    if target_obj == 99999.0:
                        continue
                    # Get the absolute index of the current epoch.
                    n = (
                        sub_i * sessions_per_subject * session_epochs[split_type]
                        + ses_i * session_epochs[split_type]
                        + epoch_i
                    )
                    # Slice the current epoch out of the data stream.
                    # rows x ch x time <- ch x time.
                    ds[split_type][n, :, :] = data[
                        :, epoch_loc + epoch_start : epoch_loc + epoch_end
                    ]
                    # Label the stimulus channel at the start of the epoch.
                    ds[split_type][n, 0, :] = np.full(
                        shape=epoch_length,
                        fill_value=target_obj,
                    )
                    pbar.update(1)
                    epoch_i += 1
                    if is_test_run and epoch_i >= session_epochs[split_type]:
                        break
    if is_test_run:
        ds["test"] = ds["train"].copy()
    assert all([isinstance(value, np.memmap) for value in ds.values()])
    return ds


def get_spectrogram(signal: torch.Tensor, n_fft: int, hop_length: int):
    window = torch.hann_window(n_fft)
    signal = (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)
    stft = torch.stft(signal, n_fft, hop_length, window=window, return_complex=True)
    # Freq 0 is not needed because the signal is normalized.
    return stft[:, 1:, :-1].abs() ** 2


def get_collate_fn(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    start_token_id_sequence: Tensor,
    stop_token_id: int,
    pad_token_id: int,
    get_spectrogram: bool,
    n_fft: int | None = None,
    fft_hop_length: int | None = None,
    things_concepts_path: str = "data/things_concepts.csv",
) -> Callable[[list[np.memmap]], dict[str, torch.Tensor]]:
    # Load the map from object ID to word.
    things_concepts = pd.read_csv(things_concepts_path)
    tokenizer.pad_token_id = pad_token_id
    stop_token = tokenizer.decode([stop_token_id])
    if get_spectrogram:
        assert n_fft is not None and fft_hop_length is not None

    # Define transformation function with parameters.
    def collate_fn(
        samples: list[np.memmap],
    ) -> dict[str, torch.Tensor]:
        batch_size = len(samples)
        object_words = []
        eeg_features = []
        for sample in samples:
            object_words.append(things_concepts["Word"][sample[0][0]])
            # If `get_spectrogram, sample is of shape (N_C, NF, T).
            # else, sampel is of shape (N_C, T)
            eeg_features.append(
                get_spectrogram(torch.tensor(sample[1:, :]), n_fft, fft_hop_length)
                if get_spectrogram
                else torch.tensor(sample[1:, :])
            )
        # We are doing all of the special tokens manually because (1) We do not trust HF, and (2) we have more control.
        # Here we add the stop token manually because it will then be included in the _attended to_ region of the attenton mask.
        # Which is not the default behaviour if we have `add_special_tokens=False`, because then all added tokens are treated like padding (i.e., not attended to).
        objects = [
            " " + object_word.lower().strip() + stop_token
            for object_word in object_words
        ]
        tokenizer_out = tokenizer.batch_encode_plus(
            objects,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        attention_mask = tokenizer_out["attention_mask"]
        input_ids = tokenizer_out["input_ids"]
        start_sequences = torch.tile(start_token_id_sequence, (batch_size, 1))
        assert isinstance(attention_mask, torch.Tensor)
        decoder_attention_mask = torch.cat(
            [torch.ones_like(start_sequences), attention_mask], dim=1
        )
        input_ids = torch.cat(
            (
                start_sequences,
                tokenizer_out["input_ids"],
            ),  # type: ignore
            dim=1,
        )
        assert isinstance(eeg_features, list)

        return {
            "input_features": torch.stack(eeg_features),
            "input_ids": input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

    return collate_fn
