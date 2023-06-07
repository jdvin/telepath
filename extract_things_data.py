import argparse
import logging
import os
from typing import Dict, List

import mne
import polars as pl
import tqdm

mne.set_log_level(logging.CRITICAL)

parser = argparse.ArgumentParser()

parser.add_argument("participant", type=str, default="sub-02")
parser.add_argument("--root_path", type=str, default="/Volumes/T7/datasets/things-eeg/")


def flatten_epoch_dict(
    epoch_dict: Dict[str, Dict[int, float]]
) -> Dict[str, List[float]]:
    """Dicts with non-string keys cannot be placed into rust structs."""
    return {key: list(value.values()) for key, value in epoch_dict.items()}


def extract_from_eeg_file(
    participant: str,
    root_path: str,
    data_path_template: str = "{participant}/eeg/{participant}_task-rsvp_",
) -> pl.DataFrame:
    raw = mne.io.read_raw_brainvision(
        root_path + data_path_template.format(participant=participant) + "eeg.vhdr",
        preload=True,
    )
    events_from_annotations, events_dict = mne.events_from_annotations(raw)

    # T=0 w.r.t. each epoch is one sample before the object is displayed.
    epochs = mne.Epochs(
        raw,
        events_from_annotations,
        events_dict,
        tmin=0.001,
        tmax=0.05,
        baseline=None,
        preload=True,
    )
    object_events_df = pl.read_csv(
        root_path + data_path_template.format(participant=participant) + "events.tsv",
        separator="\t",
    )
    object_on_epoch = []
    object_off_epoch = []
    i = 0
    for object_onset in tqdm.tqdm(object_events_df["onset"]):
        while True:
            # If the object onset is within 1 sample of the start of the epoch.
            if abs(object_onset - epochs[i].events[0, 0]) <= 1:
                # Add this and subsequent epoch to the column arrays.
                # TODO: This is a lot of transformations...
                object_on_epoch.append(
                    flatten_epoch_dict(epochs[i].to_data_frame().to_dict())
                )
                object_off_epoch.append(
                    flatten_epoch_dict(epochs[i + 1].to_data_frame().to_dict())
                )
                # We know the next epoch is not for the next object, so we skip it.
                i += 2
                break
            i += 1

    object_events_df = object_events_df.with_columns(
        pl.Series(name="object_on_epoch", values=object_on_epoch, dtype=pl.Struct),
        pl.Series(name="object_off_epoch", values=object_off_epoch, dtype=pl.Struct),
    )

    object_events_df.write_json(
        file=os.path.join(root_path, "combined", f"{participant}.json"), pretty=True
    )
    return object_events_df


def load_participant(
    participant: str,
    participant_data: dict,
    root_path: str,
):
    eeg_object_events: pl.DataFrame = extract_from_eeg_file(
        participant=participant,
        root_path=root_path,
    )
    # Add participant data to object_events.
    for key, value in participant_data.items():
        eeg_object_events = eeg_object_events.with_columns(pl.lit(value).alias(key))


if __name__ == "__main__":
    args = parser.parse_args()
    participant_data: pl.DataFrame = pl.read_csv(
        args.root_path + "participants.tsv", separator="\t"
    )
    load_participant(
        participant=args.participant,
        participant_data=participant_data.row(
            by_predicate=(pl.col("participant_id") == args.participant), named=True
        ),
        root_path=args.root_path,
    )
