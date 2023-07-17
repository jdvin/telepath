import argparse
import gc
import logging
import multiprocessing
import os
from typing import Any, Dict, List

import mne
import polars as pl
import tqdm

mne.set_log_level(logging.CRITICAL)

parser = argparse.ArgumentParser()

parser.add_argument("--root_path", type=str, default="/Volumes/T7/datasets/things-eeg/")
parser.add_argument("--n", type=int, default=None)

args = parser.parse_args()

ROOT_PATH = args.root_path


def flatten_epoch_dict(
    epoch_dict: Dict[str, Dict[int, float]]
) -> Dict[str, List[float]]:
    """Dicts with non-string keys cannot be placed into rust structs."""
    return {key: list(value.values()) for key, value in epoch_dict.items()}


def extract_from_eeg_file(
    participant: str,
    data_path_template: str = "{participant}/eeg/{participant}_task-rsvp_",
) -> bool:
    # Read the raw data through MNE.
    raw = mne.io.read_raw_brainvision(
        ROOT_PATH + data_path_template.format(participant=participant) + "eeg.vhdr",
        preload=False,
    )
    # Extract
    events_from_annotations, events_dict = mne.events_from_annotations(raw)

    # T=0 w.r.t. each epoch is one sample before the object is displayed.
    epochs = mne.Epochs(
        raw,
        events_from_annotations,
        events_dict,
        tmin=0.001,
        tmax=0.05,
        baseline=None,
        preload=False,
    )

    object_events_df = pl.read_csv(
        ROOT_PATH + data_path_template.format(participant=participant) + "events.tsv",
        separator="\t",
        infer_schema_length=10000,
        dtypes={"rt": "f64"},
    )
    object_on_epoch = []
    object_off_epoch = []
    i = 0
    for object_onset in tqdm.tqdm(object_events_df["onset"], desc=participant):
        while True:
            # If the object onset is within 1 sample of the start of the epoch.
            if abs(object_onset - epochs[i].events[0, 0]) <= 1:
                # Add this and subsequent epoch to the column arrays.
                # TODO: This is a lot of transformations and duplicate memory...
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

    out_path = os.path.join(ROOT_PATH, "combined", f"{participant}.json")

    object_events_df.write_json(file=out_path, pretty=True)

    return True


def load_participant(args: tuple[str, dict]):
    participant, participant_data = args
    eeg_object_events: pl.DataFrame = extract_from_eeg_file(
        participant=participant,
    )
    # Add participant data to object_events.
    for key, value in participant_data.items():
        eeg_object_events = eeg_object_events.with_columns(pl.lit(value).alias(key))


if __name__ == "__main__":
    participants = [f"sub-{'0' if i < 10 else ''}{i}" for i in range(1, 51)]

    if not os.path.exists(os.path.join(ROOT_PATH, "combined")):
        os.mkdir(os.path.join(ROOT_PATH, "combined"))

    participant_data: List[Dict[str, Any]] = pl.read_csv(
        ROOT_PATH + "participants.tsv", separator="\t"
    ).to_dicts()
    args = zip(participants, participant_data)
    with multiprocessing.Pool(3) as p:
        success = p.map(load_participant, args)

    print(
        f"Successfully extracted data for {sum(success)}/{len(participants)} participants."
    )
