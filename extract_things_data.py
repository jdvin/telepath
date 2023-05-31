import argparse
import logging

import mne
import polars as pl
import tqdm

mne.set_log_level(logging.CRITICAL)

parser = argparse.ArgumentParser()

parser.add_argument("participant", type=str, default="sub-02")
parser.add_argument("--root_path", type=str, default="/Volumes/T7/datasets/things-eeg/")


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
    epochs = mne.Epochs(
        raw,
        events_from_annotations,
        events_dict,
        tmin=0.0,
        tmax=(0.05).jso,
        baseline=None,
    )
    object_events_df = pl.read_csv(
        root_path + data_path_template.format(participant=participant) + "events.tsv",
        separator="\t",
    )
    object_on_epoch = []
    object_off_epoch = []
    i = -1
    for object_onset in tqdm.tqdm(object_events_df["onset"]):
        while True:
            i += 1
            # If the object onset is within 1 sample of the start of the epoch.
            if abs(object_onset - epochs[i].events[0, 0]) <= 1:
                # Add this and subsequent epoch to the column arrays.
                object_on_epoch.append(epochs[i].to_data_frame().to_json())
                object_off_epoch.append(epochs[i + 1].to_data_frame().to_json())
                import pdb

                pdb.set_trace()
                break

        # Add stim_on and stim_off epochst to `object_events.`
    # Save `object_events.`
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
