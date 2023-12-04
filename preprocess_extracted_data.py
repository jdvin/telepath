import argparse
import os
import json

import polars as pl

parser = argparse.ArgumentParser()

parser.add_argument(
    "--root_path", type=str, default="/Volumes/T7/datasets/things-eeg/combined/"
)


SUBJECT_DATA = "sub-16.json"

ELECTRODES = [
    "Fp1",
    "Fz",
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
    "C1",
    "Cz",
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
    "AF8",
    "AF4",
    "F2",
    "FCz",
]


if __name__ == "__main__":
    args = parser.parse_args()
    df = pl.read_json(os.path.join(args.root_path, SUBJECT_DATA))

    df = df.filter((pl.col("blocksequencenumber") >= 0) & (pl.col("istarget") == 0))

    object_on_epoch = df.select(["eventnumber", "object_on_epoch"]).unnest(
        "object_on_epoch"
    )
    object_off_epoch = df.select(["eventnumber", "object_off_epoch"]).unnest(
        "object_off_epoch"
    )

    combined = df.select(["eventnumber", "object"])

    for electrode in ELECTRODES:
        electrode_combined_epoch = df.select(
            pl.concat_list(object_on_epoch[electrode], object_off_epoch[electrode])
        )
        combined = combined.with_columns(electrode_combined_epoch)

    # Result of a column with data: [Fp1_0, Fp1_1, Fp1_2, ..., Fp1_99, Fz_0, Fz_1, ...]
    combined = combined.with_columns(
        pl.select(pl.concat_list(combined.select(ELECTRODES)))
    ).rename({"Fp1": "eeg"})

    preprocessed_path = os.path.join(
        args.root_path, SUBJECT_DATA.split(".")[0] + "_preprocessed" + ".jsonl"
    )

    # JSONL export.
    with open(preprocessed_path, "w") as f:
        for row in combined.to_dicts():
            row.update({"electrode_order": ELECTRODES, "samples_per_epoch": 100})
            f.write(json.dumps(row) + "\n")
