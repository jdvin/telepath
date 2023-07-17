import argparse
import os

import polars as pl

parser = argparse.ArgumentParser()

parser.add_argument("--root_path", type=str, default="/Volumes/T7/datasets/things-eeg/")


SUBJECT_DATA = "combined/sub-16.json"

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

    cols = df.columns

    object_on_epoch = df.select(["eventnumber", "object_on_epoch"]).unnest(
        "object_on_epoch"
    )
    object_off_epoch = (
        df.select(["eventnumber", "object_off_epoch"])
        .unnest("object_off_epoch")
        .unnest(ELECTRODES)
    )

    obj = df.select("object")

    df = pl.concat()
