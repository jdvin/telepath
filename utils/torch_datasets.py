from typing import Any

from torch.utils.data import Dataset

class ThingsEEGDataset(Dataset):
    def __init__(self, filepaths: list[str]):
        self.filepaths = filepaths
        self.length_map: dict[str, int] = {}
        for filepath in filepaths:
            with open(filepath, "r") as f:
                self.length_map[filepath] = sum(1 for _ in f)
        

    def _get_line(self, file_index: int, row_index: int) -> dict[str, Any]:
        """"""

        with open(self.filepaths[file_index], "r") as f:
            return {"": 1}          
