import yaml

import torch


def load_yaml(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_batch(dataloader, device: str) -> dict[str, torch.Tensor]:
    batch = next(iter(dataloader))

    for key, value in batch:
        if device == "cpu":
            batch[key] = value.to(device)
        else:
            batch[key] = value.pin_memory().to(device, non_blocking=True)
    return batch
