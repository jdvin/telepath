from transformers import AutoTokenizer
import torch

from .data_utils import get_transform


def test_transform():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    transform = get_transform(
        tokenizer=tokenizer, max_length=5, num_channels=2, num_samples=3
    )

    batch = {
        "object": ["dog", "cat", "fish"],
        "eeg": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
    }
    transformed_batch: dict[str, torch.Tensor] = transform(batch)

    assert torch.equal(
        transformed_batch["eeg"],
        torch.tensor(
            [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]
        ),
    )

    assert isinstance(transformed_batch["input_ids"], torch.Tensor)
    assert transformed_batch["input_ids"].shape == (3, 5)
