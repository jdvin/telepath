from transformers import AutoTokenizer
import torch

from .data_utils import get_transform


def test_transform():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    transform = get_transform(
        tokenizer=tokenizer,
        max_length=5,
        num_channels=2,
        num_samples=3,
        start_token_id=50257,
        stop_token_id=50256,
        pad_token_id=50256,
    )

    batch = {
        "object": ["dog", "cat", "fish"],
        "eeg": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
    }
    # import pdb

    # pdb.set_trace()
    transformed_batch: dict[str, torch.Tensor] = transform(batch)
    # The transformation should take a batch of cancatenated channel signals over time and batch of nested sample from all channels per time step.
    assert torch.equal(
        transformed_batch["eeg"],
        torch.tensor(
            [
                [[1, 4], [2, 5], [3, 6]],
                [[1, 4], [2, 5], [3, 6]],
                [[1, 4], [2, 5], [3, 6]],
            ]
        ),
    )
    assert isinstance(transformed_batch["input_ids"], torch.Tensor)
    assert transformed_batch["input_ids"].shape == (3, 5)
    assert transformed_batch["input_ids"][:, 0].eq(50257).all().item()
    assert transformed_batch["input_ids"][:, -1].eq(50256).all().item()
