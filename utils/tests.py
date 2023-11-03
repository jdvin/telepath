from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.gpt import GPT
from utils.data_utils import get_transform


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


def test_gpt():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hf_gpt = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    custom_gpt = GPT.from_pretrained("gpt2").eval()

    inputs = tokenizer("The quick brown fox jumps over the lazy", return_tensors="pt")[
        "input_ids"
    ]

    # import pdb

    # pdb.set_trace()
    hf_emb = hf_gpt.transformer.wte(inputs)
    custom_emb = custom_gpt.transformer.wte(inputs)
    assert torch.equal(hf_emb, custom_emb)

    pos = torch.arange(0, inputs.size(-1), dtype=torch.long)
    hf_posemb = hf_gpt.transformer.wpe(pos)
    custom_posemb = custom_gpt.transformer.wpe(pos)
    assert torch.equal(hf_posemb, custom_posemb)

    x = hf_emb + hf_posemb

    cus_opt = custom_gpt.transformer.blocks[0].ln_1(x)
    hf_opt = hf_gpt.transformer.h[0].ln_1(x)
    assert torch.equal(cus_opt, hf_opt)

    cus_opt = custom_gpt.transformer.blocks[0].attn(cus_opt)
    hf_opt = hf_gpt.transformer.h[0].attn(hf_opt)
    assert torch.equal(cus_opt, hf_opt)

    cus_opt = custom_gpt.transformer.blocks[0].ln_2(cus_opt)
    hf_opt = hf_gpt.transformer.h[0].ln_2(hf_opt)
    assert torch.equal(cus_opt, hf_opt)

    cus_opt = custom_gpt.transformer.blocks[0].mlp(cus_opt)
    hf_opt = hf_gpt.transformer.h[0].mlp(hf_opt)
    assert torch.equal(cus_opt, hf_opt)

    custom_outputs = custom_gpt(inputs)
    hf_outputs = hf_gpt(inputs).logits
    assert torch.equal(custom_outputs, hf_outputs)
