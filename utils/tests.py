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


activatons = None

tokenizer = AutoTokenizer.from_pretrained("gpt2")
hf_gpt = AutoModelForCausalLM.from_pretrained("gpt2").eval()
custom_gpt = GPT.from_pretrained("gpt2").eval()


def test_embeddings():
    global activations
    inputs = tokenizer("The quick brown", return_tensors="pt")["input_ids"]

    hf_emb = hf_gpt.transformer.wte(inputs)
    custom_emb = custom_gpt.transformer.wte(inputs)
    assert torch.equal(hf_emb, custom_emb)
    pos = torch.arange(0, inputs.size(-1), dtype=torch.long)
    hf_posemb = hf_gpt.transformer.wpe(pos)
    custom_posemb = custom_gpt.transformer.wpe(pos)
    assert torch.equal(hf_posemb, custom_posemb)

    activations = hf_emb + hf_posemb


def test_layer_norm():
    global activations
    cus_opt = custom_gpt.transformer.blocks[0].ln_1(activations)
    hf_opt = hf_gpt.transformer.h[0].ln_1(activations)
    assert torch.allclose(cus_opt, hf_opt, atol=1e-3)
    activations = hf_opt


def test_attention():
    global activations
    cus_attn = custom_gpt.transformer.blocks[0].attn
    hf_attn = hf_gpt.transformer.h[0].attn
    cus_opt = cus_attn(activations)
    hf_opt = hf_attn(activations)[0]

    assert torch.allclose(cus_opt, hf_opt, atol=1e-3)
    activations = hf_opt


def test_mlp():
    global activations
    cus_mlp = custom_gpt.transformer.blocks[0].mlp
    hf_mlp = hf_gpt.transformer.h[0].mlp

    cus_opt = cus_mlp(activations)
    hf_opt = hf_mlp(activations)
    assert torch.allclose(cus_opt, hf_opt, atol=1e-2)
