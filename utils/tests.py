import random

from transformers import AutoTokenizer, WhisperForConditionalGeneration
import torch
from torch.nn import functional as F

# from src.gpt import GPT
from src.telepath import TelepathConfig, Telepath, TextDecoder


def test_transform():
    raise NotImplementedError()
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


# activatons = None

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# hf_gpt = AutoModelForCausalLM.from_pretrained("gpt2").eval()


# def test_embeddings():
#     global activations
#     inputs = tokenizer("The quick brown", return_tensors="pt")["input_ids"]

#     hf_emb = hf_gpt.transformer.wte(inputs)
#     custom_emb = custom_gpt.transformer.wte(inputs)
#     assert torch.equal(hf_emb, custom_emb)
#     pos = torch.arange(0, inputs.size(-1), dtype=torch.long)
#     hf_posemb = hf_gpt.transformer.wpe(pos)
#     custom_posemb = custom_gpt.transformer.wpe(pos)
#     assert torch.equal(hf_posemb, custom_posemb)

#     activations = hf_emb + hf_posemb


# def test_layer_norm():
#     global activations
#     cus_opt = custom_gpt.transformer.blocks[0].ln_1(activations)
#     hf_opt = hf_gpt.transformer.h[0].ln_1(activations)
#     assert torch.allclose(cus_opt, hf_opt, atol=1e-3)
#     activations = hf_opt


# def test_attention():
#     global activations
#     cus_attn = custom_gpt.transformer.blocks[0].attn
#     hf_attn = hf_gpt.transformer.h[0].attn
#     cus_opt = cus_attn(activations)
#     hf_opt = hf_attn(activations)[0]
#     assert torch.allclose(cus_opt, hf_opt, atol=1e-3)
#     activations = hf_opt


# def test_mlp():
#     global activations
#     cus_mlp = custom_gpt.transformer.blocks[0].mlp
#     hf_mlp = hf_gpt.transformer.h[0].mlp

#     cus_opt = cus_mlp(activations)
#     hf_opt = hf_mlp(activations)
#     assert torch.allclose(cus_opt, hf_opt, atol=1e-2)


iteration = 0


def test_generation():
    global iteration
    # TODO: Modify this to test for the continuous alignment of the embedding and the output tokens.
    config = TelepathConfig(
        n_eeg_channels=63, pretrained_whisper="openai/whisper-tiny", fft_hop_length=1
    )
    dec = Telepath.from_pretrained(config).decoder
    stop_token_id = 50256
    max_length = 10

    def dummy_forward(input_ids, embeddings, inference) -> torch.Tensor:
        global iteration
        # Each output is the value of the embedding for the given input unless `embedding_value == iteration`, then we append the stop token to end the sequence.
        out = torch.tensor(
            [
                [stop_token_id] if iteration == embed[0][0] else [embed[0][0]]
                for embed in embeddings
            ]
        )

        out = F.one_hot(
            out,
            dec.vocab_size,
        )
        iteration += 1
        return out

    dec.forward = dummy_forward  # type: ignore

    # Ensure that we are not just getting lucky based on the order that the generations are completing (which depends on the embedding).
    for embeddings in [
        torch.tensor([[[1]], [[2]], [[3]]]),
        torch.tensor([[[1]], [[3]], [[2]]]),
        torch.tensor([[[2]], [[1]], [[3]]]),
        torch.tensor([[[2]], [[3]], [[1]]]),
        torch.tensor([[[3]], [[1]], [[2]]]),
        torch.tensor([[[3]], [[2]], [[1]]]),
    ]:
        iteration = 0
        print(embeddings)
        generations = dec.generate(
            input_ids=torch.tensor([[0], [0], [0]]),
            embed=embeddings,
            max_length=max_length,
            stop_token=stop_token_id,
        )
        print(generations)

        assert sorted(generations, key=lambda g: len(g)) == [
            [0, 1, stop_token_id],
            [0, 2, 2, stop_token_id],
            [0, 3, 3, 3, stop_token_id],
        ]


def test_telepath():
    config = TelepathConfig(
        n_eeg_channels=63,
        pretrained_whisper="openai/whisper-tiny",
        fft_hop_length=1,
        encoder_block_size=50,
        decoder_block_size=100,
    )

    model = Telepath(config)

    features = torch.rand(3, 63, 80, 100)
    ipt_ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
    attention_mask = torch.ones_like(ipt_ids)
    output = model(input_ids=ipt_ids, eeg=features, attention_mask=attention_mask)
    assert output[0].shape == (3, 2, config.decoder_vocab_size)


def test_telepath_from_pretrained():
    config = TelepathConfig(
        pretrained_whisper="openai/whisper-tiny",
        n_eeg_channels=1,
        fft_hop_length=1,
        encoder_block_size=1500,
        decoder_block_size=10,
    )

    model = Telepath.from_pretrained(config)
    model.encoder.embed_electrodes.weight = torch.nn.Parameter(
        torch.zeros_like(model.encoder.embed_electrodes.weight)
    )
    ref_model = WhisperForConditionalGeneration.from_pretrained(
        config.pretrained_whisper
    )

    features = torch.rand(3, 1, 80, 3000)
    ipt_ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
    attention_mask = torch.ones_like(ipt_ids)
    enc, logits = model(input_ids=ipt_ids, eeg=features, attention_mask=attention_mask)
    ref_output = ref_model(
        input_features=features.squeeze(1),
        decoder_input_ids=ipt_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    breakpoint()
    assert output[0].shape == (3, 2, config.decoder_vocab_size)
