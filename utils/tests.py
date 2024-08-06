import random

from transformers import AutoTokenizer, WhisperForConditionalGeneration
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt

from src.telepath import TelepathConfig, Telepath, TextDecoder
from utils.metrics import position_in_cycle

torch.set_grad_enabled(False)

torch.random.manual_seed(42)

activatons = None
resid = None
ref_activations = None
ref_resid = None

config = TelepathConfig(
    pretrained_whisper="openai/whisper-tiny",
    n_eeg_channels=1,
    fft_hop_length=1,
    encoder_block_size=1500,
    decoder_block_size=10,
    dropout=0.0,
    scale_exponent=-0.5,
)

model = Telepath.from_pretrained(config)
ref_model = WhisperForConditionalGeneration.from_pretrained(config.pretrained_whisper)


def get_rtol(a, b):
    return torch.abs(a - b) / torch.abs(b)


def test_position_in_cycle():
    for i in range(1, 21):
        if i < 11:
            assert position_in_cycle((i, 10)) == i
        else:
            assert position_in_cycle((i, 10)) == i - 10


def test_encoder_conv1():
    global activations
    global ref_activations
    inputs = torch.rand(3, 80, 1, 3000)
    activations = model.encoder.conv1(inputs)
    ref_activations = ref_model.get_encoder().conv1(inputs.squeeze(2))
    assert torch.allclose(activations.squeeze(2), ref_activations, rtol=1e-3)
    print(
        "post conv1 rtol:", torch.max(get_rtol(activations.squeeze(2), ref_activations))
    )


def test_encoder_conv2():
    global activations
    global ref_activations
    ref_activations = ref_model.get_encoder().conv2(ref_activations)
    activations = model.encoder.conv2(activations)
    assert torch.allclose(activations.squeeze(2), ref_activations, rtol=1e-3)
    print(
        "post conv2 rtol:", torch.max(get_rtol(activations.squeeze(2), ref_activations))
    )


def test_positional_embeddings():
    global activations
    global ref_activations
    ref_activations = ref_activations.permute(0, 2, 1)
    activations = activations.permute(0, 3, 2, 1)
    activations = activations + model.encoder.embed_positions.weight[None, :, None, :]
    ref_activations = ref_activations + ref_model.get_encoder().embed_positions.weight
    activations = activations.reshape(-1, 1500, 384)
    assert torch.equal(activations, ref_activations)
    print("post pos emb rtol:", torch.max(get_rtol(activations, ref_activations)))


def test_attn_layer_norm():
    global activations
    global ref_activations
    global resid
    global ref_resid
    resid = activations.detach()
    ref_resid = ref_activations.detach()
    ref_activations = (
        ref_model.get_encoder().layers[0].self_attn_layer_norm(ref_activations)
    )
    activations = model.encoder.blocks[0].attn_ln(activations)
    assert torch.allclose(activations, ref_activations, rtol=1e-3)
    print("post attn lnorm rtol:", torch.max(get_rtol(activations, ref_activations)))


def test_attention():
    global activations
    global ref_activations
    ref_activations = ref_model.get_encoder().layers[0].self_attn(ref_activations)[0]
    activations = model.encoder.blocks[0].attn(activations)
    print(ref_activations.shape)
    print(activations.shape)
    for i, (batch, ref_batch) in enumerate(zip(activations, ref_activations)):
        plt.imsave(f"batch{i}.png", get_rtol(batch, ref_batch))
        assert torch.allclose(batch, ref_batch, rtol=1e-3)
    assert torch.allclose(
        activations, ref_activations, rtol=1e-3
    ), f"rtol={torch.max(get_rtol(activations, ref_activations[0]))}"
    print("post attn rtol:", torch.max(get_rtol(activations, ref_activations[0])))


def test_attn_residual():
    global activations
    global ref_activations
    global resid
    global ref_resid
    ref_activations = ref_resid + ref_activations
    activations = resid + activations
    resid = activations.clone()
    ref_resid = ref_activations.clone()
    assert torch.allclose(activations, ref_activations, rtol=1e-3)
    print("post residual rtol:", torch.max(get_rtol(activations, ref_activations)))


def test_mlp_layer_norm():
    global activations
    global ref_activations
    ref_activations = (
        ref_model.get_encoder().layers[0].final_layer_norm(ref_activations)
    )
    activations = model.encoder.blocks[0].mlp_ln(activations)
    assert torch.allclose(activations, ref_activations, rtol=1e-3)
    print("post mlp lnorm rtol:", torch.max(get_rtol(activations, ref_activations)))


def test_mlp():
    global activations
    global ref_activations
    mlp = model.encoder.blocks[0].mlp
    ref_activations = ref_model.get_encoder().layers[0].fc1(ref_activations)
    ref_activations = ref_model.get_encoder().layers[0].activation_fn(ref_activations)
    ref_activations = ref_model.get_encoder().layers[0].fc2(ref_activations)
    activations = mlp(activations)
    assert torch.allclose(activations, ref_activations, rtol=1e-3)
    print("post mlp rtol:", torch.max(get_rtol(activations, ref_activations)))


def test_mlp_residual():
    global activations
    global ref_activations
    global resid
    global ref_resid
    ref_activations = ref_resid + ref_activations
    activations = resid + activations
    resid = activations.clone()
    ref_resid = ref_activations.clone()
    assert torch.allclose(activations, ref_activations, rtol=1e-3)
    print("post mlp residual rtol:", torch.max(get_rtol(activations, ref_activations)))


def test_encoder_block():
    global activations
    layer_head_mask = torch.ones(model.encoder.blocks[0].attn.n_heads)
    attention_mask = torch.ones(3, 1, 1500, 1500)
    ref_activations = ref_model.get_encoder().layers[0](
        activations,
        attention_mask,
        layer_head_mask,
    )
    activations = model.encoder.blocks[0](activations)
    for i, (batch, ref_batch) in enumerate(zip(activations, ref_activations[0])):
        plt.imsave(f"enc_batch{i}.png", get_rtol(batch, ref_batch))
        # assert torch.allclose(batch, ref_batch, rtol=1e-3)
    assert torch.allclose(
        activations, ref_activations, rtol=1e-3
    ), f"rtol={get_rtol(activations, ref_activations)}"


# iteration = 0


# def test_generation():
#     global iteration
#     # TODO: Modify this to test for the continuous alignment of the embedding and the output tokens.
#     config = TelepathConfig(
#         n_eeg_channels=63, pretrained_whisper="openai/whisper-tiny", fft_hop_length=1
#     )
#     dec = Telepath.from_pretrained(config).decoder
#     stop_token_id = 50256
#     max_length = 10
#     def dummy_forward(input_ids, embeddings, inference) -> torch.Tensor:
#         global iteration
#         # Each output is the value of the embedding for the given input unless `embedding_value == iteration`, then we append the stop token to end the sequence.
#         out = torch.tensor(
#             [
#                 [stop_token_id] if iteration == embed[0][0] else [embed[0][0]]
#                 for embed in embeddings
#             ]
#         )

#         out = F.one_hot(
#             out,
#             dec.vocab_size,
#         )
#         iteration += 1
#         return out

#     dec.forward = dummy_forward  # type: ignore

#     # Ensure that we are not just getting lucky based on the order that the generations are completing (which depends on the embedding).
#     for embeddings in [
#         torch.tensor([[[1]], [[2]], [[3]]]),
#         torch.tensor([[[1]], [[3]], [[2]]]),
#         torch.tensor([[[2]], [[1]], [[3]]]),
#         torch.tensor([[[2]], [[3]], [[1]]]),
#         torch.tensor([[[3]], [[1]], [[2]]]),
#         torch.tensor([[[3]], [[2]], [[1]]]),
#     ]:
#         iteration = 0
#         print(embeddings)
#         generations = dec.generate(
#             input_ids=torch.tensor([[0], [0], [0]]),
#             embed=embeddings,
#             max_length=max_length,
#             stop_token=stop_token_id,
#         )
#         print(generations)

#         assert sorted(generations, key=lambda g: len(g)) == [
#             [0, 1, stop_token_id],
#             [0, 2, 2, stop_token_id],
#             [0, 3, 3, 3, stop_token_id],
#         ]

# def test_encoder_forward():


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
    assert output[1].shape == (3, 2, config.decoder_vocab_size)


def test_telepath_from_pretrained():
    model.encoder.embed_electrodes.weight = torch.nn.Parameter(
        torch.zeros_like(model.encoder.embed_electrodes.weight)
    )

    features = torch.rand(3, 1, 80, 3000)
    ipt_ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
    attention_mask = torch.ones_like(ipt_ids)
    output = model(input_ids=ipt_ids, eeg=features, attention_mask=attention_mask)
    ref_output = ref_model(
        input_features=features.squeeze(1),
        decoder_input_ids=ipt_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    assert output[0].shape == ref_output.encoder_last_hidden_state.shape
    assert torch.allclose(
        output[0], ref_output.encoder_last_hidden_state, rtol=1e-3
    ), f"hidden states rtol={torch.max(torch.abs(output[0] - ref_output.encoder_last_hidden_state)/torch.abs(ref_output.encoder_last_hidden_state))}"
    assert output[1].shape == ref_output.logits.shape
    assert torch.allclose(
        output[1], ref_output.logits, rtol=1e-3
    ), f"logits rtol={torch.max(torch.abs(output[1] - ref_output.logits)/torch.abs(ref_output.logits))}"
