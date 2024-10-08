import random

from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    WhisperModel,
)
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt

from src.telepath import TelepathConfig, Telepath, TextDecoder
from utils.metrics import position_in_cycle

torch.set_grad_enabled(False)

torch.random.manual_seed(42)


def debug_hook(*args, **kwargs):
    breakpoint()


activatons = None
resid = None
ref_activations = None
ref_resid = None


config = TelepathConfig(
    encoder_pretrained_model="openai/whisper-small",
    decoder_pretrained_model="google/t5-v1_1-base",
    n_eeg_channels=1,
    fft_hop_length=1,
    encoder_block_size=1500,
    decoder_block_size=10,
    dropout=0.0,
    encoder_scale_exponent=-0.5,
)

model = Telepath.from_pretrained(config)
ref_encoder = WhisperModel.from_pretrained(
    config.encoder_pretrained_model
).get_encoder()
ref_config = T5Config.from_pretrained(config.decoder_pretrained_model)
ref_config.dropout_rate = 0.0
ref_decoder = T5ForConditionalGeneration.from_pretrained(
    config.decoder_pretrained_model, config=ref_config
).get_decoder()

for block in ref_decoder.block:
    block.layer[2].DenseReluDense.dropout.p = 0.0
ref_decoder.dropout.p = 0.0


def get_rtol(a, b):
    return torch.abs(a - b) / torch.abs(b)


def get_causal_mask_for_bias(bias: torch.Tensor):
    return torch.triu(torch.full_like(bias, torch.finfo(bias.dtype).min), diagonal=1)


def test_position_in_cycle():
    for i in range(1, 21):
        if i < 11:
            assert position_in_cycle((i, 10)) == i
        else:
            assert position_in_cycle((i, 10)) == i - 10


def test_encoder():
    model.encoder.embed_electrodes.weight = torch.nn.Parameter(
        torch.zeros_like(model.encoder.embed_electrodes.weight)
    )
    global activations
    global ref_activations
    inputs = torch.rand(3, 1, 80, 3000)
    activations = model.encoder(inputs)
    ref_activations = ref_encoder(inputs.squeeze(1), output_hidden_states=True)
    assert torch.equal(activations, ref_activations.last_hidden_state)


def test_relative_position_bias():
    attn = model.decoder.blocks[0].attn
    rp_bias = attn.rp_bias
    bias_live = rp_bias(10, 10)
    bias_static = attn.bias
    ref_attn = ref_decoder.block[0].layer[0].SelfAttention
    ref_bias = ref_attn.compute_bias(10, 10)
    # Test that the bias is calculated the same.
    assert torch.equal(bias_live, ref_bias)
    ref_bias = ref_bias + get_causal_mask_for_bias(ref_bias)
    # Test that the bias is recalculated correctly after loading weights.
    assert torch.equal(
        ref_bias,
        bias_static,
    )


def test_relative_position_attention():
    ipt = torch.randn(3, 10, model.config.d_model)
    attention = model.decoder.blocks[0].attn
    out = attention(x=ipt.clone())
    ref_attention = ref_decoder.block[0].layer[0].SelfAttention
    ref_out = ref_attention(
        ipt.clone(),
        mask=get_causal_mask_for_bias(ref_attention.compute_bias(10, 10)),
    )[0]
    assert torch.equal(out, ref_out)


def test_relative_decoder_block():
    ipt = torch.randn(3, 10, model.config.d_model)
    block = model.decoder.blocks[0]

    # ref_decoder.block[0].layer[2].DenseReluDense.dropout.p = 0.0
    out = block(x=ipt.clone(), xc=activations)
    ref_attention = ref_decoder.block[0]
    ref_out = ref_attention(
        ipt.clone(),
        encoder_hidden_states=ref_activations.last_hidden_state,
        attention_mask=get_causal_mask_for_bias(
            ref_attention.layer[0].SelfAttention.compute_bias(10, 10)
        ),
    )[0]
    assert torch.allclose(out, ref_out, atol=1e-3)


def test_decoder():
    global activations
    global ref_activations
    inputs = torch.tensor([[0, 1], [0, 1], [0, 1]])
    activations = model.decoder(
        inputs, activations, return_hidden_states=True
    )  # , attention_mask)
    ref_activations = ref_decoder(
        input_ids=inputs,
        encoder_hidden_states=ref_activations.last_hidden_state,
        # encoder_attention_mask=attention_mask,
        output_hidden_states=True,
    )
    assert torch.equal(activations, ref_activations.last_hidden_state)


# # iteration = 0


# # def test_generation():
# #     global iteration
# #     # TODO: Modify this to test for the continuous alignment of the embedding and the output tokens.
# #     config = TelepathConfig(
# #         n_eeg_channels=63, pretrained_whisper="openai/whisper-tiny", fft_hop_length=1
# #     )
# #     dec = Telepath.from_pretrained(config).decoder
# #     stop_token_id = 50256
# #     max_length = 10
# #     def dummy_forward(input_ids, embeddings, inference) -> torch.Tensor:
# #         global iteration
# #         # Each output is the value of the embedding for the given input unless `embedding_value == iteration`, then we append the stop token to end the sequence.
# #         out = torch.tensor(
# #             [
# #                 [stop_token_id] if iteration == embed[0][0] else [embed[0][0]]
# #                 for embed in embeddings
# #             ]
# #         )

# #         out = F.one_hot(
# #             out,
# #             dec.vocab_size,
# #         )
# #         iteration += 1
# #         return out

# #     dec.forward = dummy_forward  # type: ignore

# #     # Ensure that we are not just getting lucky based on the order that the generations are completing (which depends on the embedding).
# #     for embeddings in [
# #         torch.tensor([[[1]], [[2]], [[3]]]),
# #         torch.tensor([[[1]], [[3]], [[2]]]),
# #         torch.tensor([[[2]], [[1]], [[3]]]),
# #         torch.tensor([[[2]], [[3]], [[1]]]),
# #         torch.tensor([[[3]], [[1]], [[2]]]),
# #         torch.tensor([[[3]], [[2]], [[1]]]),
# #     ]:
# #         iteration = 0
# #         print(embeddings)
# #         generations = dec.generate(
# #             input_ids=torch.tensor([[0], [0], [0]]),
# #             embed=embeddings,
# #             max_length=max_length,
# #             stop_token=stop_token_id,
# #         )
# #         print(generations)

# #         assert sorted(generations, key=lambda g: len(g)) == [
# #             [0, 1, stop_token_id],
# #             [0, 2, 2, stop_token_id],
# #             [0, 3, 3, 3, stop_token_id],
# #         ]

# # def test_encoder_forward():


# def test_telepath():
#     config = TelepathConfig(
#         n_eeg_channels=63,
#         pretrained_whisper="openai/whisper-tiny",
#         fft_hop_length=1,
#         encoder_block_size=50,
#         decoder_block_size=100,
#     )

#     model = Telepath(config)

#     features = torch.rand(3, 63, 80, 100)
#     ipt_ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
#     attention_mask = torch.ones_like(ipt_ids)
#     output = model(input_ids=ipt_ids, eeg=features, attention_mask=attention_mask)
#     assert output[1].shape == (3, 2, config.decoder_vocab_size)


# def test_telepath_from_pretrained():
#     model.encoder.embed_electrodes.weight = torch.nn.Parameter(
#         torch.zeros_like(model.encoder.embed_electrodes.weight)
#     )

#     features = torch.rand(3, 1, 80, 3000)
#     ipt_ids = torch.tensor([[0, 1], [0, 1], [0, 1]])
#     attention_mask = torch.ones_like(ipt_ids)
#     output = model(input_ids=ipt_ids, eeg=features, attention_mask=attention_mask)
#     ref_output = ref_model(
#         input_features=features.squeeze(1),
#         decoder_input_ids=ipt_ids,
#         attention_mask=attention_mask,
#         output_hidden_states=True,
#     )
#     assert output[0].shape == ref_output.encoder_last_hidden_state.shape
#     assert torch.allclose(
#         output[0], ref_output.encoder_last_hidden_state, rtol=1e-3
#     ), f"hidden states rtol={torch.max(torch.abs(output[0] - ref_output.encoder_last_hidden_state)/torch.abs(ref_output.encoder_last_hidden_state))}"
#     assert output[1].shape == ref_output.logits.shape
#     assert torch.allclose(
#         output[1], ref_output.logits, rtol=1e-3
#     ), f"logits rtol={torch.max(torch.abs(output[1] - ref_output.logits)/torch.abs(ref_output.logits))}"
