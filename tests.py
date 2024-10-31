from dataclasses import dataclass
import random
from typing import Any, Callable

from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    WhisperModel,
)
import torch
from torch import nn
from torch.nn import functional as F

from src.components.activations import GEGLU
from src.telepath import TelepathConfig, TelepathGenerator, TextDecoder, TextEncoder
from utils.metrics import position_in_cycle

torch.set_grad_enabled(False)

torch.random.manual_seed(42)


def debug_hook(*args, **kwargs):
    breakpoint()


activatons = None
resid = None
ref_activations = None
ref_resid = None
REF_TEXT_MODEL = "google/t5-v1_1-base"

ref_config = T5Config.from_pretrained(REF_TEXT_MODEL)
config = TelepathConfig(
    n_eeg_channels=63,
    text_encoder_pretrained_model=REF_TEXT_MODEL,
    text_encoder_vocab_size=ref_config.vocab_size,
    text_encoder_block_size=10,
    d_model=ref_config.d_model,
    text_encoder_d_mlp=ref_config.d_ff,
    n_heads=ref_config.num_heads,
    text_encoder_n_layers=ref_config.num_layers,
    dropout=0.0,
)
ref_config.dropout_rate = 0.0
ref_text_model = T5ForConditionalGeneration.from_pretrained(
    config.text_encoder_pretrained_model, config=ref_config
)


def save_tensor_forward_pre_hook(
    module: nn.Module,
    input: Any,
    input_to_target_map: Callable[[Any], torch.Tensor],
    name: str,
) -> None:
    pass


def save_tensor_forward_hook(
    module: nn.Module,
    input: Any,
    output: Any,
    output_to_target_map: Callable[[Any], torch.Tensor],
    name: str,
) -> None:
    pass


@dataclass
class ModuleMap:
    name: str
    input_to_target_map: Callable[[Any], torch.Tensor]
    output_to_target_map: Callable[[Any], torch.Tensor]


def test_divergence(
    module1: nn.Module,
    module2: nn.Module,
    module_map_pairs: list[tuple[ModuleMap, ModuleMap]],
):
    pass


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


def test_text_encoder():
    global activations
    global ref_activations
    ref_encoder = ref_text_model.get_encoder()
    encoder = TextEncoder.from_pretrained(config)
    inputs = torch.tensor([[0, 1], [0, 1], [0, 1]])
    encoder.blocks[0].register_forward_pre_hook(debug_hook)
    ref_encoder.block[0].register_forward_pre_hook(debug_hook)
    activations = encoder(inputs)
    ref_activations = ref_encoder(input_ids=inputs)
    assert torch.equal(activations, ref_activations.last_hidden_state)


def test_text_decoder():
    global activations
    global ref_activations
    ref_decoder = ref_text_model.get_decoder()
    decoder = TextEncoder.from_pretrained(config)
    inputs = torch.tensor([[0, 1], [0, 1], [0, 1]])
    activations = decoder(inputs, activations, return_hidden_states=True)
    ref_activations = ref_decoder(
        input_ids=inputs,
        encoder_hidden_states=ref_activations.last_hidden_state,
    )
    assert torch.equal(activations, ref_activations.last_hidden_state)


iteration = 0


def test_generation():
    global iteration
    # TODO: Modify this to test for the continuous alignment of the embedding and the output tokens.
    decoder = TextDecoder.from_pretrained(config)

    stop_token_id = 999
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
            decoder.vocab_size,
        )
        iteration += 1
        return out

    decoder.forward = dummy_forward  # type: ignore

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
        generations = decoder.generate(
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
