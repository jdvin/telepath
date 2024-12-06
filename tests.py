from contextlib import contextmanager
from dataclasses import dataclass
import functools
import os
import random
from typing import Any, Callable
from transformers import (
    AutoModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    WhisperModel,
)
from loguru import logger
import pytest
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from src.components.activations import GEGLU
from src.telepath import TelepathConfig, TelepathTrainer, TextDecoder, TextEncoder
from utils.metrics import position_in_cycle

torch.set_grad_enabled(False)

torch.random.manual_seed(42)


def debug_hook(*args, **kwargs):
    breakpoint()


activations = None
resid = None
ref_activations = None
ref_resid = None
REF_TEXT_MODEL = "google-t5/t5-large"

ref_config = T5Config.from_pretrained(REF_TEXT_MODEL)
config = TelepathConfig(
    n_eeg_channels=63,
    text_encoder_pretrained_model="sentence-transformers/sentence-t5-large",
    neural_encoder_block_size=400,
    text_encoder_block_size=10,
    d_model=1024,
    shared_emb_dim=768,
    neural_encoder_d_mlp=3072,
    text_encoder_d_mlp=4096,
    text_encoder_vocab_size=32128,
    n_heads=16,
    neural_encoder_n_layers=24,
    text_encoder_n_layers=24,
    cache_text_embeddings=True,
    text_encoder_use_geglu=False,
    dropout=0.0,
)
ref_config.dropout_rate = 0.0
ref_text_model = T5ForConditionalGeneration.from_pretrained(
    REF_TEXT_MODEL, config=ref_config
)
ref_encoder = ref_text_model.get_encoder()
ref_decoder = ref_text_model.get_decoder()

identity = lambda x: x
element_0 = lambda x: x[0]
hf_t5_encoder_inputs_to_kwargs = lambda inputs: {"input_ids": inputs[0]}
text_encoder_attn_inputs_to_kwargs = lambda inputs: {"x": inputs[0]}


def save_tensor_forward_pre_hook(
    module: nn.Module,
    input: Any,
    input_to_target_map: Callable[[Any], torch.Tensor],
    name: str,
) -> None:
    t = input_to_target_map(input)
    torch.save(t, name + ".pt")


def save_tensor_forward_hook(
    module: nn.Module,
    input: Any,
    output: Any,
    output_to_target_map: Callable[[Any], torch.Tensor],
    name: str,
) -> None:
    t = output_to_target_map(output)
    torch.save(t, name + ".pt")


@dataclass
class ModuleMap:
    name: str
    input_to_target_map: Callable[[Any], torch.Tensor] = element_0
    output_to_target_map: Callable[[Any], torch.Tensor] = identity


def install_hooks(
    module: nn.Module, name: str, module_map: ModuleMap
) -> tuple[str, str]:
    pre_hook_tensor_path = name + "-pre"
    pre_hook = functools.partial(
        save_tensor_forward_pre_hook,
        input_to_target_map=module_map.input_to_target_map,
        name=pre_hook_tensor_path,
    )
    module.register_forward_pre_hook(pre_hook)
    post_hook_tensor_path = name + "-post"
    post_hook = functools.partial(
        save_tensor_forward_hook,
        output_to_target_map=module_map.input_to_target_map,
        name=post_hook_tensor_path,
    )
    module.register_forward_hook(post_hook)
    return pre_hook_tensor_path, post_hook_tensor_path


def check_divergence(
    module1: nn.Module,
    module1_inputs_to_kwargs: Callable[[tuple], dict],
    module2: nn.Module,
    module2_inputs_to_kwargs: Callable[[tuple], dict],
    test_input: tuple,
    module_map_pairs: list[tuple[ModuleMap, ModuleMap]],
):
    # List of [module_index, (module_path_pre, module_path_post)].
    module1_indexed_tensor_paths: list[tuple[int, tuple[str, str]]] = []
    module2_indexed_tensor_paths: list[tuple[int, tuple[str, str]]] = []
    for module_map_pair in module_map_pairs:
        # TODO: This for-loop logic can be included in the `install_hooks` method, it is repeated verbatim per module.
        for i, (name, mod) in enumerate(module1.named_modules()):
            # Install hooks in each matching module and capture the paths.
            if name.endswith(module_map_pair[0].name):
                module1_indexed_tensor_paths.append(
                    (i, install_hooks(mod, name, module_map_pair[0]))
                )
        for i, (name, mod) in enumerate(module2.named_modules()):
            if name.endswith(module_map_pair[1].name):
                module2_indexed_tensor_paths.append(
                    (i, install_hooks(mod, name, module_map_pair[1]))
                )
    assert len(module1_indexed_tensor_paths) == len(module2_indexed_tensor_paths)
    module1(**module1_inputs_to_kwargs(test_input))
    module2(**module2_inputs_to_kwargs(test_input))
    module1_tensor_paths: list[str] = [
        module_tensor_path
        for module_tensor_paths in sorted(module1_indexed_tensor_paths, key=element_0)
        for module_tensor_path in module_tensor_paths[1]
    ]
    module2_tensor_paths: list[str] = [
        module_tensor_path
        for module_tensor_paths in sorted(module2_indexed_tensor_paths, key=element_0)
        for module_tensor_path in module_tensor_paths[1]
    ]
    for (
        module1_tensor_path,
        module2_tensor_path,
    ) in zip(module1_tensor_paths, module2_tensor_paths):
        print(f"Checking equivalence: {module1_tensor_path} == {module2_tensor_path}.")
        m1 = torch.load(module1_tensor_path + ".pt")
        m2 = torch.load(module2_tensor_path + ".pt")
        assert torch.equal(
            m1, m2
        ), f"Model divergence detected: {module1_tensor_path} != {module2_tensor_path}"
        print("Success!")


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


def check_relative_position_bias(model, ref_model, is_causal):
    attn = model.blocks[0].attn
    rp_bias = attn.rp_bias
    bias_live = rp_bias(10, 10)
    bias_static = attn.bias
    ref_attn = ref_model.block[0].layer[0].SelfAttention
    ref_bias = ref_attn.compute_bias(10, 10)
    # Test that the bias is calculated the same.
    assert torch.equal(bias_live, ref_bias)
    if is_causal:
        ref_bias = ref_bias + get_causal_mask_for_bias(ref_bias)
    # Test that the bias is recalculated correctly after loading weights.
    assert torch.equal(
        ref_bias,
        bias_static,
    )


def test_text_encoder():
    global activations
    global ref_activations
    global ref_encoder
    encoder = TextEncoder.from_pretrained(ref_encoder, config)
    check_relative_position_bias(encoder, ref_encoder, False)
    inputs = torch.tensor([[0, 1], [0, 1], [0, 1]])
    # encoder.blocks[0].register_forward_pre_hook(debug_hook)
    # ref_encoder.block[0].register_forward_pre_hook(debug_hook)
    map = [
        (
            ModuleMap(".q_proj"),
            ModuleMap(".q"),
        ),
        (
            ModuleMap(".k_proj"),
            ModuleMap(".k"),
        ),
        (
            ModuleMap(".v_proj"),
            ModuleMap(".v"),
        ),
        (
            ModuleMap(".out_proj"),
            ModuleMap(".o"),
        ),
    ]
    check_divergence(
        encoder,
        text_encoder_attn_inputs_to_kwargs,
        ref_encoder,
        hf_t5_encoder_inputs_to_kwargs,
        (inputs,),
        map,
    )
    activations = encoder(inputs)
    ref_activations = ref_encoder(input_ids=inputs).last_hidden_state
    assert torch.equal(activations, ref_activations)


def test_text_decoder():
    global activations
    global ref_activations
    global ref_decoder
    decoder = TextDecoder.from_pretrained(ref_decoder, config)
    inputs = torch.tensor([[0, 1], [0, 1], [0, 1]])
    activations = decoder(inputs, True, activations)
    ref_activations = ref_decoder(
        input_ids=inputs,
        encoder_hidden_states=ref_activations,
    )
    assert torch.equal(activations, ref_activations.last_hidden_state)


iteration = 0


def test_generation():
    global iteration
    global ref_decoder
    # TODO: Modify this to test for the continuous alignment of the embedding and the output tokens.
    decoder = TextDecoder.from_pretrained(ref_decoder, config)

    stop_token_id = 999
    max_length = 10

    def dummy_forward(x, xc, inference, kv_cache) -> torch.Tensor:
        global iteration
        # Each output is the value of the embedding for the given input unless `embedding_value == iteration`, then we append the stop token to end the sequence.
        out = torch.tensor(
            [
                [stop_token_id] if iteration == embed[0][0] else [embed[0][0]]
                for embed in xc
            ]
        )

        out = F.one_hot(
            out,
            decoder.n_vocab,
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
        generations = decoder.generate(
            input_ids=torch.tensor([[0], [0], [0]]),
            embed=embeddings,
            max_length=max_length,
            stop_token=stop_token_id,
        )

        assert sorted(generations, key=lambda g: len(g)) == [
            [0, 1, stop_token_id],
            [0, 2, 2, stop_token_id],
            [0, 3, 3, 3, stop_token_id],
        ]


@contextmanager
def distributed_env(rank, world_size):
    """Context manager for distributed setup"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo", world_size=world_size, rank=rank  # Use gloo for testing
    )

    try:
        yield
    finally:
        dist.destroy_process_group()


def run_test_on_process(rank, world_size, test_fn):
    """Function that runs on each spawned process"""
    with distributed_env(rank, world_size):
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        test_fn(rank, world_size, device)


def run_distributed_test(test_fn, world_size=2):
    """Spawn processes and run the test"""
    mp.spawn(run_test_on_process, args=(world_size, test_fn), nprocs=world_size)


def _test_step(rank, world_size, device):
    """The actual test implementation"""
    logger.info(f"[{rank}/{world_size}] Starting test")

    model = TelepathTrainer(config, rank, world_size)
    model.to(device)
    # Create same input on all ranks
    torch.manual_seed(42)
    batch = {
        "input_features": torch.randn(4, 63, 400, device=device),
        "input_ids": torch.randint(1, 100, (4, 10), device=device),
        "object_ids": torch.randint(1, 100, (4,), device=device),
    }

    logger.info(f"[{rank}/{world_size}] Running model step")
    loss, all_logits, labels = model.step(batch)

    # Verify all ranks got the same loss
    gathered_losses = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss)

    if rank == 0:
        assert all(not torch.isnan(loss) for loss in gathered_losses)

    logger.info(f"[{rank}/{world_size}] Test complete")


def test_sigmoid_loss():
    """Non-distributed test"""
    model = TelepathTrainer(config, 0, 1)
    logits = torch.randn(2, 2)
    labels = 2 * (torch.rand(2, 2) > 0.5).float() - 1

    loss = model.sigmoid_loss(logits, labels)
    assert not torch.isnan(loss)
    # assert loss.requires_grad


@pytest.mark.distributed
def test_step_distributed():
    """The test that pytest runs"""
    run_distributed_test(_test_step)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
