from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import pandas as pd
import torch
from torch import Tensor, tensor
from torch.distributed import (
    ReduceOp,
    all_gather_object,
    all_reduce,
    all_gather_into_tensor,
)
import wandb
from transformers import WhisperTokenizer

THINGS_CONCEPTS_PATH = "data/things_concepts.csv"
SYNONYM_MAP = {
    object_word.lower().strip(): [
        synonym.lower().replace("_", " ").strip()
        for synonym in synonyms.split(",")
        if synonym.lower().replace("_", " ").strip() != object_word.lower().strip()
    ]
    for object_word, synonyms in pd.read_csv(THINGS_CONCEPTS_PATH)[
        ["Word", "WordNet Synonyms"]
    ].values
}


class MetricLogType(Enum):
    SCALAR = "scalar"
    PLOT = "plot"
    TABLE = "table"


class MetricResetRule(Enum):
    ON_LOG = "on_log"
    ON_EPOCH = "on_epoch"
    MANUAL = "manual"


class MetricType(Enum):
    STATE = "state"
    GENERATION = "generation"


Loggable = int | float
Plottable = list[tuple[float, float]]
MetricState = int | float | Tensor | list | dict | None
# TODO: Should this be dataclass with explicit possible values?
InferenceArtefacts = (
    Mapping[str, Tensor | float | int | list[str]]
    | tuple[Tensor | float | int, ...]
    | list[str]
    | Tensor
    | float
    | int
)


def return_first_value(x: InferenceArtefacts) -> MetricState:
    if isinstance(x, dict):
        list(x.values())[0]
    elif isinstance(x, tuple):
        return x[0]
    else:
        assert isinstance(x, (Tensor, float, int))
        return x


def divide(state: MetricState) -> Tensor:
    assert isinstance(state, Tensor)
    return state[0] / state[1]


add = lambda x, y: x + y
identity = lambda x: x
replace = lambda x, y: y


def all_reduce_mean(t: Tensor | list, ws: int) -> Tensor:
    assert isinstance(t, Tensor)
    all_reduce(t, op=ReduceOp.AVG)
    return t


def all_reduce_sum(t: Tensor | list, ws: int) -> Tensor:
    assert isinstance(t, Tensor)
    all_reduce(t, op=ReduceOp.SUM)
    return t


def all_gather_concat(t: Tensor | list, ws: int) -> Tensor:
    assert isinstance(t, Tensor)
    # Create a tensor to hold the concatenated values.
    out = torch.zeros(t.size(0) * ws, *t.size()[1:], dtype=t.dtype, device=t.device)
    all_gather_into_tensor(out, t.contiguous())
    return out


def all_gather_append(t: Tensor | list, ws: int) -> list:
    assert isinstance(t, list)
    out = [None] * ws
    all_gather_object(out, t)
    return out


def row_concat(t1: MetricState, t2: MetricState) -> Tensor:
    assert isinstance(t1, Tensor) and isinstance(t2, Tensor)
    return torch.concat([t1, t2], dim=0)


def position_in_epoch(inference_artefacts: InferenceArtefacts) -> int:
    """For a given intra-epoch cycle (e.g., step or microstep), given the absolute number of the current cycle (`inference_artefacts[0]`; indexed from 1)
    and the total number of cycles in an epoch (`inference_artefacts[1]`; index from 1), return the number of the current cycle relative to the current epoch.
    """
    assert isinstance(inference_artefacts, tuple)
    out = (inference_artefacts[0] - 1) % inference_artefacts[1] + 1
    assert isinstance(out, int)
    return out


def flatten_ranks(root: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    out = defaultdict(list)
    for rank in root:
        for key, item in rank.items():
            out[key].extend(item)
    return out


def get_accuracy(generations: list[dict[str, list[str]]]) -> float:
    """Slightly less naiive accuracy calculation."""
    accuracy = 0

    if isinstance(generations, list):
        flattened_generations = flatten_ranks(generations)
    else:
        flattened_generations = generations
    for target_text, pred_text in zip(
        flattened_generations["targets"], flattened_generations["predictions"]
    ):
        # TODO: This should be done in the data processing stage - check!.
        target_text = target_text.lower().strip()
        pred_text = pred_text.lower().strip()
        # Using `in` allows to account for noise in the generation at the expense of speed.
        pred_text_is_synonym = any(
            [synonym in pred_text for synonym in SYNONYM_MAP.get("true_text") or []]
        )
        if target_text in pred_text or pred_text_is_synonym:
            accuracy += 1 / len(flattened_generations["predictions"])
    return accuracy


def construct_table(
    generations: dict[str, list[str]] | list[dict[str, list[str]]]
) -> dict:
    """Construct a wandb table for logging generations.

    Generations from each rank are nested in batches."""
    out = wandb.Table(columns=["Target", "Prediction"])
    if isinstance(generations, list):
        column_data = flatten_ranks(generations)
    else:
        column_data = generations
    for target, prediction in zip(column_data["targets"], column_data["predictions"]):
        out.add_data(target, prediction)
    return out


@dataclass
class Metric:

    """An object for the unified tracking and logging of information about a training run.

    Attributes:
        state: Holds the current relevant features of the metric prior to computation. Initial state must be passed in at instantiation.
        transform_fn: Maps InferenceArtefacts to metric state. By default, this function
        accum_fn: Combines two instances of the metrics state. By default, this function will add the two states together.
        reduce_fn: Defines the ReduceOp performed on the metric state if it is a tensor and we are in a distributed setting.
        compute_fn: Maps the metric state to a loggable or plottable value. By default, this function will return the state as is.
        log_every_step: Defines whether the metric should be logged every step or if logging will need to be called manually for the instance.
        log_type: Defines how the metric should be logged.
        reset_rule: Defines when the metric should be set to the intial state. If no reset is desired, set to MetricResetRule.MANUAL.
    """

    state: Tensor | int | float | list | None = None
    metric_type: MetricType = MetricType.STATE
    log_every_step: bool = True
    log_type: MetricLogType = MetricLogType.SCALAR
    reset_rule: MetricResetRule = MetricResetRule.ON_LOG
    transform_fn: Callable[[InferenceArtefacts], MetricState] = return_first_value
    accum_fn: Callable[[MetricState, MetricState], MetricState] = add
    reduce_fn: Callable[[Tensor | list, int], Tensor | list] = all_reduce_mean
    compute_fn: Callable[[MetricState], Any] = identity
    device: str | int = "cpu"
    world_size: int = 1
    is_distributed: bool = False

    def __post_init__(self):
        if isinstance(self.state, Tensor):
            self.state = self.state.to(self.device)
        self._default_state = deepcopy(self.state)

    @property
    def value(self) -> Any:
        assert self.state is not None
        if isinstance(self.state, Tensor) and self.is_distributed:
            self.state = self.reduce_fn(self.state, torch.distributed.get_world_size())
        return self.compute_fn(self.state)

    def update(self, inference_artefacts: InferenceArtefacts) -> None:
        if self.state is None:
            self.state = self.transform_fn(inference_artefacts)
        else:
            self.state = self.accum_fn(
                self.state,
                self.transform_fn(inference_artefacts),
            )

    def log(self, key: str) -> None:
        value = self.value
        if self.device in (0, "cuda", "cuda:0", "cpu", "mps"):
            if isinstance(value, Tensor):
                value = value.item()
            wandb.log(
                {
                    key.replace("_", "/"): (
                        plot.line(**value) if isinstance(value, dict) else value
                    )
                },
                commit=False,
            )
        if self.reset_rule == MetricResetRule.ON_LOG:
            self.reset()

    def reset(self) -> None:
        self.state = deepcopy(self._default_state)

    def clone(self, device: str | int, world_size: int) -> "Metric":
        cln = deepcopy(self)
        cln.device = device
        cln.world_size = world_size
        cln.is_distributed = world_size > 1
        cln.__post_init__()
        return cln


class MetricKey(Enum):
    TRAIN_LOSS = "train_loss"
    TRAIN_GRADNORM = "train_gradnorm"
    MICROSTEP = "microstep"
    STEP = "step"
    EPOCHSTEP = "epochstep"
    EPOCHMICROSTEP = "epochmicrostep"
    LR = "lr"
    EPOCH = "epoch"
    VAL_LOSS = "val_loss"
    VAL_ACCURACY = "val_accuracy"
    VAL_GENERATIONS = "val_generations"


METRICS = {
    MetricKey.TRAIN_LOSS: Metric(
        torch.tensor([0.0]),
    ),
    MetricKey.TRAIN_GRADNORM: Metric(
        tensor([0.0]),
    ),
    MetricKey.MICROSTEP: Metric(
        1,
        reset_rule=MetricResetRule.MANUAL,
    ),
    MetricKey.STEP: Metric(
        1,
        reset_rule=MetricResetRule.MANUAL,
    ),
    MetricKey.EPOCHSTEP: Metric(
        1,
        transform_fn=position_in_epoch,
        accum_fn=replace,
        reset_rule=MetricResetRule.MANUAL,
    ),
    MetricKey.EPOCHMICROSTEP: Metric(
        1,
        transform_fn=position_in_epoch,
        accum_fn=replace,
        reset_rule=MetricResetRule.MANUAL,
    ),
    MetricKey.LR: Metric(
        0,
        accum_fn=replace,
        reset_rule=MetricResetRule.MANUAL,
    ),
    MetricKey.EPOCH: Metric(
        1,
        reset_rule=MetricResetRule.MANUAL,
    ),
    MetricKey.VAL_LOSS: Metric(
        tensor([0.0]),
        log_every_step=False,
    ),
    MetricKey.VAL_ACCURACY: Metric(
        None,
        MetricType.GENERATION,
        False,
        transform_fn=identity,
        reduce_fn=all_gather_append,
        compute_fn=get_accuracy,
    ),
    MetricKey.VAL_GENERATIONS: Metric(
        None,
        MetricType.GENERATION,
        False,
        transform_fn=identity,
        reduce_fn=all_gather_append,
        compute_fn=construct_table,
    ),
}


def get_metrics(
    keys: list[MetricKey], device: str | int, world_size: int
) -> dict[str, Metric]:
    return {key.value: METRICS[key].clone(device, world_size) for key in keys}
