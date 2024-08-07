from dataclasses import dataclass, asdict, fields
import itertools
from typing import Callable, List, Tuple
import torch


@dataclass
class Params:
    def asdict(self):
        return asdict(self)


@dataclass
class HyperParams(Params):
    device: torch.device
    dtype: torch.dtype
    distribution: Callable


@dataclass
class LoopParams:
    def __iter__(self):
        return self.permutations()

    def permutations(self):
        members = fields(self)
        field_values = [getattr(self, field.name) for field in members]
        return itertools.product(*field_values)


@dataclass
class HyperParamLoop(LoopParams):
    device: List[torch.device]
    dtype: List[torch.dtype]
    distribution: List[Callable]


def initialise_weights(module: torch.nn.Module, weight_dist: Callable):
    weights_and_biases = set(
        [
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "nn.ConvTranspose2d",
            "nn.ConvTranspose3d",
        ]
    )  # TODO: check if there is a better way to do this
    if module.__class__.__name__ in weights_and_biases:
        weight_dist(module.weight)  # weight initialisation
        if module.bias is not None:
            weight_dist(module.bias)  # bias initialisation
