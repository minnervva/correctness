import torch
from dataclasses import dataclass, asdict
from typing import List, Tuple

from kernels.utils import (
    HyperParamLoop,
    HyperParams,
    LoopParams,
    Params,
    initialise_weights,
)


@dataclass
class ReplicationPadHyperParams(HyperParams):
    pad: int


@dataclass
class ReplicationPadLoop(HyperParamLoop):
    pad: List[int]


@dataclass
class BatchDim(Params):
    batch_size: int
    dim: Tuple


@dataclass
class BatchDimLoop(LoopParams):
    batch_size: List
    dim: List


def replication_pad_loop(nnmodule, replication_pad_loop, data_loop):
    for params in replication_pad_loop:
        # setup the model
        model_params = ReplicationPadHyperParams(*params)
        model = nnmodule(**asdict(model_params))
        initialise_weights(model, model_params.distribution)
        model = model.to(model_params.dtype)
        model = model.to(model_params.device)
        for data_dims in data_loop:
            data_params = BatchDim(*data_dims)
            dims = [data_params.batch_size, model_params.in_channels, *data_params.dim]
            input = torch.randn(dims, dtype=model_params.dtype)
            input = input.to(model_params.device)
            yield model, input, model_params, data_params
