import torch
from dataclasses import dataclass, asdict
from kernels.utils import (
    HyperParamLoop,
    HyperParams,
    LoopParams,
    Params,
    initialise_weights,
)
from typing import List, Tuple


@dataclass
class AvgPoolHyperParams(HyperParams):
    kernel_size: Tuple
    stride: int
    padding: int
    ceil_mode: bool
    count_include_pad: bool


@dataclass
class AvgPoolLoop(HyperParamLoop):
    kernel_size: List[Tuple]
    stride: List[int]
    padding: List[int]
    ceil_mode: List[bool]
    count_include_pad: List[bool]


@dataclass
class BatchDim(Params):
    batch_size: int
    dim: Tuple


@dataclass
class BatchDimLoop(LoopParams):
    batch_size: List
    dim: List


def avg_pool_loop(nnmodule, avg_pool_loop, data_loop):
    for params in avg_pool_loop:
        # setup the model
        model_params = AvgPoolHyperParams(*params)
        make_model = model_params.asdict()
        make_model.pop("device")
        make_model.pop("dtype")
        make_model.pop("distribution")
        model = nnmodule(**make_model)
        initialise_weights(model, model_params.distribution)
        model = model.to(model_params.dtype)
        model = model.to(model_params.device)
        for data_dims in data_loop:
            data_params = BatchDim(*data_dims)
            dims = [data_params.batch_size, *data_params.dim]
            input = torch.randn(dims, dtype=model_params.dtype)
            input = input.to(model_params.device)
            yield model, input, model_params, data_params
