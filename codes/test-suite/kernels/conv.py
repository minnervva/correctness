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
class ConvHyperParams(HyperParams):
    in_channels: int
    out_channels: int
    kernel_size: Tuple
    stride: int
    padding: int
    dilation: int
    groups: int


@dataclass
class ConvLoop(HyperParamLoop):
    in_channels: List[int]
    out_channels: List[int]
    kernel_size: List[Tuple]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    groups: List[int]


@dataclass
class BatchDim(Params):
    batch_size: int
    dim: Tuple


@dataclass
class BatchDimLoop(LoopParams):
    batch_size: List
    dim: List


def convolution_loop(nnmodule, conv_loop, data_loop):
    for params in conv_loop:
        # setup the model
        model_params = ConvHyperParams(*params)
        make_model = model_params.asdict()
        make_model.pop("distribution")
        try:
            model = nnmodule(**make_model)
        except:
            continue
        initialise_weights(model, model_params.distribution)
        model = model.to(model_params.dtype)
        model = model.to(model_params.device)
        for data_dims in data_loop:
            data_params = BatchDim(*data_dims)
            dims = [data_params.batch_size, model_params.in_channels, *data_params.dim]
            input = torch.randn(dims, dtype=model_params.dtype)
            input = input.to(model_params.device)
            yield model, input, model_params, data_params
