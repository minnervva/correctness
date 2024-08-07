import torch
from dataclasses import dataclass, asdict
from kernels.utils import HyperParams, HyperParamLoop, LoopParams, Params
from typing import List, Tuple


@dataclass
class GatherHyperParams(HyperParams):
    dim: int


@dataclass
class GatherLoop(HyperParamLoop):
    dim: List[int]


@dataclass
class GatherDim(Params):
    input_dim: Tuple
    reduction_ratio: float


@dataclass
class GatherDimLoop(LoopParams):
    input_dim: List[Tuple]
    reduction_ratio: List[float]


def gather_loop(func_name: str, gather_loop, data_loop):
    for params in gather_loop:
        gather_params = GatherHyperParams(*params)

        for d_params in data_loop:
            dim_params = GatherDim(*d_params)

            index = (
                torch.randint(
                    low=0,
                    high=int(dim_params.input_dim[0] * dim_params.reduction_ratio),
                    size=dim_params.input_dim,
                )
                .to(torch.int64)
                .to(gather_params.device)
            )
            input = (
                gather_params.distribution(torch.zeros(dim_params.input_dim))
                .to(gather_params.dtype)
                .to(gather_params.device)
            )
            if gather_params.dim >= len(dim_params.input_dim):
                continue
            yield torch.gather, {
                "input": input,
                "dim": gather_params.dim,
                "index": index,
            }, gather_params, dim_params
