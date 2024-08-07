import torch
from dataclasses import dataclass, asdict
from kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple


@dataclass
class ScatterHyperParams(HyperParams):
    dim: int
    reduce: str


@dataclass
class ScatterLoop(HyperParamLoop):
    dim: List[int]
    reduce: List[str]


@dataclass
class ScatterDim(Params):
    input_dim: Tuple
    reduction_ratio: float


@dataclass
class ScatterDimLoop(LoopParams):
    input_dim: List[Tuple]
    reduction_ratio: List[float]


def scatter_loop(func_name: str, scatter_loop, data_loop):
    for params in scatter_loop:
        scatter_params = ScatterHyperParams(*params)

        for d_params in data_loop:
            dim_params = ScatterDim(*d_params)

            src = (
                scatter_params.distribution(torch.zeros(dim_params.input_dim))
                .to(scatter_params.dtype)
                .to(scatter_params.device)
            )
            reduced_dim = tuple(
                [
                    int(dim_params.input_dim[i] * dim_params.reduction_ratio)
                    for i in range(len(dim_params.input_dim))
                ]
            )

            index = (
                torch.randint(low=0, high=reduced_dim[0], size=reduced_dim)
                .to(torch.int64)
                .to(scatter_params.device)
            )
            input = (
                scatter_params.distribution(torch.zeros(reduced_dim))
                .to(scatter_params.dtype)
                .to(scatter_params.device)
            )
            if scatter_params.dim >= len(dim_params.input_dim):
                break
            yield getattr(input, func_name), {
                "dim": scatter_params.dim,
                "index": index,
                "src": src,
                "reduce": scatter_params.reduce,
            }, scatter_params, dim_params
