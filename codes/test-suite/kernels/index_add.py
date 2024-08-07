import torch
from dataclasses import dataclass, asdict
from kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple


@dataclass
class IndexAddHyperParams(HyperParams):
    dim: int


@dataclass
class IndexAddLoop(HyperParamLoop):
    dim: List[int]


@dataclass
class IndexAddDim(Params):
    input_dim: Tuple
    reduction_ratio: float


@dataclass
class IndexAddDimLoop(LoopParams):
    input_dim: List[Tuple]
    reduction_ratio: List[float]


def index_add_loop(func_name: str, index_add_loop, data_loop):
    for params in index_add_loop:
        index_add_params = IndexAddHyperParams(*params)

        for d_params in data_loop:
            dim_params = IndexAddDim(*d_params)
            index = (
                torch.randint(
                    low=0,
                    high=int(dim_params.input_dim[0] * dim_params.reduction_ratio),
                    size=(int(dim_params.input_dim[0] * dim_params.reduction_ratio),),
                )
                .to(torch.int64)
                .to(index_add_params.device)
            )
            input = (
                index_add_params.distribution(torch.zeros(dim_params.input_dim))
                .to(index_add_params.dtype)
                .to(index_add_params.device)
            )

            source = (
                index_add_params.distribution(
                    torch.zeros(
                        int(dim_params.input_dim[0] * dim_params.reduction_ratio),
                        dim_params.input_dim[1],
                    )
                )
                .to(index_add_params.dtype)
                .to(index_add_params.device)
            )

            if index_add_params.dim >= len(dim_params.input_dim):
                break
            yield getattr(input, func_name), {
                "source": source,
                "dim": index_add_params.dim,
                "index": index,
            }, index_add_params, dim_params
