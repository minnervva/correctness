import torch
from dataclasses import dataclass, asdict
from kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple


@dataclass
class IndexCopyHyperParams(HyperParams):
    dim: int


@dataclass
class IndexCopyLoop(HyperParamLoop):
    dim: List[int]


@dataclass
class IndexCopyDim(Params):
    input_dim: Tuple
    reduction_ratio: float


@dataclass
class IndexCopyDimLoop(LoopParams):
    input_dim: List[Tuple]
    reduction_ratio: List[float]


def index_copy_loop(func_name: str, index_copy_loop, data_loop):
    for params in index_copy_loop:
        index_copy_params = IndexCopyHyperParams(*params)

        for d_params in data_loop:
            dim_params = IndexCopyDim(*d_params)
            index = (
                torch.randint(
                    low=0,
                    high=int(dim_params.input_dim[0] * dim_params.reduction_ratio),
                    size=(int(dim_params.input_dim[0] * dim_params.reduction_ratio),),
                )
                .to(torch.int64)
                .to(index_copy_params.device)
            )
            input = (
                index_copy_params.distribution(torch.zeros(dim_params.input_dim))
                .to(index_copy_params.dtype)
                .to(index_copy_params.device)
            )

            source = (
                index_copy_params.distribution(
                    torch.zeros(
                        int(dim_params.input_dim[0] * dim_params.reduction_ratio),
                        dim_params.input_dim[1],
                    )
                )
                .to(index_copy_params.dtype)
                .to(index_copy_params.device)
            )

            if index_copy_params.dim >= len(dim_params.input_dim):
                break
            yield getattr(input, func_name), {
                "source": source,
                "dim": index_copy_params.dim,
                "index": index,
            }, index_copy_params, dim_params
