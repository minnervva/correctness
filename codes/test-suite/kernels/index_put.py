import torch
from dataclasses import dataclass, asdict
from kernels.utils import HyperParamLoop, HyperParams, LoopParams, Params
from typing import List, Tuple


@dataclass
class IndexPutHyperParams(HyperParams):
    accumulate: bool


@dataclass
class IndexPutLoop(HyperParamLoop):
    accumulate: List[bool]


@dataclass
class IndexPutDim(Params):
    input_dim: Tuple
    reduction_ratio: float


@dataclass
class IndexPutDimLoop(LoopParams):
    input_dim: List[Tuple]
    reduction_ratio: List[float]


def index_put_loop(func_name: str, index_put_loop, data_loop):
    for params in index_put_loop:
        index_put_params = IndexPutHyperParams(*params)

        for d_params in data_loop:
            dim_params = IndexPutDim(*d_params)
            idim = dim_params.input_dim
            rr = dim_params.reduction_ratio
            input = (
                index_put_params.distribution(torch.zeros(idim))
                .to(index_put_params.dtype)
                .to(index_put_params.device)
            )
            indices = tuple(
                (
                    torch.randint(low=0, high=int(idim[0] * rr), size=(1, idim[0]))
                    .to(torch.int64)
                    .to(index_put_params.device),
                    torch.randint(low=0, high=int(idim[0] * rr), size=(1, idim[0]))
                    .to(torch.int64)
                    .to(index_put_params.device),
                )
            )
            values = (
                index_put_params.distribution(torch.zeros(int(idim[0])))
                .to(index_put_params.dtype)
                .to(index_put_params.device)
            )

            yield getattr(input, func_name), {
                "indices": indices,
                "values": values,
                "accumulate": index_put_params.accumulate,
            }, index_put_params, dim_params
