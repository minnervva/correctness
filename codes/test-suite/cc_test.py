import torch
import pandas as pd
import time
import numpy as np
import os

from kernels.benchmark import *
from kernels.avg_pool import *
from kernels.conv import *
from kernels.gather import *
from kernels.scatter import *
from kernels.scatter_reduce import *
from kernels.replication_pad import *
from kernels.index_add import *
from kernels.index_copy import *
from kernels.index_put import *
from kernels.utils import *

cpu = torch.device("cpu")
gpu = torch.device("cuda")


if __name__ == "__main__":
    niterations = 100

    with torch.no_grad():
        avg_pool_params = AvgPoolLoop(
            kernel_size=[(3, 3, 3), (5, 5, 5), (7, 7, 7)],
            stride=[1, 3, 5],
            padding=[0, 1],
            ceil_mode=[True, False],
            count_include_pad=[True, False],
            device=[gpu],
            dtype=[torch.float32],
            distribution=[torch.nn.init.normal_],
        )
        avg_pool_dims = BatchDimLoop(
            batch_size=[1, 3],
            dim=[
                (1, 16, 16, 16),
                (3, 64, 64, 64),
            ],
        )
        nn_benchmark(
            avg_pool_params,
            avg_pool_dims,
            avg_pool_loop,
            torch.nn.AvgPool3d,
            niterations,
        )

        # 1d convolutions
        conv_params = ConvLoop(
            in_channels=[3, 7],
            out_channels=[3, 7],
            kernel_size=[(3,), (5,), (9,)],
            stride=[1, 3, 5],
            padding=[0, 1],
            dilation=[1, 2, 4],
            groups=[1, 3],
            device=[gpu],
            dtype=[torch.float32],
            distribution=[torch.nn.init.normal_],
        )
        data_loop = BatchDimLoop(
            batch_size=[1, 3],
            dim=[(100,), (1000,), (10000,)],
        )

        nn_benchmark(
            conv_params, data_loop, convolution_loop, torch.nn.Conv1d, niterations
        )
        nn_benchmark(
            conv_params,
            data_loop,
            convolution_loop,
            torch.nn.ConvTranspose1d,
            niterations,
        )
        # 2d convolutions
        conv_params = ConvLoop(
            in_channels=[3, 7],
            out_channels=[3, 7],
            kernel_size=[
                (3, 3),
                (5, 5),
                (9, 9),
            ],
            stride=[1, 3, 5],
            padding=[0, 1],
            dilation=[1, 2, 4],
            groups=[1, 3],
            device=[gpu],
            dtype=[torch.float32],
            distribution=[torch.nn.init.normal_],
        )
        data_loop = BatchDimLoop(
            batch_size=[1, 3],
            dim=[(100, 100)],
        )

        nn_benchmark(
            conv_params, data_loop, convolution_loop, torch.nn.Conv2d, niterations
        )
        nn_benchmark(
            conv_params,
            data_loop,
            convolution_loop,
            torch.nn.ConvTranspose2d,
            niterations,
        )
        conv_loop = ConvLoop(
            in_channels=[3],
            out_channels=[3],
            kernel_size=[
                (3, 3, 3),
                (5, 5, 5),
            ],
            stride=[1, 3, 5],
            padding=[0, 1],
            dilation=[1, 2, 4],
            groups=[1, 3],
            device=[gpu],
            dtype=[torch.float32],
            distribution=[torch.nn.init.normal_],
        )
        data_loop = BatchDimLoop(
            batch_size=[1, 3],
            dim=[(100, 100, 100), (200, 200, 200)],
        )

        # nn_benchmark(
        #    conv_loop, data_loop, convolution_loop, torch.nn.Conv3d, niterations
        # )
        # nn_benchmark(
        #    conv_loop,
        #    data_loop,
        #    convolution_loop,
        #    torch.nn.ConvTranspose3d,
        #    niterations,
        # )

        replication_pad_loop = ReplicationPadLoop(
            pad=[0],
            device=[gpu],
            dtype=[torch.float32],
            distribution=[torch.nn.init.normal_],
        )
        data_loop = BatchDimLoop(
            batch_size=[1, 3],
            dim=[(100, 1_000)],
        )

    scatter_params = ScatterLoop(
        dim=[0],
        reduce=["add", "multiply"],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    scatter_dims = ScatterDimLoop(
        input_dim=[
            (100,),
            (500,),
            (1_000,),
            (10_000,),
            (100, 100),
            (500, 500),
            (1_000, 1_000),
        ],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(scatter_params, scatter_dims, scatter_loop, "scatter", niterations)

    scatter_params = ScatterReduceLoop(
        dim=[0],
        reduce=["sum", "mean"],
        include_self=[True, False],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    scatter_dims = ScatterReduceDimLoop(
        input_dim=[
            (100,),
            (500,),
            (1_000,),
            (10_000,),
            (100, 100),
            (500, 500),
            (1_000, 1_000),
        ],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(
        scatter_params, scatter_dims, scatter_reduce_loop, "scatter_reduce", niterations
    )

    gather_params = GatherLoop(
        dim=[0],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    gather_dims = GatherDimLoop(
        input_dim=[
            (100,),
            (1_000,),
        ],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    func_benchmark(gather_params, gather_dims, gather_loop, "gather", niterations)

    index_add_params = IndexAddLoop(
        dim=[0],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )

    index_add_dims = IndexAddDimLoop(
        input_dim=[(100, 100), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(
        index_add_params, index_add_dims, index_add_loop, "index_add", niterations
    )

    index_copy_params = IndexAddLoop(
        dim=[0],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )

    index_copy_dims = IndexAddDimLoop(
        input_dim=[(100, 100), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(
        index_copy_params, index_copy_dims, index_copy_loop, "index_copy", niterations
    )

    index_put_params = IndexPutLoop(
        accumulate=[True, False],
        device=[gpu],
        dtype=[torch.float32],
        distribution=[torch.nn.init.normal_],
    )
    index_put_dims = IndexPutDimLoop(
        input_dim=[(100, 100), (1_000, 1_000)],
        reduction_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    func_benchmark(
        index_put_params, index_put_dims, index_put_loop, "index_put", niterations
    )
