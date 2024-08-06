# Introduction

This repository contains additional information as well as the codes used for the paper titled "Floating-point non-associativity and non-reproducibility within asynchronous parallel programming". 

# Parallel sum
 
## code

The c++ code used to measure the scalar variability induced by atomic operations
can be found in the code directory. The compilation only requires CMake, CUDA or
HIP. To compile it, clone the repository and use the command

```[bash]
cmake -B build -S . -DREDUCE_USE_CUDA -DCMAKE_CUDA_ARCHITECTURES=70 
```
for support of the V100 or 
```[bash]
cmake -B build -S . -DREDUCE_USE_HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a+xnacks 
```

the AMD Mi250X GPU. The variable `CMAKE_CUDA_ARCHITECTURES` or
`CMAKE_HIP_ARCHITECTURES` can be set to different GPU families (80 for the A100,
90 for the H100 and GH200 GPUs).

Double precision atomic operations are supported on the Mi250X GPU but require
an additional compiler flag enabling unsafe math optimizations. The `make`
command will generate one executable named `test_reduce`. the executable has
several options controlling the calculations parameters. `test_reduce --help`
will return the full list of all available options.

We run the following commands for the article 
```
./test_reduce -S 1000000 --max_reduction_size 1000000 -A 10.0 -d uniform --atomic_only
./test_reduce -S 1000000 --max_reduction_size 1000000 -A 10.0 -d normal
```

The executable generates some csv files containing results of the variability
for the different distributions. The name of the distribution and the GPU types
are included in the data file name. The format of the files containing the
variability data is given by

```
10,0.0000000000000000e+00,0.0000000000000000e+00,-0.0000000000000000e+00,,...
100,0.0000000000000000e+00,0.0000000000000000e+00,-0.0000000000000000e+00,...
```

Each line starts with the number of floating elements to be summed followed by
the measured variability.

The file containing the timings data is also a text file, each line giving the
timings for different values of the parameters and name of the kernels. The
format of each line is given by

```
implementation,block size,grid size,timing mean value,timing standard deviation
```

For instance the following line gives the parameters and timings for the `atomic_only` implementation of the sum. 

```
atomic_only,512,32768,0.872566850400000,0.000070762366890
```

The Mathematica directory contains mathematica files to explore the variability
data and generate the figures included in this material and the paper. The data
used for the article can be found in the data directory.

## Simulating the behavior of atomic operations

Atomic operations on GPU provide a mean for multiple threads to update the same
local area in memory. Each thread will issue the atomic operation which will be
serialized and executed in a un specified order by a dedicated unit of the
memory controller. It is a major problem because floating point arithmetic is
non associative. Summing three floating point numbers a, b, c with two atomicAdd
can return any the three outcomes `(a + b) + c` or `(a + c) + b` or `(b + c) +
a` which are potentially different due to rounding errors. 

Applying an atomic operation on a array of data is equivalent to permuting the
elements of the array before applying the deterministic version of the
operation. It is better illustrated with the following python code that computes
the sum of array before and after randomly permuting the elements.

```[python]
 import numpy as np
 from numpy.random import MT19937, RandomState, SeedSequence
 rs = RandomState(MT19937(SeedSequence(123456789)))
 length = 100
 for i in range(1, 6):
     x = rs.standard_normal(length)
     sum_x = np.sum(x)
     for i in range(1,10):
         res = sum_x - np.sum(np.random.permutation(x))
         Vs= res / sum_x
         print(f"Size {length}: {res} {Vs}")
     length *= 10
```

Running this code on a laptop will return this output
```
Size 100: 0.0 -0.0
Size 100: 3.552713678800501e-15 -2.8582217558337204e-16
Size 100: 1.7763568394002505e-15 -1.4291108779168602e-16
Size 100: 1.7763568394002505e-15 -1.4291108779168602e-16
Size 100: -1.7763568394002505e-15 1.4291108779168602e-16
Size 100: 1.7763568394002505e-15 -1.4291108779168602e-16
Size 100: 1.7763568394002505e-15 -1.4291108779168602e-16
Size 100: 1.7763568394002505e-15 -1.4291108779168602e-16
Size 100: 0.0 -0.0
Size 1000: -1.7763568394002505e-15 -3.7335207564099476e-16
Size 1000: -8.881784197001252e-15 -1.8667603782049737e-15
Size 1000: 0.0 0.0
Size 1000: -1.2434497875801753e-14 -2.613464529486963e-15
Size 1000: -1.7763568394002505e-15 -3.7335207564099476e-16
Size 1000: 5.329070518200751e-15 1.1200562269229843e-15
Size 1000: -7.105427357601002e-15 -1.493408302563979e-15
Size 1000: 3.552713678800501e-15 7.467041512819895e-16
Size 1000: 1.7763568394002505e-15 3.7335207564099476e-16
Size 10000: -1.4210854715202004e-14 5.165619713608847e-16
Size 10000: -1.7763568394002505e-14 6.457024642011058e-16
Size 10000: -2.4868995751603507e-14 9.039834498815481e-16
Size 10000: -1.7763568394002505e-14 6.457024642011058e-16
Size 10000: -2.1316282072803006e-14 7.74842957041327e-16
Size 10000: -1.4210854715202004e-14 5.165619713608847e-16
Size 10000: -1.7763568394002505e-14 6.457024642011058e-16
Size 10000: 3.552713678800501e-14 -1.2914049284022117e-15
Size 10000: 1.7763568394002505e-14 -6.457024642011058e-16
Size 100000: 1.7053025658242404e-13 4.238867544267839e-16
Size 100000: 1.1368683772161603e-13 2.825911696178559e-16
Size 100000: 0.0 0.0
Size 100000: 5.684341886080802e-14 1.4129558480892796e-16
Size 100000: 2.2737367544323206e-13 5.651823392357118e-16
Size 100000: 1.7053025658242404e-13 4.238867544267839e-16
Size 100000: 5.684341886080802e-14 1.4129558480892796e-16
Size 100000: 1.7053025658242404e-13 4.238867544267839e-16
Size 100000: 5.684341886080802e-14 1.4129558480892796e-16
Size 1000000: 2.5579538487363607e-13 1.394635689364491e-15
Size 1000000: 6.821210263296962e-13 3.719028504971976e-15
Size 1000000: 5.115907697472721e-13 2.789271378728982e-15
Size 1000000: 5.400124791776761e-13 2.944230899769481e-15
Size 1000000: 4.263256414560601e-13 2.324392815607485e-15
Size 1000000: 4.831690603168681e-13 2.634311857688483e-15
Size 1000000: 1.7053025658242404e-13 9.29757126242994e-16
Size 1000000: 5.115907697472721e-13 2.789271378728982e-15
Size 1000000: 6.821210263296962e-13 3.719028504971976e-15
```

The error grows with the number of floating points elements and can reach values
that are above the threshold of many correctness tests found in the quantum package
CP2K.

## Details about the different implementations

Due to space constraint, the main text only contains a general description of
the different implementations of the sum on GPU. The main issue with GPU
programming is the absence of explicit barrier synchronization when a kernel is
executed. The following code

```[c++]
__global__ void reduce_single_thread(double *sdata, int size, double *res) {
     if (blockIdx.x != 0)
         return;
        
     const int tid = threadIdx.x;
     double sum = 0.0;
     if (tid == 0)
         for (int i = 0; i < size; ++i)
             sum += sdata[i];
     res[0] = sum;
}
```

computes the sum using the recursive algorithm on one GPU thread. this
implementation is the simplest deterministic implementation from a programmatic
point of view but it is slow as only one thread is used to complete the sum. The
following code is also convenient to compute the sum but is not deterministic by
construction as the order in which the atomic operation is executed is runtime
dependent.

```[c++]
__global__ void reduce_atomic_only(double *sdata, int size, double *res) {
    if (blockIdx.x != 0)
        return;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(res, sdata[tid]);
}
```

This implementation is four order of magnitude slower than any non optimized
implementation of the pairwise method but it is still conveniently used in more
complex kernels.

The pairwise method is more appropriate for GPU. One half of the element of an
array is added in pair with the second half $t_i = x_{i} + x_{i + N / 2}$ an
operation that can be done in parallel on GPU. The process composed of log_2 (N)
steps is repeated recursively on the array $t_i$ to get the sum.

## Statistical properties of the non-deterministic summation

The following figures show the scalar variability calculated on the Mi250X vs V100 and GH200 vs V100 on the same x axis for a set of uniformly distributed floating point numbers taken between 0 and 10. We use (SPA)  and (SPS) for the non-deterministic and deterministic implementations of the sum. The runtime parameters are identical for both kernels.   

![mi250x_v100_scalar_variability](figures/Difference_distribution_mi250x_v100_uniform_distribution.pdf
"PDF of the scalar variability for the Mi250X and V100 GPU")
![gh200_v100_scalar_variability](figures/Difference_distribution_gh200_v100_uniform_distribution.pdf
"PDF of the scalar variability for the GH200 and V100 GPU")

The PDF of the scalar variability are almost identical between the GH200 and
V100 GPU while we can observe some difference between the pdf measured on the
Mi250X and V100 GPUs. Although the code uses the same random number generator
there is no guaranty to get the same random sequence on AMD and NVIDIA GPU even
with the same initial seed as the implementation of the RNG might differ between
the curand and hiprand libraries.

The PDF measured on all GPU converge to the normal distribution supporting the
assumption that the variability introduced by atomic operations can be treatd as
Gaussian noise. 

This result is not general though as the of the PDF for the variability
calculated with the (AO) method is a multimodal distribution. 

# performance comparison

## Parameters for the performance comparison

To measure the timings of the different implementations of the parallel sum
included in the `c++` code, we generate a set of 100 arrays of 4 M uniformly
distributed floating point numbers and compute the time required to calculate
the sum for all implementations for different values of the kernels parameters
$N_t$ and $N_B$. $N_t$ can only take values up to `1024`. The hardware
limitation for $N_b$ have no practical impact some these tests. We do not
optimize the code for each specific GPU either.

All implementations are tested for $N_t = 64, 128, 256, 512$. The number of
block N_b follows a geometric progression with a step of 4. It is automatically
limited to $(N + 2 N_t - 1) / 2 N_t$ when $N_b$ is above this value.

The timings are averaged over 10 runs to gather statistics. 

## results

The main text gives a short summary of the performance tests. According to our measurements, the (AO) method is the slowest method by many order of magnitude of all methods we tested. The fastest method is the non-deterministic (SPA) method on V100 and GH200 followed by the (SPS) method on V100 and the sum implementation found in the CUB library. The fastest method on Mi250X is the (TPRC) method followed by the sum function provided by the hipcub library. Most of the timings are within 5% from each others which is not enough to conclude if non deterministic implementations of the sum are faster than their deterministic counterpart. 

The results also show relativelly small variations of the timing for a large set of kernel parameters.  

