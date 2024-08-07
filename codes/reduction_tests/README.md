4# Introduction

This repository contains a code written to experiment with the reduce function using various implementations on GPU. We check principally determinism of the implementation on various GPU family and the runtime behavior depending on the GPU kernel parameters. The code also include a simplified version of the reduce function based on the CUB/HIPCUB libraries. They all use the the pairwise reduction algorithm and are supposed to be optimizied for each platform. 

This code is highly and although the implementation is reliable, we strongly suggest the reader to use the CUB/HIPCUB as a default implementation for simplicity. The only advantage that this implementation has is more control over the kernels parameters. 

This implementation contains different methods to the reduction since our primary is to study the impact of atomic operations on the final result. The GPU code runs first a block reduction based on the pairwise algorithm and then either use atomicAdd, offload the last stage to the CPU or compute the final reduction locally. All methods illustrate several programming techniques with their pro and con. There is nothing new here as most of this can be found in the CUDA examples for instance. 

The block reduction is based on Mark Harris (NVIDIA) presentation.

# compilation
Compiling the code requires cmake > 3.24. All dependencies are automagically handled by cmake excepted the compiler and CUDA or ROCM. We tested this code on both CUDA 12.4 and ROCM 6.0.0 but earlier versions should work as well. The code is generic enough so that specialization only really happens in one function. 

To compile with cuda with V100 support

```[shell]
mkdir build-cuda
cd build-cuda
cmake -DREDUCE_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 .. 
make
```

To compile with HIP support

```[shell]
mkdir build-hip
cd build-hip
cmake -DREDUCE_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a+xsnack" ..
make
```


