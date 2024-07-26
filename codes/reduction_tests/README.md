# Building the code

This readme contains information about compiling and running the reduction tests either locally or on clusters such as frontier or summit. The program depends on the CUDA/HIP SDK, `cxxopts`, and the `fmt` library. The `cxxopts` and the `fmt` libraries are automatically compiled by `cmake`. The build system requires cmake 3.24 or above to get full support of HIP and CUDA. It also support `scorep` when power monitoring is required.

To build the code simply enter the root directory and enter

```
mkdir build
cd build
cmake -DREDUCE_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 ..
make
```
to compile the code for NVIDIA GPUs. the variable `CMAKE_CUDA_ARCHITECTURES` should be set to the specific GPU target (V100 (`-DCMAKE_CUDA_ARCHITECTURES=70`), A100 (`-DCMAKE_CUDA_ARCHITECTURES=80`), GH200 (`-DCMAKE_CUDA_ARCHITECTURES=90`)). Nothing else is required as cmake will automatically detect the CUDA SDK.

To compile the code with AMD GPU support, replace the camke command with
```
cmake -DREDUCE_USE_HIP=ON ..
```
The architecture is automatically selected when the code is compiled on frontier (Mi250X (gfx90a) i think). `CMake` will configure the additional options automatically for the hardware atomicAdd support.

## Adding power measurements with scorep

Configuring and compiling the code with power measurement can activated with the following command

```
cmake -DREDUCE_USE_CUDA=ON -DREDUCE_SCOREP=ON ..
make
```

# Description of the different tests

The program runs three tests. The first test is a reproducibility test that computes the sum of 100 random sequences of 1 million elements. Two variants of the algorithm are tested, the shared memory and shfl_down variants of the tree reduction algorithm. The only difference between the implementation is how the reduction is executed on the last warp. HIP below 6.3 does not seem to support the shuffle down implementation so only the shared memory variant is used in that case. We compare the results to the reference values obtained on V100. The tests pass if the results of both variants of the sum is bitwise identical to the reference values.

The second test computes the relative error between the reproducible sum and its atomic variant for different random sequences. We generate 100 random sequences of say 1 million elements and repeat the summation multiple time over to get the distribution of the relative error for various lengths. Results are stored in the file `test2.csv` for further processing.

The last test measures the time to run different implementations of the sum for various thread block and grid sizes and print the results on the screen. The results are also saved in the text file 'timings.dat'. We also test the CUB/hipcub variants that are supposed to be optimized for each architecture.

* Options controlling the program

The user can control the program execution with the following options

- `max_reduction_size` and `min_reduction_size`: define the maximum and minimum number of floating point elements to be summed. The tests will aply the summation starting from `min_reduction_size` to `max_reduction_size` by increasing the number of elements by a factor ten after each round.
- `number_of_samples`: define the sample size of the relative error.
- `distribution_type` : `normal` or `uniform` distributions. We can keep the default values for power measurement. It only impact the second test results

