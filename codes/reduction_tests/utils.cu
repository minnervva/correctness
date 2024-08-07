#ifdef REDUCE_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#endif

#ifdef REDUCE_USE_HIP
#include <hiprand.h>
#endif

#include "utils.hpp"
#include <vector>
#include <whip.hpp>

#define TPB 64
typedef long long int_64;
/* Ada discussion : Feb. 26th 2024
 * - purely random neighbor distribution up to 500 neighbors
 * - scale random variables by 10 (whatever value)
 * - H20-128 coordinates with PBC radius 10 A
 * - print individual components
 */

__global__ void uniform_double_gpu(const int number_of_elements__,
                                   const double xmin, const double xmax,
                                   double *rng) {
  if (threadIdx.x + blockIdx.x * blockDim.x >= number_of_elements__)
    return;
  rng[threadIdx.x + blockIdx.x * blockDim.x] =
      (xmax - xmin) * (rng[threadIdx.x + blockIdx.x * blockDim.x] - 0.5);
}

void uniform_double(const int number_of_elements__, const double xmin,
                    const double xmax, double *rng) {
  uniform_double_gpu<<<(number_of_elements__ + 128 - 1) / 128, 128>>>(
      number_of_elements__, xmin, xmax, rng);
}

__global__ void scale_the_darn_thing_gpu(int *neighbor_list__,
                                         const int number_of_elements__,
                                         const int number_of_atoms__) {
  if (threadIdx.x + blockIdx.x * blockDim.x >= number_of_elements__)
    return;

  neighbor_list__[threadIdx.x + blockIdx.x * blockDim.x] =
      neighbor_list__[threadIdx.x + blockIdx.x * blockDim.x] %
      number_of_atoms__;
}

void scale_the_darn_thing(int *neighbor_list__, const int number_of_elements__,
                          const int number_of_atoms__) {
  dim3 block_grid((number_of_elements__ + 128 - 1) / 128);
  dim3 thread_grid(128);

  scale_the_darn_thing_gpu<<<block_grid, thread_grid>>>(
      neighbor_list__, number_of_elements__, number_of_atoms__);
}

template <typename T>
__global__ void center_and_scale_distribution_gpu(T *data__, const T scal__,
                                                  const T center__,
                                                  const int size__) {
  const int id = threadIdx.x + blockDim.x * blockIdx.x;
  if (id >= size__)
    return;
  data__[id] = scal__ * (data__[id] - center__);
}

template <typename T>
__host__ void center_and_scale_distribution(T *data__, const T scal__,
                                            const T center__,
                                            const size_t size__) {
  int grid_size = (size__ + 128 - 1) / 128;
  center_and_scale_distribution_gpu<<<grid_size, 64>>>(data__, scal__, center__,
                                                       size__);
}

void generate_random_numbers(std::string distribution__, generator_t gen__,
                             const double amplitude__,
                             const double distribution_center__,
                             const double standard_deviation__, double *data__,
                             const size_t length__) {
  if (distribution__ == "uniform") {
    GenerateUniformDouble(gen__, data__, length__);
    center_and_scale_distribution<double>(data__, amplitude__,
                                          distribution_center__, length__);
  }

  if (distribution__ == "normal") {
    GenerateNormalDouble(gen__, data__, length__, distribution_center__,
                         standard_deviation__);
  }
}
