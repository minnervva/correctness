#include <vector>
#include <whip.hpp>

#include "utils.hpp"

#ifdef REDUCE_USE_CUDA
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
#endif
/* ---------------------------------------------------------------------------
 */
/* Taken from the cuda samples sdk */
/* ----------------------------------------------------------------------------*/

template <typename T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <> struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T, int blockSize, warp_reduction_method warp_method__>
__device__ __inline__ void warpReduce(T *sdata, T &mySum) {
#ifdef REDUCE_USE_CUDA
  if (warp_method__ == warp_reduction_method::shuffle_down) { // shuffle
    if (threadIdx.x < warpSize) {
      for (int offset = warpSize / 2; offset > 0; offset /= 2)
        mySum += __shfl_down_sync(0xffffffff, mySum, offset);
    }
    return;
  } else {
    // fully unroll reduction within a single warp
    if (threadIdx.x < 16)
      sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 16];
    __syncthreads();
    if (threadIdx.x < 8)
      sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 8];
    __syncthreads();
    if (threadIdx.x < 4)
      sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 4];
    __syncthreads();
    if (threadIdx.x < 2)
      sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 2];
    __syncthreads();
    if (threadIdx.x < 1)
      mySum = mySum + sdata[threadIdx.x + 1];
    __syncthreads();
  }
#else
  // hip only supports __shfl_down_sync starting from ROCM 6.3
  // so we stick shared memory reduction on AMD hardware for the time being
  // __syncwarp() is not supported by hip.
  // fully unroll reduction within a single warp
  if (threadIdx.x < 16)
    sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8)
    sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4)
    sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2)
    sdata[threadIdx.x] = mySum = mySum + sdata[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1)
    mySum = mySum + sdata[threadIdx.x + 1];

#endif
  return;
}
/* it can also be done with shuffle and cooperative groups. cooperative groups
   is supported on AMD with ROCM 6 I think. Does not offer anything special
   except compacity.

   The block reduction can also be implemented with the warpReduction only but
   it will change the order of the operations and we will have to be careful
   with the warp size difference between NVIDIA and AMD GPU if we want a cross
   platform code.


*/

template <class T, int blockSize, warp_reduction_method warp_method>
__device__ __inline__ void blockReduce(T *sdata, T &mySum,
                                       const unsigned int tid) {
  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata[tid] = mySum = mySum + sdata[tid + 32];
  }
  __syncthreads();

  // apply warp reduction for the first warp of the thread group.
  warpReduce<T, blockSize, warp_method>(sdata, mySum);
  __syncthreads();
}
/*
  All the code is taken From Mark Harris (NVIDIA) original work on
  reduction. It is probably a bit outdated nowadays as the block reduction
  can be done with specific hardware instructions. This code
  is generic and will work equally well on AMD and NVIDIA GPUs

  This version adds multiple elements per thread sequentially.  This reduces
  the overall cost of the algorithm while keeping the work complexity O(n) and
  the step complexity O(log n). (Brent's Theorem optimization)

  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
  In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
  If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/

/* ----------------------------------------------------------------------------------
 */

template <typename T, unsigned int blockSize,
          last_stage_summation_algorithm last_stage,
          warp_reduction_method warp_method__, algorithm_type algo__>
__global__ void reduce_gpu(const T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  // use the end of scratch memory to store how many block are already done.

  // Really error prone. The best is to have a dedicated argument for it
  unsigned int *retirementCount = (unsigned int *)(g_odata + gridDim.x);

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  const bool single_round_ =
      (last_stage == last_stage_summation_algorithm::pairwise_method) ||
      (last_stage == last_stage_summation_algorithm::recursive_method) ||
      (last_stage == last_stage_summation_algorithm::kahan_method);

  blockReduce<T, blockSize, warp_method__>(sdata, mySum, tid);
  if (algo__ == algorithm_type::deterministic) {
    // write result for this block to global mem. We do not optimize away the
    // global write when we use the single pass algorithm with determinism
    // because it would otherwise make the code non deterministic which is
    // something we want to avoid at all costs.
    if (tid == 0)
      g_odata[blockIdx.x] = mySum;
    // We have the option to let the last block do the final reduction instead
    // of either calling the function once more or copy the partial results to
    // CPU and do the final reduction on CPU. Both ways are deterministic but
    // will lead to slightly different answers because of the order of the
    // operations.

    // On CPU we use Kahan while on GPU we use the tree reduction again. Which
    // one is faster is also interesting.

    if (single_round_) {
      if (gridDim.x > 1) {
        __shared__ bool amLast;

        // wait until all outstanding memory instructions in this thread are
        // Finished
        __threadfence();

        // Thread 0 takes a ticket
        if (tid == 0) {
          unsigned int ticket = atomicInc(retirementCount, gridDim.x);
          // If the ticket ID is equal to the number of blocks, we are the last
          // block!
          amLast = (ticket == gridDim.x - 1);
        }

        __syncthreads();

        // The last block sums the results of all other blocks
        if (amLast) {
          int i = tid;
          mySum = (T)0;

          switch (last_stage) {
          case last_stage_summation_algorithm::kahan_method:
            if (threadIdx.x == 0)
              mySum = KahanBabushkaNeumaierSum(g_odata, gridDim.x);
            break;
          case last_stage_summation_algorithm::recursive_method:
            if (threadIdx.x == 0)
              mySum = recursive_sum(g_odata, gridDim.x);
            break;
          case last_stage_summation_algorithm::pairwise_method:
          default: {
            while (i < gridDim.x) {
              mySum += g_odata[i];
              i += blockSize;
            }
            blockReduce<T, blockSize, warp_method__>(sdata, mySum, tid);
          } break;
          }
          __syncthreads();
          if (tid == 0) {
            g_odata[0] = mySum;
          }
        }
      }
    }
  } else {
    if (tid == 0)
      atomicAdd(&g_odata[0], mySum);
    return;
  }
}

/*
 * this kernel calculates the sum of floating point numbers and register the
 * block index associated to the atomic operation executed on the memory
 * controller.
 */

template <class T, unsigned int blockSize>
__global__ void
reduce_register_atomic_gpu(const T *__restrict__ g_idata, T *g_odata,
                           int *__restrict__ block_index__, unsigned int n) {
  // Handle to thread block group
  T *sdata = SharedMemory<T>();
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  // use the end of scratch memory to store how many block are already done.

  unsigned int *retirementCount = (unsigned int *)(g_odata + gridDim.x + 1);
  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (i + blockSize < n)
      mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  blockReduce<T, blockSize, warp_reduction_method::shared_memory>(sdata, mySum,
                                                                  tid);
  // write result for this block to global mem. We do not optimize away the
  // global write when we use the single pass algorithm with determinism
  // because it would otherwise make the code non deterministic which is
  // something we want to avoid at all costs.
  if (tid == 0)
    g_odata[blockIdx.x] = mySum;
  // We have the option to let the last block do the final reduction instead
  // of either calling the function once more or copy the partial results to
  // CPU and do the final reduction on CPU. Both ways are deterministic but
  // will lead to slightly different answers because of the order of the
  // operations.

  // On CPU we use Kahan while on GPU we use the tree reduction again. Which
  // one is faster is also interesting.

  if (gridDim.x > 1) {
    __shared__ bool amLast;

    // wait until all outstanding memory instructions in this thread are
    // Finished
    __threadfence();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = atomicInc(retirementCount, gridDim.x);
      block_index__[ticket] = blockIdx.x;
      // If the ticket ID is equal to the number of blocks, we are the last
      // block!
      amLast = (ticket == gridDim.x - 1);
    }

    __syncthreads();

    // The last block sums the results of all other blocks
    if (amLast) {
      int i = tid;
      mySum = (T)0;

      while (i < gridDim.x) {
        mySum += g_odata[i];
        i += blockSize;
      }

      blockReduce<T, blockSize, warp_reduction_method::shared_memory>(
          sdata, mySum, tid);

      if (tid == 0) {
        g_odata[0] = mySum;
      }
    }
  }
}

/* ----------------------------------------------------------------------------------
 */

template <typename T, last_stage_summation_algorithm last_stage,
          warp_reduction_method warp_reduction__, algorithm_type algo__>
void reduce_step(whip::stream_t stream, int size, int threads, int blocks,
                 const T *d_idata, T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize = threads * sizeof(T);

  switch (threads) {
  case 512:
    reduce_gpu<T, 512, last_stage, warp_reduction__, algo__>
        <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
    break;
  case 256:
    reduce_gpu<T, 256, last_stage, warp_reduction__, algo__>
        <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
    break;

  case 128:
    reduce_gpu<T, 128, last_stage, warp_reduction__, algo__>
        <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
    break;

  case 64:
    reduce_gpu<T, 64, last_stage, warp_reduction__, algo__>
        <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
    break;
  }
}

int get_block_size(const int size__) {
  int block_size = 32;
  if (size__ > 64) {
    block_size = 128;
  }
  if (size__ > 128) {
    block_size = 256;
  }
  if (size__ > 256) {
    block_size = 512;
  }
  return block_size;
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2(n / 2) : maxThreads;
    blocks = max(1, n / (threads * 2));
  }

  blocks = min(maxBlocks, blocks);
}

template <class T>
__global__ void reduce_atomic_cuda(const T *__restrict__ data__,
                                   const size_t size__, T *scratch__) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size__)
    return;
  atomicAdd(scratch__, data__[tid]);
}

template <class T>
void reduce_atomic(whip::stream_t stream__, const int size__, const T *data__,
                   T *scratch__) {
  const int block_size_ = 128;
  const int grid_size_ = (size__ + block_size_ - 1) / block_size_;
  reduce_atomic_cuda<<<grid_size_, block_size_, 0, stream__>>>(data__, size__,
                                                               scratch__);
}

template <class T>
T reduce(reduction_method method__, whip::stream_t stream__,
         const int block_size__, const int grid_size__, const int size__,
         const T *data__, T *scratch__) {
  T *out;
  int grid_size = 1;
  int block_size = 1;
  const T *in = data__;
  out = scratch__;
  double result = 0.0;

  block_size = block_size__;
  if (block_size__ < 64)
    block_size = 64;

  if (block_size__ > 512)
    block_size = 512;

  grid_size = (size__ + 2 * block_size - 1) / (2 * block_size);
  if ((grid_size__ > 0) && (grid_size__ < grid_size))
    grid_size = grid_size__;

  std::vector<T> tmp(grid_size);

  // <typename T, bool shuffle__ = false, bool single_round_ = false,
  //  bool deterministic = true>

  whip::memset_async(out, 0, sizeof(T) * (grid_size + 1), stream__);

  bool single_pass = false;
  switch (method__) {
  case reduction_method::single_pass_gpu_det_shuffle_kahan_gpu: {
    reduce_step<T, last_stage_summation_algorithm::kahan_method,
                warp_reduction_method::shuffle_down,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_det_shuffle_recursive_gpu: {
    reduce_step<T, last_stage_summation_algorithm::recursive_method,
                warp_reduction_method::shuffle_down,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_det_shuffle: {
    reduce_step<T, last_stage_summation_algorithm::pairwise_method,
                warp_reduction_method::shuffle_down,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_det_kahan_gpu: {
    reduce_step<T, last_stage_summation_algorithm::kahan_method,
                warp_reduction_method::shared_memory,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_det_recursive_gpu: {
    reduce_step<T, last_stage_summation_algorithm::recursive_method,
                warp_reduction_method::shared_memory,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_det_shared: {
    reduce_step<T, last_stage_summation_algorithm::pairwise_method,
                warp_reduction_method::shared_memory,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_shuffle_atomic: {
    reduce_step<T, last_stage_summation_algorithm::none,
                warp_reduction_method::shuffle_down,
                algorithm_type::last_stage_atomic_add>(
        stream__, size__, block_size, grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_shared_atomic: {
    reduce_step<T, last_stage_summation_algorithm::none,
                warp_reduction_method::shared_memory,
                algorithm_type::last_stage_atomic_add>(
        stream__, size__, block_size, grid_size, in, out);
    single_pass = true;
  } break;
  case reduction_method::single_pass_gpu_atomic_only: {
    reduce_atomic(stream__, size__, in, out);
    single_pass = true;
  } break;
  case reduction_method::two_pass_gpu_det_shuffle_recursive_cpu:
  case reduction_method::two_pass_gpu_det_shuffle_kahan_cpu:
    reduce_step<T, last_stage_summation_algorithm::none,
                warp_reduction_method::shuffle_down,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    break;
  case reduction_method::two_pass_gpu_det_recursive_cpu:
  case reduction_method::two_pass_gpu_det_kahan_cpu:
    reduce_step<T, last_stage_summation_algorithm::none,
                warp_reduction_method::shared_memory,
                algorithm_type::deterministic>(stream__, size__, block_size,
                                               grid_size, in, out);
    break;
  case reduction_method::cub_reduce_method:
    return 0.0;
    break;
  default:
    std::cout << "Oh Oh: wrong method\n";
    break;
  }

  if (single_pass)
    whip::memcpy_async(&result, out, sizeof(T), whip::memcpy_device_to_host,
                       stream__);
  else
    whip::memcpy_async(tmp.data(), out, sizeof(T) * grid_size,
                       whip::memcpy_device_to_host, stream__);

  whip::stream_synchronize(stream__);

  switch (method__) {
  case reduction_method::two_pass_gpu_det_shuffle_recursive_cpu:
  case reduction_method::two_pass_gpu_det_recursive_cpu:
    result = recursive_sum(tmp.data(), tmp.size());
    break;
  case reduction_method::two_pass_gpu_det_shuffle_kahan_cpu:
  case reduction_method::two_pass_gpu_det_kahan_cpu:
    result = KahanBabushkaNeumaierSum(tmp.data(), tmp.size());
    break;
  default:
    break;
  }

  return result;
}

template <typename T>
T cub_reduce(whip::stream_t stream__, size_t &temp_storage_bytes,
             T *temp_storage, const size_t size__, T *data__, T *scratch__) {
  T res__ = 0;
#ifdef REDUCE_USE_CUDA
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, data__,
                         static_cast<double *>(scratch__), size__, stream__);
#else
  hipcub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, data__,
                            static_cast<double *>(scratch__), size__, stream__);
#endif

  if (scratch__ != NULL)
    whip::memcpy_async(&res__, scratch__, sizeof(T),
                       whip::memcpy_device_to_host, stream__);
  whip::stream_synchronize(stream__);
  return res__;
}

template double reduce(reduction_method method__, whip::stream_t stream__,
                       const int fixed_block__, const int grid_size__,
                       const int size__, const double *data__,
                       double *scratch__);

template <class T>
T reduce_register_atomic(whip::stream_t stream__, const int size__, T *data__,
                         int *block_index__, T *scratch__) {
  T *in, *out;
  int block_size = 64;
  int grid_size = (size__ + 2 * block_size - 1) / (2 * block_size);
  in = data__;
  out = scratch__;
  double result = 0.0;
  int smemSize = 64 * sizeof(T);
  // number of threads per block
  // size of the block grid. The reduce algorithm add at least two elements
  // per thread. so the block grid should NOT be size__/block_size but
  // size__/ 2 * block_size__).
  std::vector<T> tmp(grid_size, 0.0);
  whip::memset_async(out + grid_size + 1, 0, sizeof(T), stream__);
  reduce_register_atomic_gpu<T, 64>
      <<<grid_size, block_size, smemSize, stream__>>>(in, out, block_index__,
                                                      size__);
  whip::memcpy_async(&result, out, sizeof(T), whip::memcpy_device_to_host,
                     stream__);
  whip::stream_synchronize(stream__);

  return result;
}

template double reduce_register_atomic(whip::stream_t stream__,
                                       const int size__, double *data__,
                                       int *block_index__, double *scratch__);

template double cub_reduce(whip::stream_t stream__, size_t &temp_storage_bytes,
                           double *temp_storage, const size_t size__,
                           double *data__, double *scratch__);

#ifdef USE_MAIN
int main(int argc, char **argv) {
  double *table = nullptr;
  double *scratch = nullptr;
  double *temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  int number_of_fp_elements = 1000000;
  curandGenerator_t gen_;
  whip::stream_t stream_;

  whip::stream_create(&stream_);
  curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen_, 0x234abe);

  whip::malloc(&table, sizeof(double) * number_of_fp_elements);
  whip::malloc(&scratch, sizeof(double) * number_of_fp_elements);

  curandGenerateUniformDouble(gen_, table, number_of_fp_elements);
  // we ignore the result
  cub_reduce(stream_, temp_storage_bytes, temp_storage, number_of_fp_elements,
             table, scratch);

  whip::malloc(&temp_storage, sizeof(double) * temp_storage_bytes);
  for (int round = 0; round < 10; round++) {
    // reduction using cub
    double res1 = cub_reduce(stream_, temp_storage_bytes, temp_storage,
                             number_of_fp_elements, table, scratch);
    // reduction using the two stages reduction
    double res2 = reduce_step_det<double>(
        stream_, 512, -1, number_of_fp_elements, table, scratch);
    // use the thread_fence instruction and a counter. Check the case
    // deterministic +  single
    double res3 = reduce(reduction_method::single_pass_gpu_det_shared, stream_,
                         512, -1, number_of_fp_elements, table, scratch);
    uint64_t resi1;
    uint64_t resi2;
    uint64_t resi3;

    memcpy(&resi1, &res1, sizeof(double));
    memcpy(&resi2, &res2, sizeof(double));
    memcpy(&resi3, &res3, sizeof(double));
    std::cout << "Results : supposed to be deterministic: " << resi3
              << " Deterministic two stages: " << resi2;
    std::cout << " CUb version: " << resi1 << std::endl;
  }

  whip::stream_destroy(stream_);
  whip::free(table);
  whip::free(scratch);
  whip::free(temp_storage);
  curandDestroyGenerator(gen_);

  return 0;
}
#endif
