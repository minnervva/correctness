#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

#ifdef REDUCE_USE_CUDA
#include <cub/cub.cuh>
#else
#include <hipcub/device/device_scan.hh>
#endif
#include "utils.hpp"
#include <whip.hpp>

template <typename T> void test_prefix_sum(const int number_of_items__) {
  T *data__;
  T *res__;

  T *res_copy_fp__;
  void *res_copy__;
  void *res_copy1__;
  generator_t gen_;
  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  whip::malloc(&data__, sizeof(T) * number_of_items__);
  whip::malloc(&res__, sizeof(T) * number_of_items__);
  res_copy__ = std::malloc(sizeof(T) * number_of_items__);
  res_copy1__ = std::malloc(sizeof(T) * number_of_items__);
  res_copy_fp__ = (double *)std::malloc(sizeof(T) * number_of_items__);

  CreateGenerator(&gen_, 0x127956abced);

  GenerateUniformDouble(gen_, data__, number_of_items__);

  // compute the reference list

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, data__,
                                res__, number_of_items__);
  whip::malloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, data__,
                                res__, number_of_items__);

  whip::memcpy(res_copy_fp__, res__, sizeof(T) * number_of_items__,
               whip::memcpy_device_to_host);

  // Now run the exclusive scan multiple times over
  std::setprecision(16);
  for (int i = 0; i < 10; i++) {
    std::cout << "Itteration #"<< i << std::endl;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, data__,
                                  res__, number_of_items__);
    whip::memcpy(res_copy__, res__, sizeof(T) * number_of_items__,
                 whip::memcpy_device_to_host);
    uint64_t *rs1 = static_cast<uint64_t *>(res_copy__);
    uint64_t *rs2 = static_cast<uint64_t *>((void*)res_copy_fp__);
    T *rs3 = static_cast<T*>(res_copy__);
    const int rc = std::memcmp(res_copy__, res_copy_fp__, sizeof(T) *number_of_items__);
    if (rc != 0) {
      std::cout << "Check which element are different\n";
    for (int s = 0; s < number_of_items__; s++) {
      if (rs1[s] != rs2[s]) {
        std::cout << "Not deterministic: " << s << " "
                  << std::abs((rs3[s] - res_copy_fp__[s]) / res_copy_fp__[s])
                  << std::endl;
        break;
      }
    }
  }
  }
  whip::free(d_temp_storage);
  whip::free(data__);
  whip::free(res__);
  std::free(res_copy__);
  std::free(res_copy_fp__);
  std::free(res_copy1__);
  DestroyGenerator(gen_);
}

int main(int argc, char **argv) {
  test_prefix_sum<double>(1000000);
  return 0;
}
