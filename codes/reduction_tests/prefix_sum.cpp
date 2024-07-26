#include "whip/hip/whip.hpp"
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

#include <whip.hpp>

#ifdef REDUCE_USE_CUDA#include <cub/cub.cuh>
#else
#include <hipcub/device/device_scan.hh>
#endif
#include "utils.hpp"

template<typename T> void test_prefix_sum(const int number_of_items__) {
    T *data__;
    T *res__;

    T *res_copy_fp__;
    void *res_copy__;
    void *res_copy1__;
    generator_t gen_;
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    whip::malloc(&data__, sizeof(T) *number_of_items__);
    whip::malloc(&res__, sizeof(T) *number_of_items__);
    res_copy__ = std::malloc(sizeof(T) *number_of_items__);
    res_copy1__ = std::malloc(sizeof(T) *number_of_items__);
    res_copy_fp__ = std::malloc(sizeof(T) *number_of_items__);

    CreateGenerator(&gen_, 0x127956abced);

    GenerateUniformDouble(gen_, data__, number_of_items__);

    // compute the reference list

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        data__, res__, number_of_items__);
    whip::malloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        data__, res__, number_of_items__);

    whip::memcpy(res_copy_fp__, res__, sizeof(T) *number_of_items__, whip::memcpy_host_to_device);
    std::memcpy(res_copy__, res_copy_fp__, sizeof(T) * number_of_items__);

    // Now run the exclusive scan multiple times over
    for (int i = 0; i < 10; i++) {
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            data__, res__, number_of_items__);
        whip::memcpy(res_copy_fp__, res__, sizeof(T) *number_of_items__, whip::memcpy_host_to_device);
            T *rs1 = static_cast<T> (res_copy);
            for (int s = 0; s < number_of_items__; s++)
                if (std::abs((rs1[s] - res_copy_fp__[s])/res_copy_fp__[s]) > 0.000000000001)
                    fmt::print("Not deterministic: {} {:.15f} {:.15f}\n", rs1[s], res_copy_fp__[s]);
    }

    whip::free(d_temp_storage);
    whip::free(data__);
    whip::free(res__);
    std::free(res_copy__);
    std::free(res_copy_fp__);
    std::free(res_copy1__);
    DestroyGenerator(gen_);
}

int main(int argc, char **argv)
{
    test_prefix_sum<double>(1000000);
    return 0;
}
