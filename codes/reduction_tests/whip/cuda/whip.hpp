// whip
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if !(defined(WHIP_CUDA) || defined(WHIP_HIP))
#error "whip requires exactly one of WHIP_CUDA and WHIP_HIP to be defined. Neither is defined."
#endif

#if defined(WHIP_CUDA) && defined(WHIP_HIP)
#error "whip requires exactly one of WHIP_CUDA and WHIP_HIP to be defined. Both are defined."
#endif

#if defined(WHIP_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(WHIP_HIP)
#include <hip/hip_runtime.h>
#endif

#define WHIP_STRINGIFY(x) WHIP_STRINGIFY_IMPL(x)
#define WHIP_STRINGIFY_IMPL(x) #x

#include <cstddef>
#include <stdexcept>
#include <string>

namespace whip {
inline constexpr std::size_t version_major = 0;
inline constexpr std::size_t version_minor = 3;
inline constexpr std::size_t version_patch = 0;
inline constexpr const char *version_string = "0.3.0";

// Types
using device_prop = cudaDeviceProp;
using dim3 = dim3;
using error_t = cudaError_t;
using event_t = cudaEvent_t;
using memcpy_kind = cudaMemcpyKind;
using stream_callback_t = cudaStreamCallback_t;
using stream_t = cudaStream_t;

// Constants
inline constexpr int event_disable_timing = cudaEventDisableTiming;
inline constexpr memcpy_kind memcpy_host_to_host = cudaMemcpyHostToHost;
inline constexpr memcpy_kind memcpy_host_to_device = cudaMemcpyHostToDevice;
inline constexpr memcpy_kind memcpy_device_to_host = cudaMemcpyDeviceToHost;
inline constexpr memcpy_kind memcpy_device_to_device = cudaMemcpyDeviceToDevice;
inline constexpr memcpy_kind memcpy_default = cudaMemcpyDefault;
inline constexpr int stream_non_blocking = cudaStreamNonBlocking;

// Errors
inline constexpr error_t success = cudaSuccess;
inline constexpr error_t error_not_ready = cudaErrorNotReady;

inline const char* get_error_string(error_t error) {
#if defined(WHIP_CUDA)
    return cudaGetErrorString(error);
#elif defined(WHIP_HIP)
    return hipGetErrorString(error);
#endif
}

inline const char *get_error_name(error_t error) {
#if defined(WHIP_CUDA)
  return cudaGetErrorName(error);
#elif defined(WHIP_HIP)
  return hipGetErrorName(error);
#endif
}

namespace impl {
inline std::string make_error_string(error_t error, char const *function) {
  return std::string("[whip] ") + function + " returned " + get_error_name(error) + " (" + get_error_string(error) +
         ")";
}

inline std::string make_error_string(error_t error) {
  return std::string("[whip] ") + WHIP_STRINGIFY(WHIP_BACKEND) " function call returned " + get_error_name(error) + " (" +
         get_error_string(error) + ")";
}
} // namespace impl

// Custom exception which wraps a CUDA/HIP error
class exception final : public std::runtime_error {
public:
  explicit exception(error_t error) : std::runtime_error(impl::make_error_string(error)), error(error) {}
  explicit exception(error_t error, char const *function)
      : std::runtime_error(impl::make_error_string(error, function)), error(error) {}
  error_t get_error() const noexcept { return error; }

private:
  error_t error;
};

// Check an error and throw an exception on failure.
inline void check_error(error_t e) {
  if (e != success) {
    throw exception(e);
  }
}

namespace impl {
inline void check_error(error_t e, char const *function) {
  if (e != success) {
    throw exception(e, function);
  }
}

// Check an error and throw an exception on failure, except error_not_ready.
// This is useful for query functions.
inline bool check_error_query(error_t e, char const *function) {
  switch (e) {
  case success:
    return true;
  case error_not_ready:
    return false;
  default:
    throw exception(e, function);
  }
}
} // namespace impl

// Functions
inline constexpr auto check_last_error = []() { return whip::impl::check_error(cudaGetLastError(), "cudaGetLastError"); };
inline constexpr auto device_get_stream_priority_range = [](int* least, int* greatest) { return whip::impl::check_error(cudaDeviceGetStreamPriorityRange(least, greatest), "cudaDeviceGetStreamPriorityRange"); };
inline constexpr auto device_synchronize = []() { return whip::impl::check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); };
inline constexpr auto device_reset = []() { return whip::impl::check_error(cudaDeviceReset(), "cudaDeviceReset"); };
inline constexpr auto event_create = [](event_t* event) { return whip::impl::check_error(cudaEventCreate(event), "cudaEventCreate"); };
inline constexpr auto event_create_with_flags = [](event_t* event, unsigned flags) { return whip::impl::check_error(cudaEventCreateWithFlags(event, flags), "cudaEventCreateWithFlags"); };
inline constexpr auto event_destroy = [](event_t event) { return whip::impl::check_error(cudaEventDestroy(event), "cudaEventDestroy"); };
inline constexpr auto event_elapsed_time = [](float* milliseconds, event_t start_event, event_t stop_event) { return whip::impl::check_error(cudaEventElapsedTime(milliseconds, start_event, stop_event), "cudaEventElapsedTime"); };
inline constexpr auto event_synchronize = [](event_t event) { return whip::impl::check_error(cudaEventSynchronize(event), "cudaEventSynchronize"); };
inline constexpr auto event_ready = [](event_t event) { return whip::impl::check_error_query(cudaEventQuery(event), "cudaEventQuery"); };
inline constexpr auto event_record = [](event_t event, stream_t stream) { return whip::impl::check_error(cudaEventRecord(event, stream), "cudaEventRecord"); };
inline constexpr auto free = [](auto* p) { return whip::impl::check_error(cudaFree(p), "cudaFree"); };
inline constexpr auto free_async = [](auto* p, stream_t stream) { return whip::impl::check_error(cudaFreeAsync(p, stream), "cudaFreeAsync"); };
inline constexpr auto free_host = [](auto* p) { return whip::impl::check_error(cudaFreeHost(p), "cudaFreeHost"); };
inline constexpr auto get_device = [](int* device) { return whip::impl::check_error(cudaGetDevice(device), "cudaGetDevice"); };
inline constexpr auto get_device_count = [](int* count) { return whip::impl::check_error(cudaGetDeviceCount(count), "cudaGetDeviceCount"); };
inline constexpr auto get_device_properties = [](device_prop* prop, int device) { return whip::impl::check_error(cudaGetDeviceProperties(prop, device), "cudaGetDeviceProperties"); };
inline constexpr auto launch_kernel = [](const auto* f, dim3 num_blocks, dim3 dim_blocks, void** args, std::size_t shared_mem_bytes, stream_t stream) { return whip::impl::check_error(cudaLaunchKernel(f, num_blocks, dim_blocks, args, shared_mem_bytes, stream), "cudaLaunchKernel"); };
inline constexpr auto malloc = [](auto** p, std::size_t size) { return whip::impl::check_error(cudaMalloc(p, size), "cudaMalloc"); };
inline constexpr auto malloc_async = [](auto** p, std::size_t size, stream_t stream) { return whip::impl::check_error(cudaMallocAsync(p, size, stream), "cudaMallocAsync"); };
inline constexpr auto malloc_host = [](auto** p, std::size_t size) { return whip::impl::check_error(cudaMallocHost(p, size), "cudaMallocHost"); };
inline constexpr auto mem_get_info = [](std::size_t* free, std::size_t* total) { return whip::impl::check_error(cudaMemGetInfo(free, total), "cudaMemGetInfo"); };
inline constexpr auto memcpy = [](auto* dst, const auto* src, std::size_t size_bytes, memcpy_kind kind) { return whip::impl::check_error(cudaMemcpy(dst, src, size_bytes, kind), "cudaMemcpy"); };
inline constexpr auto memcpy_2d = [](auto* dst, std::size_t dpitch, const auto* src, std::size_t spitch, std::size_t width, std::size_t height, memcpy_kind kind) { return whip::impl::check_error(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind), "cudaMemcpy2D"); };
inline constexpr auto memcpy_2d_async = [](auto* dst, std::size_t dpitch, const auto* src, std::size_t spitch, std::size_t width, std::size_t height, memcpy_kind kind, stream_t stream) { return whip::impl::check_error(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream), "cudaMemcpy2DAsync"); };
inline constexpr auto memcpy_async = [](auto* dst, const auto* src, std::size_t size_bytes, memcpy_kind kind, stream_t stream) { return whip::impl::check_error(cudaMemcpyAsync(dst, src, size_bytes, kind, stream), "cudaMemcpyAsync"); };
inline constexpr auto memset = [](auto* dst, int value, std::size_t size_bytes) { return whip::impl::check_error(cudaMemset(dst, value, size_bytes), "cudaMemset"); };
inline constexpr auto memset_2d = [](auto* dst, std::size_t pitch, int value, std::size_t width, std::size_t height) { return whip::impl::check_error(cudaMemset2D(dst, pitch, value, width, height), "cudaMemset2D"); };
inline constexpr auto memset_2d_async = [](auto* dst, std::size_t pitch, int value, std::size_t width, std::size_t height, stream_t stream) { return whip::impl::check_error(cudaMemset2DAsync(dst, pitch, value, width, height, stream), "cudaMemset2DAsync"); };
inline constexpr auto memset_async = [](auto* dst, int value, std::size_t size_bytes, stream_t stream) { return whip::impl::check_error(cudaMemsetAsync(dst, value, size_bytes, stream), "cudaMemsetAsync"); };
inline constexpr auto set_device = [](int device) { return whip::impl::check_error(cudaSetDevice(device), "cudaSetDevice"); };
inline constexpr auto stream_add_callback = [](stream_t stream, stream_callback_t callback, void* user_data, unsigned int flags) { return whip::impl::check_error(cudaStreamAddCallback(stream, callback, user_data, flags), "cudaStreamAddCallback"); };
inline constexpr auto stream_create = [](stream_t* stream) { return whip::impl::check_error(cudaStreamCreate(stream), "cudaStreamCreate"); };
inline constexpr auto stream_create_with_flags = [](stream_t* stream, unsigned int flags) { return whip::impl::check_error(cudaStreamCreateWithFlags(stream, flags), "cudaStreamCreateWithFlags"); };
inline constexpr auto stream_create_with_priority = [](stream_t* stream, unsigned int flags, int priority) { return whip::impl::check_error(cudaStreamCreateWithPriority(stream, flags, priority), "cudaStreamCreateWithPriority"); };
inline constexpr auto stream_destroy = [](stream_t stream) { return whip::impl::check_error(cudaStreamDestroy(stream), "cudaStreamDestroy"); };
inline constexpr auto stream_get_flags = [](stream_t stream, unsigned int* flags) { return whip::impl::check_error(cudaStreamGetFlags(stream, flags), "cudaStreamGetFlags"); };
inline constexpr auto stream_ready = [](stream_t stream) { return whip::impl::check_error_query(cudaStreamQuery(stream), "cudaStreamQuery"); };
inline constexpr auto stream_synchronize = [](stream_t stream) { return whip::impl::check_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize"); };
} // namespace whip
