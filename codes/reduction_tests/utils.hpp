#pragma once
#include <cmath>
#include <string>
#include <vector>

#ifdef REDUCE_USE_CUDA
#include <curand.h>
typedef curandGenerator_t generator_t;
#else
#include <hiprand.h>
typedef hiprandGenerator_t generator_t;
#endif

#include <whip.hpp>

void scale_the_darn_thing(int *neighbor_list__, const int number_of_elements__,
                          const int number_of_atoms__);

void uniform_double(const int number_of_elements__, const double xmin,
                    const double xmax__, double *rng);

template <class T> T reduce_atomic(whip::stream_t stream__, const int size__, T*data__, T*scratch__);

template <typename T>
T reduce_step_non_det(whip::stream_t stream__, const int block_size__,
                      const int grid_size__, const int size__, T *data__,
                      T *scratch__);
template <typename T>
T reduce_step_det(whip::stream_t stream__, const int block_size__,
                  const int grid_size__, const int size__, T *data__,
                  T *scratch__);
template <typename T>
T reduce_step_det_single_round(whip::stream_t stream__, const int block_size__,
                               const int grid_size__, const int size__,
                               T *data__, T *scratch__);
template <typename T>
T reduce_step_det_shuffle_single_round(whip::stream_t stream__,
                                       const int block_size__,
                                       const int grid_size__, const int size__,
                                       T *data__, T *scratch__);
template <typename T>
T reduce_step_det_shuffle(whip::stream_t stream__, const int block_size__,
                          const int grid_size__, const int size__, T *data__,
                          T *scratch__);
template <typename T>
T cub_reduce(whip::stream_t stream__, size_t &temp_storage_bytes,
             T *temp_storage, const size_t size__, T *data__, T *scratch__);
template <typename T>
void center_and_scale_distribution(T *data__, const T scal__, const T center__,
                                   const int size__);

/* ref : https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
template <typename T>
T KahanBabushkaNeumaierSum(const T *data__, const int size__) {
  T sum = (T)0;
  T c = (T)0;
  for (int i = 0; i < size__; i++) {
    T t = sum + data__[i];
    if (std::abs(sum) > std::abs(data__[i]))
      c +=
          (sum - t) +
          data__[i]; // If sum is bigger, low-order digits of input[i] are lost.
    else
      c += (data__[i] - t) + sum; // Else low-order digits of sum are lost.
    sum = t;
  }
  return sum + c;
}

template <typename T> T KahanBabushkaNeumaierSum(const std::vector<T> &data__) {
  T sum = (T)0;
  T c = (T)0;
  for (auto &elm_ : data__) {
    T t = sum + elm_;
    if (std::abs(sum) > std::abs(elm_))
      c += (sum - t) +
           elm_; // If sum is bigger, low-order digits of input[i] are lost.
    else
      c += (elm_ - t) + sum; // Else low-order digits of sum are lost.
    sum = t;
  }
  return sum + c;
}

template <typename T> T min(std::vector<T> data__) {
  T min__ = 10000000000000000;
  for (auto &t_ : data__)
    min__ = std::min(t_, min__);
  return min__;
}

template <typename T> T max(std::vector<T> data__) {
  T max__ = -10000000000000000;
  for (auto &t_ : data__)
    max__ = std::max(t_, max__);
  return max__;
}

template <typename T> T mean(std::vector<T> data__) {
  double inv_size = 1.0 / ((T)data__.size());
  return KahanBabushkaNeumaierSum(data__) * inv_size;
}

template <typename T> T mean(T *data__, const int size__) {
  double inv_size = 1.0 / ((T)size__);
  return KahanBabushkaNeumaierSum(data__, size__) * inv_size;
}

template <typename T> T std_dev(std::vector<T> data__) {
  const T mean_data = mean(data__);
  std::vector<T> tmp(data__);
  for (auto &t_ : tmp)
    t_ -= mean_data;

  for (auto &t_ : tmp)
    t_ *= t_;

  double inv_size = 1.0 / ((T)data__.size() - 1);

  return std::sqrt(KahanBabushkaNeumaierSum(tmp) * inv_size);
}

template <typename T> T std_dev(T *data__, const int size__) {
  const T mean_data = mean(data__, size__);

  std::vector<T> tmp(size__);
  for (int i = 0; i < tmp.size(); i++)
    tmp[i] = (data__[i] - mean_data) * (data__[i] - mean_data);

  double inv_size = 1.0 / ((T)(size__ - 1));

  return std::sqrt(KahanBabushkaNeumaierSum(tmp) * inv_size);
}

void generate_random_numbers(std::string distribution__, generator_t gen__,
                             const double amplitude__,
                             const double distribution_center__,
                             const double standard_deviation__, double *data__,
                             const size_t length__);

__inline__ void CreateGenerator(generator_t *gen_, const unsigned int seed__) {
#ifdef REDUCE_USE_CUDA
  curandCreateGenerator(gen_, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(*gen_, seed__);
#else
  hiprandCreateGenerator(gen_, HIPRAND_RNG_PSEUDO_DEFAULT);
  hiprandSetPseudoRandomGeneratorSeed(*gen_, seed__);
#endif
}

__inline__ void GenerateUniformDouble(generator_t gen__, double *data__,
                                      size_t length__) {
#ifdef REDUCE_USE_CUDA
  curandGenerateUniformDouble(gen__, data__, length__);
#else
  hiprandGenerateUniformDouble(gen__, data__, length__);
#endif
}

__inline__ void GenerateNormalDouble(generator_t gen__, double *data__,
                                     const size_t length__,
                                     const double distribution_center__,
                                     const double standard_deviation__) {
#ifdef REDUCE_USE_CUDA
  curandGenerateNormalDouble(gen__, data__, length__, distribution_center__,
                             standard_deviation__);
#else
  hiprandGenerateNormalDouble(gen__, data__, length__, distribution_center__,
                              standard_deviation__);
#endif
}
__inline__ void DestroyGenerator(generator_t gen__) {
#ifdef REDUCE_USE_CUDA
  curandDestroyGenerator(gen__);
#else
  hiprandDestroyGenerator(gen__);
#endif
}
