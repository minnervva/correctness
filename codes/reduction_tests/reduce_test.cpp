#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

std::string GPU_TYPE;

#ifdef REDUCE_USE_CUDA
#include "whip/cuda/whip.hpp"
#include <curand.h>
#else
#include "whip/hip/whip.hpp"
#include <hiprand/hiprand.h>
#endif
// #include <fmt/core.h>

#include "mdarray.hpp"
#include "timing/rt_graph.hpp"
#include "utils.hpp"

std::vector<std::pair<std::string, reduction_method>> method_to_test;

/* Reference values for summing 100 samples of 1 M uniformly distributed FP
 * elements the RNG is mercenne twister from the c++ library not the GPU
 * implementation as the runtime parameters are GPU dependent.
 */

uint64_t reference_values_v100[] = {
    0x4088aadc4d7d66f9, 0xc0a06542c32c56d0, 0xc09909802a95df8a,
    0xc0556565d156b821, 0x40a7c35eced3b108, 0x40ab116c5466dddd,
    0x409bbb30d2f6dc9d, 0xc0a6641f93cfae58, 0x408f970b4855c84a,
    0xc0861279a19a8fe0, 0xc0a3148b0cd0589b, 0xc09bddd730473acf,
    0xc0ad9a4cd6437282, 0x40b0fa343a55eb74, 0xc09108d5c0380f15,
    0xc0b22a51858db8e4, 0xc08d6306d324b436, 0x40b457c006c2162c,
    0xc0a84a3be6011dc4, 0xc0b56fe8c50c43a1, 0x40a95fbcb33276d4,
    0xc0aba6fa8f38fecd, 0x40a814445518ea57, 0x40792e76197d6659,
    0xc0b8332d4cd0e061, 0x40a0e4c2333039ec, 0x409958c3d99d172e,
    0xc081239b82f55b49, 0xc0bd63b44bc6eb96, 0xc0a16ffbe1d3e58e,
    0x40a29eb4e2766228, 0xc08b2c94a7117d96, 0x40869ad828f9f466,
    0xc0bc809a96f8d4b1, 0xc09662df0264ec9d, 0xc0a0959ec1fe8fdf,
    0xc073f3943734c84d, 0xc0b65464b23d2413, 0xc0997a22080c1c40,
    0xc094f034979d00b2, 0xc0ac80657dde0b57, 0xc0a9e3592b6874c3,
    0x40959e487d58b512, 0x404e30d224700181, 0x40a4a75abb2bbd46,
    0x40901bc68f2979ab, 0x408b667d4577ea7d, 0xc0433cc853acafe7,
    0xc08413ff93b3206e, 0x40b1b3643aae56bf, 0x4094ef481201a8bd,
    0xc0ad5dec08e61367, 0xc0ad1df2c55f3816, 0xc089370f100157b3,
    0x406a8e0cb0b8a036, 0x4088e5477fa89fb8, 0x4079de37759eed86,
    0xc097edcc3d81366c, 0xc0b0258223dc3769, 0x40901cb3dcad267d,
    0x40b0d0440cf3dcac, 0xc0b0f0dac41a8232, 0xc0937d78ef3f941a,
    0x40b338bcaee50f12, 0xc07df17229151c5e, 0xc07d3d2c45763bb6,
    0xc0b62ee3a15db949, 0x407150ef98c4a802, 0xc09aedf25d32e248,
    0x40b043652ab83dc2, 0x4091a6eae4c9ad5e, 0x40a7e54f9aab0ce3,
    0x4085411444858b4c, 0x40b40d736dd7d20c, 0x40789225b4096b10,
    0xc05dc34bd79c768a, 0x40a78ece1a211390, 0xc0af967cc576b0bb,
    0xc069b1f872911c76, 0xc0a752063e2b1aec, 0x40931208c9765b96,
    0xc065f886839c4f81, 0xc0a65486cd62b604, 0x40a33079946df58c,
    0x40a0f04a730a8ce0, 0xc0861357e9019bd1, 0xc0a3114afcefbd3b,
    0x40ac89a87a117778, 0xc0adf6741b606a2b, 0xc0a55eb5d389c41e,
    0xc09dd26bc45f342e, 0x40ab6f28a83aaa29, 0x40b416d0efe26de4,
    0xc09a82c0ba76b590, 0xc073e59c5202ac5b, 0xc0aac55297dd04ab,
    0xc0a719c47cfb1437, 0x40b4db896e16b1b4, 0x4099f05780b24f47,
    0xc08aaedab816ae70};

void generate_random_numbers_cpu(
    std::string dist_type__, std::mt19937 engine__,
    std::uniform_real_distribution<double> &dist_uniform__,
    std::normal_distribution<double> &dist_normal__,
    mdarray<double, 2, CblasRowMajor> &data__) {
  if (dist_type__ == "normal")
    for (int i = 0; i < data__.size(); i++)
      data__[i] = dist_normal__(engine__);
  else
    for (int i = 0; i < data__.size(); i++)
      data__[i] = dist_uniform__(engine__);
}

void sanity_check(whip::stream_t &stream_) {
  mdarray<double, 1, CblasRowMajor> sum_(20001);
  double *scratch__ = nullptr;
  for (int i = 0; i < sum_.size(); i++) {
    const double x = (i - 10000) * 0.001;
    sum_(i) = std::exp(-x * x * 0.5);
  }
  sum_.allocate(memory_t::device);
  sum_.copy<memory_t::host, memory_t::device>();
  whip::malloc(&scratch__, sizeof(double) * 10000);

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "Sanity checks", 70);

  const double result_of_sum = 2.506628274631001;
  fmt::print("Computing the numerical integral of exp(-x^2/2) "
             "between -10 and 10\n\n");
  fmt::print("{0:<50} {1:>20.15f}\n",
             "Result of the sum with Erf function:", result_of_sum);
  fmt::print("{0:<50} {1:>20.15f}\n", "Checking Kahan sum:",
             KahanBabushkaNeumaierSum(sum_.at<device_t::CPU>(), sum_.size()) *
                 0.001);
  fmt::print("{0:<50} {1:>20.15f}\n", "Checking Recursive sum:",
             recursive_sum(sum_.at<device_t::CPU>(), sum_.size()) * 0.001);
  for (auto &&mt : method_to_test) {
    if (mt.second != reduction_method::cub_reduce_method) {
      double result = reduce<double>(mt.second, stream_, 256, 0, sum_.size(),
                                     sum_.at<device_t::GPU>(), scratch__);
      fmt::print("Checking {0:<41} {1:>20.15f}\n", mt.first + ":",
                 result * 0.001);
    }
  }
  fmt::print("\n");
  whip::free(scratch__);
  sum_.clear();
}

void performance_tests(cxxopts::ParseResult &result, rt_graph::Timer &timer__,
                       whip::stream_t &stream__) {

  /* print pretty progress bar */

  indicators::ProgressBar bar{
      indicators::option::BarWidth{50},
      indicators::option::Start{"["},
      indicators::option::Fill{"■"},
      indicators::option::Lead{"■"},
      indicators::option::Remainder{"-"},
      indicators::option::End{" ]"},
      indicators::option::ForegroundColor{indicators::Color::cyan},
      indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};

  /*
   * There is no specific reason why we choose these numbers.
   * They can be anything. Just happens to be a power of two
   */

  const int number_of_fp__ = 512 * 8192;
  const int number_of_samples__ = 100;
  const int max_num_blocks__ = (number_of_fp__ + 64 - 1) / 64;

  generator_t gen_;
  void *scratch__ = nullptr;
  mdarray<double, 2, CblasRowMajor> data__(number_of_samples__, number_of_fp__);

  /* variables needed for the CUB reduction */
  size_t temp_storage_bytes;
  double *temp_storage = NULL;

  data__.allocate(memory_t::device);
  whip::malloc(&scratch__, sizeof(double) * number_of_fp__);

  /* Initializing the cub algorithm */
  cub_reduce(stream__, temp_storage_bytes, temp_storage, number_of_fp__,
             data__.at<device_t::GPU>(0, 0), static_cast<double *>(scratch__));
  whip::malloc(&temp_storage, temp_storage_bytes);

  CreateGenerator(&gen_, 0x514ea3f);

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "Performance tests", 70);

  fmt::print("Measure the time required to execute 100 reductions on {}\n"
             "elements with different reduction algorithms and runtime\n"
             "parameters.\n\n",
             number_of_fp__);

  timer__.start("gen_rng_gpu");
  GenerateUniformDouble(gen_, data__.at<device_t::GPU>(), data__.size());
  timer__.stop("gen_rng_gpu");

  // now a list of variant we want to measure runtime

  std::vector<int> block_size = {64, 128, 256, 512};
  std::vector<int> grid_size;

  grid_size.clear();
  for (int block = 64; block < max_num_blocks__; block *= 2) {
    grid_size.push_back(block);
  }

  mdarray<double, 4, CblasRowMajor> timings_(
      method_to_test.size(), block_size.size(), grid_size.size(), 10);
  mdarray<double, 3, CblasRowMajor> means_(method_to_test.size(),
                                           block_size.size(), grid_size.size());
  mdarray<double, 3, CblasRowMajor> std_dev_(
      method_to_test.size(), block_size.size(), grid_size.size());

  timings_.zero();

  timer__.start("calculate_reduction");

  // Hide cursor
  indicators::show_console_cursor(false);

  fmt::print("Collecting timing data. Be patient\n");
  for (int mt = 0; mt < method_to_test.size(); mt++) {
    std::string log_string = method_to_test[mt].first + " " +
                             std::to_string(mt + 1) + "/" +
                             std::to_string(method_to_test.size());
    bar.set_option(indicators::option::PostfixText{log_string});
    bar.set_progress(mt / method_to_test.size());

    switch (method_to_test[mt].second) {
    case reduction_method::cub_reduce_method: {
      const auto start{std::chrono::steady_clock::now()};
      for (int sample = 0; sample < data__.size(0); sample++) {
        double res_ =
            cub_reduce(stream__, temp_storage_bytes, temp_storage,
                       number_of_fp__, data__.at<device_t::GPU>(sample, 0),
                       static_cast<double *>(scratch__));
      }
      auto stop{std::chrono::steady_clock::now()};
      const std::chrono::duration<double> elapsed_seconds{stop - start};
      for (int bk = 0; bk < block_size.size(); bk++) {
        for (int gs = 0; gs < grid_size.size(); gs++) {
          for (int st = 0; st < 10; st++) {
            timings_(mt, bk, gs, st) = elapsed_seconds.count();
          }
        }
      }
    } break;
    case reduction_method::single_pass_gpu_atomic_only: {
      std::vector<double> tt(10, 0.0);
      for (int st = 0; st < 10; st++) {
        auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res_ = reduce(method_to_test[mt].second, stream__,
                               block_size[0], grid_size[0], number_of_fp__,
                               data__.at<device_t::GPU>(sample, 0),
                               static_cast<double *>(scratch__));
        }
        auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        tt[st] = elapsed_seconds.count();
      }
      for (int bk = 0; bk < block_size.size(); bk++) {
        for (int gs = 0; gs < grid_size.size(); gs++) {
          for (int st = 0; st < 10; st++) {
            timings_(mt, bk, gs, st) = tt[st];
          }
        }
      }
    } break;
    default: {
      for (int bk = 0; bk < block_size.size(); bk++) {
        for (int gs = 0; gs < grid_size.size(); gs++) {
          for (int st = 0; st < 10; st++) {
            const auto start{std::chrono::steady_clock::now()};
            for (int sample = 0; sample < data__.size(0); sample++) {
              double res_ =
                  reduce(method_to_test[mt].second, stream__, block_size[bk],
                         grid_size[gs], number_of_fp__,
                         data__.at<device_t::GPU>(sample, 0),
                         static_cast<double *>(scratch__));
            }
            const auto stop{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> elapsed_seconds{stop - start};
            timings_(mt, bk, gs, st) = elapsed_seconds.count();
          }
        }
      }
    } break;
    }
  }
  bar.set_progress(100);
  // Show cursor
  indicators::show_console_cursor(true);

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "timing results", 70);

  for (int mt = 0; mt < method_to_test.size(); mt++) {
    for (int bk = 0; bk < block_size.size(); bk++) {
      for (int gs = 0; gs < grid_size.size(); gs++) {
        means_(mt, bk, gs) =
            mean(timings_.at<device_t::CPU>(mt, bk, gs, 0), 10);
        std_dev_(mt, bk, gs) =
            std_dev(timings_.at<device_t::CPU>(mt, bk, gs, 0), 10);
      }
    }
  }
  timer__.stop("calculate_reduction");
  whip::free(temp_storage);
  whip::free(scratch__);

  std::string filename_ = "performance_results_" + GPU_TYPE + ".csv";
  FILE *f = fopen(filename_.c_str(), "w+");
  for (int mt = 0; mt < method_to_test.size(); mt++) {
    for (int bk = 0; bk < block_size.size(); bk++) {
      for (int gs = 0; gs < grid_size.size(); gs++) {
        fprintf(f, "%s,%d,%d,%.15lf,%.15lf\n", method_to_test[mt].first.c_str(),
                block_size[bk], grid_size[gs], means_(mt, bk, gs),
                std_dev_(mt, bk, gs));
      }
    }
  }
  fclose(f);
}

void cross_platform_reproducibility(cxxopts::ParseResult &result,
                                    rt_graph::Timer &timer__,
                                    whip::stream_t &stream__) {
  // Use Mersenne twister engine to generate pseudo-random numbers.
  std::mt19937 engine{0x1237698};
  const int table_length__ = 1000000;
  const int number_of_samples__ = 100;
  void *scratch__ = nullptr;

  // "Filter" MT engine's output to generate pseudo-random double values,
  // **uniformly distributed** on the closed interval [0, 1].
  // (Note that the range is [inclusive, inclusive].)

  std::uniform_real_distribution<double> dist_uniform(-5.0, 5.0);

  std::normal_distribution<double> dist_normal(
      result["distribution_center"].as<double>(),
      result["standard_deviation"].as<double>());

  /* We use the CPU for this to check that we get reproducable results on
     different platforms */

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "Cross platform reproducibility", 70);

  fmt::print(
      "Check that the deterministic version of the reduce function gives the\n"
      "same answer on all platforms. We use mersenne twister from the C++\n"
      "standard to generate the same random sequence on all platform.\n\n We\n"
      "generate 100 samples of 100000 uniformally distributed floating points\n"
      "numbers between [-5,5], and compute the associated sum on GPU using\n "
      "the tree reduction algoritm based on shared memory. It is the most\n"
      "portable code and independent of any particular GPU specific "
      "optimization.\n\n");

  mdarray<double, 2, CblasRowMajor> data__(number_of_samples__, table_length__);
  data__.allocate(memory_t::device);
  whip::malloc(&scratch__, sizeof(double) * table_length__);

  timer__.start("gen_rng_cpu");
  generate_random_numbers_cpu("uniform", engine, dist_uniform, dist_normal,
                              data__);
  timer__.stop("gen_rng_cpu");

  timer__.start("calculate_reduction");
  data__.copy<memory_t::host, memory_t::device>();

  std::vector<uint64_t> res(data__.size(0));
  //  std::cout << std::hex << "{" << std::endl;
  for (int sample = 0; sample < data__.size(0); sample++) {
    double res_ =
        reduce(reduction_method::two_pass_gpu_det_kahan_cpu, stream__, 64, -1,
               data__.size(1), data__.at<device_t::GPU>(sample, 0),
               static_cast<double *>(scratch__));
    std::memcpy(&res[sample], &res_, sizeof(double));
    //    std::cout << "0x" << res[sample] <<"," << std::endl;
  }
  //  std::cout << "}" << std::endl;
  timer__.stop("calculate_reduction");
  bool check_ok = true;
  for (int sample = 0; (sample < data__.size(0)) && check_ok; sample++) {
    if (res[sample] != reference_values_v100[sample]) {
      fmt::print("Sample {0}: Computed value: {1:#x}, reference value: {2:#x}, "
                 "value (FP): {3:f}, ref value: {4:f}\n",
                 sample, res[sample], reference_values_v100[sample],
                 ((double *)&res[sample])[0],
                 ((double *)&reference_values_v100[sample])[0]);
      check_ok = false;
    }
  }

  if (!check_ok) {
    fmt::print("The reduction algorithm using the tree reduction is not "
               "deterministic across different GPU family\n");
  }

  fmt::print("shared mem two steps: OK\n");

  for (int sample = 0; sample < data__.size(0); sample++) {
    double res_ =
        reduce(reduction_method::two_pass_gpu_det_shuffle_kahan_cpu, stream__,
               64, -1, data__.size(1), data__.at<device_t::GPU>(sample, 0),
               static_cast<double *>(scratch__));
    std::memcpy(&res[sample], &res_, sizeof(double));
  }
  check_ok = true;
  for (int sample = 0; (sample < data__.size(0)) && check_ok; sample++) {
    if (res[sample] != reference_values_v100[sample]) {
      fmt::print("Sample {0}: Computed value: {1:#x}, reference value: {2:#x}, "
                 "value (FP): {3:f}, ref value: {4:f}\n",
                 sample, res[sample], reference_values_v100[sample],
                 ((double *)&res[sample])[0],
                 ((double *)&reference_values_v100[sample])[0]);
      check_ok = false;
    }
  }

  if (!check_ok) {
    fmt::print("The reduction algorithm using the tree reduction is not "
               "deterministic across different GPU family\n");
  }

  fmt::print("shuffle variant: OK\n");
  timer__.stop("calculate_reduction");
  data__.clear();
  whip::free(scratch__);
}

void generate_variability_distribution(
    cxxopts::ParseResult &result, rt_graph::Timer &timer__,
    std::vector<whip::stream_t> &stream_vector_) {
  generator_t gen;
  double *scratch_;
  mdarray<double, 2, CblasRowMajor> data_;
  const bool variable_length_reduction_ = !result["atomic"].as<bool>();
  const int number_of_gpus = 1;
  const int average_length = 10;
  const int number_of_samples = result["number_of_samples"].as<int>();
  const int max_number_of_elements = result["max_reduction_size"].as<int>();
  const int min_number_of_elements = result["min_reduction_size"].as<int>();
  const int number_of_iterations = number_of_samples / (number_of_gpus * 100);
  const int max_blocks = 2 * (max_number_of_elements + 64 - 1) / 64;
  std::vector<int> number_of_elements_per_reduction;
  for (int j = min_number_of_elements; j <= max_number_of_elements; j *= 10) {
    number_of_elements_per_reduction.push_back(j);
  }

  if (number_of_elements_per_reduction.back() < max_number_of_elements)
    number_of_elements_per_reduction.push_back(max_number_of_elements);

  mdarray<double, 2, CblasRowMajor> result_stat_ =
      mdarray<double, 2, CblasRowMajor>(
          number_of_samples, number_of_elements_per_reduction.size());

  data_ = mdarray<double, 2, CblasRowMajor>(100, max_number_of_elements);
  whip::malloc(&scratch_, sizeof(double) * 64 * max_blocks);
  data_.allocate(memory_t::device);

  CreateGenerator(&gen, result["seed"].as<unsigned int>());

  std::vector<double> min_(number_of_elements_per_reduction.size(), 0.0);
  std::vector<double> max_(number_of_elements_per_reduction.size(), 0.0);
  mdarray<double, 2, CblasRowMajor> det_sum_(
      data_.size(0), number_of_elements_per_reduction.size());

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "test2!", 70);

  fmt::print("This function computes the scalar variability from a\n"
             "set of 100 lists of various lengths ranging from {0:d} to\n"
             "{1:d}. We use the single pass with block reduction for the\n"
             "deterministic method and the single pass with atomicAdd for\n"
             "the non deterministic variant. The only difference between\n"
             "the two methods is the last stage.\n",
             result["min_reduction_size"].as<int>(),
             result["max_reduction_size"].as<int>());

  timer__.start("sampling");
  for (int sp = 0; sp < number_of_iterations; sp++) {
    if (sp == 0) {
      timer__.start("generate_rng");
      generate_random_numbers(result["distribution_type"].as<std::string>(),
                              gen,
                              result["distribution_amplitude"].as<double>(),
                              result["distribution_center"].as<double>(),
                              result["standard_deviation"].as<double>(),
                              data_.at<device_t::GPU>(), data_.size());
      timer__.stop("generate_rng");

      timer__.start("reduction_compute_ref");
      for (int step = 0; step < number_of_elements_per_reduction.size();
           step++) {
        for (int sample = 0; sample < data_.size(0); sample++) {
          // compute the reference version using deterministic summation
          double result_reference_ =
              reduce(reduction_method::single_pass_gpu_det_shared,
                     stream_vector_[step], 64, -1,
                     number_of_elements_per_reduction[step],
                     data_.at<device_t::GPU>(sample, 0),
                     static_cast<double *>(scratch_) + step * max_blocks);
          det_sum_(sample, step) = result_reference_;
        }
      }
      timer__.stop("reduction_compute_ref");
    }
#ifdef REDUCE_USE_CUDA
    cudaThreadSynchronize();
#endif

    // compute the references values for all reductions using deterministic
    // algorithm
    timer__.start("block_reduction");

    // #pragma omp parallel for
    for (int step = 0; step < number_of_elements_per_reduction.size(); step++) {
      for (int sample = 0; sample < data_.size(0); sample++) {
        // the reference version is computed right after generating the random
        // sequences.

        // compute the non deterministic summation. Note we do *not* optimize
        // this version for performance, we maximize the use of atomic
        // operations. For a table of length l, we use (l + 64 - 1) / 64
        // atomic ops. We can use much less atomics here

        double result_non_det_ =
            reduce(reduction_method::single_pass_gpu_shared_atomic,
                   stream_vector_[step], 64, -1,
                   number_of_elements_per_reduction[step],
                   data_.at<device_t::GPU>(sample, 0),
                   static_cast<double *>(scratch_));
        result_stat_(sp * data_.size(0) + sample, step) =
            (result_non_det_ - det_sum_(sample, step)) / det_sum_(sample, step);
        min_[step] = std::min(min_[step],
                              result_stat_(sp * data_.size(0) + sample, step));
        max_[step] = std::max(max_[step],
                              result_stat_(sp * data_.size(0) + sample, step));
      }
    }
    timer__.stop("block_reduction");

    if (((sp % (number_of_iterations / 100)) == 0) && (sp != 0)) {
      std::cout << "Sample #" << sp * number_of_gpus * 100 << std::endl;
      for (int step = 0; step < number_of_elements_per_reduction.size();
           step++) {
        fmt::print("Deviation for a reduction of a table with {:7d} "
                   "elements: min: {}, max: {}\n",
                   number_of_elements_per_reduction[step], min_[step],
                   max_[step]);
      }
    }
  }

  timer__.stop("sampling");
  std::string filename_ = "relative_error_block_reduction_";

  if (result["distribution_type"].as<std::string>() == "uniform") {
    filename_ += "uniform_";
    if (std::abs(result["distribution_center"].as<double>()) > 1e-8)
      filename_ += "centered_";
  } else {
    filename_ += "normal_";
  }

  filename_ += GPU_TYPE + ".csv";
  FILE *f = fopen(filename_.c_str(), "w+");
  for (int step = 0; step < number_of_elements_per_reduction.size(); step++) {
    fprintf(f, "%d,", number_of_elements_per_reduction[step]);
    for (int sample = 0; sample < number_of_samples - 1; sample++)
      fprintf(f, "%.16e,", result_stat_(sample, step));
    fprintf(f, "%.16e\n", result_stat_(number_of_samples - 1, step));
  }
  fclose(f);

  if (result["atomic_only"].as<bool>()) {
    std::fill(min_.begin(), min_.end(), 0);
    std::fill(max_.begin(), max_.end(), 0);

    for (int sp = 0; sp < number_of_iterations; sp++) {
      timer__.start("Reduce_atomic_add_only");
      for (int step = 0; step < number_of_elements_per_reduction.size();
           step++) {
        for (int sample = 0; sample < data_.size(0); sample++) {
          double result_non_det_ =
              reduce(reduction_method::single_pass_gpu_atomic_only,
                     stream_vector_[step], 64, -1,
                     number_of_elements_per_reduction[step],
                     data_.at<device_t::GPU>(sample, 0),
                     static_cast<double *>(scratch_));
          result_stat_(sp * data_.size(0) + sample, step) =
              (result_non_det_ - det_sum_(sample, step)) /
              det_sum_(sample, step);
          min_[step] = std::min(
              min_[step], result_stat_(sp * data_.size(0) + sample, step));
          max_[step] = std::max(
              max_[step], result_stat_(sp * data_.size(0) + sample, step));
        }
      }
      timer__.stop("Reduce_atomic_add_only");
      if (((sp % (number_of_iterations / 100)) == 0) && (sp != 0)) {
        std::cout << "Sample #" << sp * number_of_gpus * 100 << std::endl;
        for (int step = 0; step < number_of_elements_per_reduction.size();
             step++) {
          fmt::print("Deviation for a reduction of a table with {:7d} "
                     "elements: min: {}, max: {}\n",
                     number_of_elements_per_reduction[step], min_[step],
                     max_[step]);
        }
      }
    }
    std::string filename_ = "relative_error_atomicAdd_only_";
    if (result["distribution_type"].as<std::string>() == "uniform") {
      filename_ += "uniform_";
      if (std::abs(result["distribution_center"].as<double>()) > 1e-8)
        filename_ += "centered_";
    } else {
      filename_ += "normal_";
    }

    filename_ += GPU_TYPE + ".csv";
    FILE *f = fopen(filename_.c_str(), "w+");
    for (int step = 0; step < number_of_elements_per_reduction.size(); step++) {
      fprintf(f, "%d,", number_of_elements_per_reduction[step]);
      for (int sample = 0; sample < number_of_samples - 1; sample++)
        fprintf(f, "%.16e,", result_stat_(sample, step));
      fprintf(f, "%.16e\n", result_stat_(number_of_samples - 1, step));
    }
    fclose(f);
  }

  DestroyGenerator(gen);
  data_.clear();
  result_stat_.clear();
  min_.clear();
  max_.clear();
  whip::free(scratch_);
}

int main(int argc, char **argv) {
  rt_graph::Timer timer_;

  cxxopts::Options options("reduce_test",
                           "Compute the relative error between the "
                           "deterministic and atomic sums on GPUs");
  options.add_options()("S,number_of_samples",
                        "Number of times to compute the relative error",
                        cxxopts::value<int>()->default_value("1000"))(
      "max_reduction_size", "Maximum number of elements for each reduction",
      cxxopts::value<int>()->default_value("100000"))(
      "min_reduction_size", "minimum number of elements for each reduction",
      cxxopts::value<int>()->default_value("10"))(
      "A,distribution_amplitude", "Amplitude of the distribution",
      cxxopts::value<double>()->default_value("10.0"))(
      "atomic_only",
      "Compute the sum using only atomic operations. Only used to compute "
      "the "
      "distribution. Very slow",
      cxxopts::value<bool>()->default_value("false"))(
      "c,distribution_center",
      "center of the distribution. The default value for the uniform "
      "distribution will return )0, 1]",
      cxxopts::value<double>()->default_value("0.0"))(
      "d,distribution_type", "type of distribution, normal, uniform",
      cxxopts::value<std::string>()->default_value("uniform"))(
      "s,number_of_streams", "number of streams to use for the computations",
      cxxopts::value<int>()->default_value("1"))(
      "r,seed", "seed for the random number generator",
      cxxopts::value<unsigned int>()->default_value("0xa238abdf"))(
      "a,atomic",
      "Keep the table length constant but change the number of atomic ops",
      cxxopts::value<bool>()->default_value("false"))(
      "standard_deviation", "Standard deviation for the normal distribution",
      cxxopts::value<double>()->default_value("0.5"))("h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "Information about the simulations", 70);

  fmt::print("{0:<40} {1:>30}\n", "RNG:", "XORWOW");
  fmt::print("{0:<40} {1:>30}\n",
             "Distribution:", result["distribution_type"].as<std::string>());
  fmt::print("{0:<40} {1:>30}\n", "Distribution amplitude:",
             result["distribution_amplitude"].as<double>());
  if (result["distribution_type"].as<std::string>() == "normal")
    fmt::print("{0:<40} {1:>30}\n",
               "Std dev.:", result["standard_deviation"].as<double>());
  fmt::print("{0:<40} {1:>30}\n", "Distribution center:",
             result["distribution_center"].as<double>());
  fmt::print("{0:<40} {1:>30}\n",
             "Max reduction size:", result["max_reduction_size"].as<int>());
  fmt::print("{0:<40} {1:>30}\n",
             "Min reduction size:", result["min_reduction_size"].as<int>());
  fmt::print("{0:<40} {1:>30}\n",
             "Number of samples:", result["number_of_samples"].as<int>());
  fmt::print("\n");
#ifdef REDUCE_USE_CUDA
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  switch (prop.major) {
  case 7:
    GPU_TYPE = "V100";
    break;
  case 8:
    GPU_TYPE = "A100";
    break;
  case 9:
    GPU_TYPE = "GH200";
    break;
  default:
    GPU_TYPE = "V100";
    break;
  }
#else
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  GPU_TYPE = "Mi250X";
#endif

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "Device properties", 70);

  fmt::print("{0:<40} {1:>30}\n", "Device name:", prop.name);
  fmt::print("{0:<40} {1:>30}\n", "Architecture:", GPU_TYPE);
  fmt::print("{0:<40} {1:>30}\n", "warp size:", prop.warpSize);
  fmt::print("\n");

#ifdef REDUCE_USE_CUDA
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_shuffle",
      reduction_method::single_pass_gpu_det_shuffle));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_shuffle_recursive",
      reduction_method::single_pass_gpu_det_shuffle_recursive_gpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_shuffle_kahan",
      reduction_method::single_pass_gpu_det_shuffle_kahan_gpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_shuffle_atomic",
      reduction_method::single_pass_gpu_shuffle_atomic));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "two_pass_gnu_det_shuffle_kahan_cpu",
      reduction_method::two_pass_gpu_det_shuffle_kahan_cpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "two_pass_gnu_det_shuffle_recursive_cpu",
      reduction_method::two_pass_gpu_det_shuffle_recursive_cpu));
#endif
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_shared", reduction_method::single_pass_gpu_det_shared));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_kahan",
      reduction_method::single_pass_gpu_det_kahan_gpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_det_recursive",
      reduction_method::single_pass_gpu_det_recursive_gpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "single_pass_gpu_shared_atomic",
      reduction_method::single_pass_gpu_shared_atomic));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "two_pass_gnu_det_recursive_cpu",
      reduction_method::two_pass_gpu_det_recursive_cpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "two_pass_gnu_det_kahan_cpu",
      reduction_method::two_pass_gpu_det_kahan_cpu));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "cub_reduce", reduction_method::cub_reduce_method));
  method_to_test.push_back(std::make_pair<std::string, reduction_method>(
      "atomic_only", reduction_method::single_pass_gpu_atomic_only));

  const bool variable_length_reduction_ = !result["atomic"].as<bool>();
  const int number_of_gpus = 1;
  const int average_length = 10;
  const int number_of_samples = result["number_of_samples"].as<int>();
  const int max_number_of_elements = result["max_reduction_size"].as<int>();
  const int min_number_of_elements = result["min_reduction_size"].as<int>();
  const int number_of_iterations = number_of_samples / (number_of_gpus * 100);
  const int max_blocks = 2 * (max_number_of_elements + 64 - 1) / 64;
  std::vector<int> number_of_elements_per_reduction;
  for (int j = min_number_of_elements; j <= max_number_of_elements; j *= 10) {
    number_of_elements_per_reduction.push_back(j);
  }

  if (number_of_elements_per_reduction.back() < max_number_of_elements)
    number_of_elements_per_reduction.push_back(max_number_of_elements);

  std::vector<whip::stream_t> stream_vector_(
      number_of_elements_per_reduction.size());

  for (auto &stream_ : stream_vector_)
    whip::stream_create(&stream_);

  sanity_check(stream_vector_[0]);

  /* Check determinism across potentially different GPU */
  timer_.start("cross_platform_reproducibility");
  cross_platform_reproducibility(result, timer_, stream_vector_[0]);
  timer_.stop("cross_platform_reproducibility");

  timer_.start("performance_test");
  performance_tests(result, timer_, stream_vector_[0]);
  timer_.stop("performance_test");
  /*
    Compare the deterministic vs non-deterministic reduction algorithm for
    reduction of different sizes. the deterministic variant uses a fixed block
    size of 64, while the grid size is computed based on it.
  */

  timer_.start("generate_variability_distribution");
  generate_variability_distribution(result, timer_, stream_vector_);
  timer_.stop("generate_variability_distribution");

  auto timing_result = timer_.process();
  std::cout << timing_result.print(
      {rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
       rt_graph::Stat::Median, rt_graph::Stat::Min, rt_graph::Stat::Max});

  for (auto stream_ : stream_vector_)
    whip::stream_destroy(stream_);

  stream_vector_.clear();
  return 0;
}
