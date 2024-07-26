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

  const double result_of_sum = 2.506628274631001;
  fmt::print("Sanity checks. Computing the numerical integral of exp(-x^2/2) between -10 and 10\n");
  fmt::print("Result of the sum with Erf function: {}\n", result_of_sum);
  fmt::print("Checking Kahan sum:                  {:.15f}\n", KahanBabushkaNeumaierSum(sum_.at<device_t::CPU>(), sum_.size())  * 0.001);
  fmt::print("Checking det sum on GPU:             {:.15f}\n", reduce_step_det<double>(stream_,
                                                                        256,
                                                                        -1, sum_.size(),
                                                                        sum_.at<device_t::GPU>(), scratch__) * 0.001);
  fmt::print("Checking non-det sum on GPU:         {:.15f}\n", reduce_step_non_det<double>(stream_,
                                                                          256,
                                                                          -1, sum_.size(),
                                                                          sum_.at<device_t::GPU>(), scratch__) * 0.001);
  fmt::print("Checking det-shuffle sum on GPU:     {:.15f}\n", reduce_step_det_shuffle<double>(stream_,
                                                                                  256,
                                                                                  -1, sum_.size(),
                                                                                  sum_.at<device_t::GPU>(), scratch__) * 0.001);
  fmt::print("Checking full atomic sum on GPU:     {:.15f}\n", reduce_atomic<double>(stream_, sum_.size(),
                                                                                               sum_.at<device_t::GPU>(), scratch__) * 0.001);

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
      indicators::option::PostfixText{"Performance tests"},
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

  mdarray<double, 4, CblasRowMajor> timings_(6, 4, 17, 10);

  timings_.zero();
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
             "", "Performance tests", 60);

  fmt::print("Measure the time required to execute 100 reductions on 1 million "
             "elements with different reduction algorithms and runtime "
             "parameters.\n\n");

  timer__.start("gen_rng_gpu");
  GenerateUniformDouble(gen_, data__.at<device_t::GPU>(), data__.size());
  timer__.stop("gen_rng_gpu");

  timer__.start("calculate_reduction");

  // Hide cursor
  indicators::show_console_cursor(false);

  fmt::print("Collecting timing data. Be patient\n");
  int bk = 0;

  bar.set_progress(0);
  bar.set_option(indicators::option::PostfixText{
      "Two steps deterministic algorithm 1 / 6"});

  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    int bi;
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int st = 0; st < 10; st++) {
        const auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res_ =
              reduce_step_det(stream__, threadBlock, block, number_of_fp__,
                              data__.at<device_t::GPU>(sample, 0),
                              static_cast<double *>(scratch__));
        }
        const auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        timings_(0, bk, bi, st) = elapsed_seconds.count();
      }
    }
  }

  bar.set_progress(17);
  bar.set_option(indicators::option::PostfixText{
      "single step deterministic algorithm 2 / 6"});
  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    int bi;
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int st = 0; st < 10; st++) {
        const auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res_ = reduce_step_det_single_round(
              stream__, threadBlock, block, number_of_fp__,
              data__.at<device_t::GPU>(sample, 0),
              static_cast<double *>(scratch__));
        }
        const auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        timings_(1, bk, bi, st) = elapsed_seconds.count();
      }
    }
  }
  bar.set_progress(33);
  bar.set_option(indicators::option::PostfixText{
      "two step deterministic algorithm with shuffle: 3 / 6"});
  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    int bi;
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int st = 0; st < 10; st++) {
        const auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res_ = reduce_step_det_shuffle(
              stream__, threadBlock, block, number_of_fp__,
              data__.at<device_t::GPU>(sample, 0),
              static_cast<double *>(scratch__));
        }
        const auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        timings_(2, bk, bi, st) = elapsed_seconds.count();
      }
    }
  }

  bar.set_progress(50);
  bar.set_option(indicators::option::PostfixText{
      "single step deterministic algorithm with shuffle: 4 / 6"});

  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    int bi;
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int st = 0; st < 10; st++) {

        const auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res_ = reduce_step_det_shuffle_single_round(
              stream__, threadBlock, block, number_of_fp__,
              data__.at<device_t::GPU>(sample, 0),
              static_cast<double *>(scratch__));
        }
        const auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        timings_(3, bk, bi, st) = elapsed_seconds.count();
      }
    }
  }

  bar.set_progress(67);
  bar.set_option(indicators::option::PostfixText{
      "two steps non deterministic algorithm with shuffle: 5 / 6"});

  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    int bi;
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int st = 0; st < 10; st++) {
        const auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res_ =
              reduce_step_non_det(stream__, threadBlock, block, number_of_fp__,
                                  data__.at<device_t::GPU>(sample, 0),
                                  static_cast<double *>(scratch__));
        }
        const auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        timings_(4, bk, bi, st) = elapsed_seconds.count();
      }
    }
  }

  bar.set_progress(84);
  bar.set_option(indicators::option::PostfixText{"CUB reduce: 6 / 6"});
  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int st = 0; st < 10; st++) {
        const auto start{std::chrono::steady_clock::now()};
        for (int sample = 0; sample < data__.size(0); sample++) {
          double res__ =
              cub_reduce(stream__, temp_storage_bytes, temp_storage,
                         number_of_fp__, data__.at<device_t::GPU>(sample, 0),
                         static_cast<double *>(scratch__));
        }
        const auto stop{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{stop - start};
        timings_(5, bk, bi, st) = elapsed_seconds.count();
      }
    }
  }
  bar.set_progress(100);
  // Show cursor
  indicators::show_console_cursor(true);

  // double res_kahan__ = KahanBabushkaNeumaierSum(data__.at<device_t::CPU>(0,
  // 0), 1000000);

  // double res_ = reduce_step_det_shuffle_single_round(stream__, 512, -1,
  // 1000000,
  //                                                   data__.at<device_t::GPU>(0,
  //                                                   0), static_cast<double
  //                                                   *>(scratch_));

  // double res1__ = cub_reduce(stream__, temp_storage_bytes, temp_storage,
  // 1000000, data__.at<device_t::GPU>(0, 0), static_cast<double *>(scratch_));

  // uint64_t res11__, res12__, res13__;
  // std::memcpy(&res11__, &res_, sizeof(double));
  // std::memcpy(&res12__, &res_kahan__, sizeof(double));
  // std::memcpy(&res13__, &res1__, sizeof(double));
  // fmt::print("Sum reduce perso {0:#x}  {1:#x} cub {2:#x}\n", res11__,
  // res12__, res13__);

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "timing results", 60);

  fmt::print("#thread #block    det    det_single_pass     det_shuffle     "
             "det_shuffle_single_pass "
             "    non-det      cub\n");

  std::vector<double> mean_(6);
  std::vector<double> std_dev_(6);

  for (int threadBlock = 64, bk = 0; threadBlock < 1024;
       threadBlock *= 2, bk++) {
    int bi;
    for (int block = 2, bi = 0; block < max_num_blocks__; block *= 2, bi++) {
      for (int i = 0; i < 6; i++) {
        mean_[i] = mean(timings_.at<device_t::CPU>(i, bk, bi, 0), 10);
        std_dev_[i] = std_dev(timings_.at<device_t::CPU>(i, bk, bi, 0), 10);
      }
      fmt::print("{} {} {:9.9} {:9.9} {:9.9} {:9.9} {:9.9} {:9.9} {:9.9} "
                 "{:9.9} {:9.9} {:9.9} {:9.9} {:9.9}\n",
                 threadBlock, block, mean_[0], std_dev_[0], mean_[1],
                 std_dev_[1], mean_[2], std_dev_[2], mean_[3], std_dev_[3],
                 mean_[4], std_dev_[4], mean_[5], std_dev_[5]);
    }
  }
  timer__.stop("calculate_reduction");
  whip::free(temp_storage);
  whip::free(scratch__);
  FILE *f = fopen("performance_results.dat", "w+");
  fwrite(timings_.at<device_t::CPU>(), sizeof(double), timings_.size(), f);
  fclose(f);
  timings_.clear();
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
             "", "Test 1!", 60);

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
    double res_ = reduce_step_det(stream__, 64, -1, data__.size(1),
                                  data__.at<device_t::GPU>(sample, 0),
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
    double res_ = reduce_step_det_shuffle(stream__, 64, -1, data__.size(1),
                                          data__.at<device_t::GPU>(sample, 0),
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

void test2(cxxopts::ParseResult &result, rt_graph::Timer &timer__,
           std::vector<whip::stream_t> &stream_vector_, generator_t gen,
           void *scratch_, mdarray<double, 2, CblasRowMajor> &data__) {
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

  std::vector<double> min_(number_of_elements_per_reduction.size(), 0.0);
  std::vector<double> max_(number_of_elements_per_reduction.size(), 0.0);
  mdarray<double, 2, CblasRowMajor> det_sum_(
      data__.size(0), number_of_elements_per_reduction.size());

  fmt::print("┌{0:─^{2}}┐\n"
             "│{1: ^{2}}│\n"
             "└{0:─^{2}}┘\n\n",
             "", "test2!", 60);

  fmt::print("This test generates samples lists of various length and sums up\n"
             "the lists using two tree reduction algorithms, one using\n"
             "deterministic implementation on CPU as a last stage, the other using\n"
             "the atomicAdd instruction. The relative error between the two\n"
             "algorithm is stored in a file for further analysis.\n\n");
  timer__.start("sampling");
  for (int sp = 0; sp < number_of_iterations; sp++) {
    if (sp == 0) {
      timer__.start("generate_rng");
      generate_random_numbers(result["distribution_type"].as<std::string>(),
                              gen,
                              result["distribution_amplitude"].as<double>(),
                              result["distribution_center"].as<double>(),
                              result["standard_deviation"].as<double>(),
                              data__.at<device_t::GPU>(), data__.size());
      timer__.stop("generate_rng");

      timer__.start("reduction_compute_ref");
      for (int step = 0; step < number_of_elements_per_reduction.size();
           step++) {
        for (int sample = 0; sample < data__.size(0); sample++) {
          // compute the reference version using deterministic summation
          double result_reference_ = reduce_step_det(
              stream_vector_[step], 64, -1,
              number_of_elements_per_reduction[step],
              data__.at<device_t::GPU>(sample, 0),
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
      for (int sample = 0; sample < data__.size(0); sample++) {
        // the reference version is computed right after generating the random
        // sequences.

        // compute the non deterministic summation. Note we do *not* optimize
        // this version for performance, we maximize the use of atomic
        // operations. For a table of length l, we use (l + 64 - 1) / 64
        // atomic ops. We can use much less atomics here

        double result_non_det_ =
            reduce_step_non_det(stream_vector_[step], 64, -1,
                                number_of_elements_per_reduction[step],
                                data__.at<device_t::GPU>(sample, 0),
                                static_cast<double *>(scratch_));
        result_stat_(sp * data__.size(0) + sample, step) =
            (result_non_det_ - det_sum_(sample, step)) / det_sum_(sample, step);
        min_[step] = std::min(min_[step],
                              result_stat_(sp * data__.size(0) + sample, step));
        max_[step] = std::max(max_[step],
                              result_stat_(sp * data__.size(0) + sample, step));
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
  if (!result["csv_file"].as<bool>()) {
    FILE *f = fopen("relative_error_block_reduction.dat", "w+");
    fwrite(result_stat_.at<device_t::CPU>(), sizeof(double),
           result_stat_.size(), f);
    fclose(f);
  } else {
    FILE *f = fopen("relative_error_block_reduction.csv", "w+");
    for (int step = 0; step < number_of_elements_per_reduction.size(); step++) {
      fprintf(f, "%d,", number_of_elements_per_reduction[step]);
      for (int sample = 0; sample < number_of_samples - 1; sample++)
        fprintf(f, "%.16e,", result_stat_(sample, step));
      fprintf(f, "%.16e\n", result_stat_(number_of_samples - 1, step));
    }
    fclose(f);
  }

  if (result["atomic_only"].as<bool>()) {
    std::fill(min_.begin(), min_.end(), 0);
    std::fill(max_.begin(), max_.end(), 0);

    for (int sp = 0; sp < number_of_iterations; sp++) {
      timer__.start("Reduce_atomic_add_only");
      for (int step = 0; step < number_of_elements_per_reduction.size();
           step++) {
        for (int sample = 0; sample < data__.size(0); sample++) {
          double result_non_det_ = reduce_atomic(
              stream_vector_[step], number_of_elements_per_reduction[step],
              data__.at<device_t::GPU>(sample, 0),
              static_cast<double *>(scratch_));
          result_stat_(sp * data__.size(0) + sample, step) =
              (result_non_det_ - det_sum_(sample, step)) /
              det_sum_(sample, step);
          min_[step] = std::min(
              min_[step], result_stat_(sp * data__.size(0) + sample, step));
          max_[step] = std::max(
              max_[step], result_stat_(sp * data__.size(0) + sample, step));
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
    if (!result["csv_file"].as<bool>()) {
      FILE *f = fopen("relative_error_atomicAdd_only.dat", "w+");
      fwrite(result_stat_.at<device_t::CPU>(), sizeof(double),
             result_stat_.size(), f);
      fclose(f);
    } else {
      FILE *f = fopen("relative_error_atomicAdd_only.csv", "w+");
      for (int step = 0; step < number_of_elements_per_reduction.size();
           step++) {
        fprintf(f, "%d,", number_of_elements_per_reduction[step]);
        for (int sample = 0; sample < number_of_samples - 1; sample++)
          fprintf(f, "%.16e,", result_stat_(sample, step));
        fprintf(f, "%.16e\n", result_stat_(number_of_samples - 1, step));
      }
      fclose(f);
    }
  }
}

void test3(cxxopts::ParseResult &result, rt_graph::Timer &timer__,
           std::vector<whip::stream_t> &stream_vector_, generator_t gen,
           void *scratch_, mdarray<double, 2, CblasRowMajor> &data__) {
  // time the operation
  // rt_graph::Timer timer_;
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

  std::vector<double> det_sum__(data__.size(0), 0.0);
  std::vector<double> results__(10000);

  mdarray<double, 3, CblasRowMajor> stats_(11, 5, average_length);
  for (int stat = 0; stat < average_length; stat++) {
    int round = 0;
    generate_random_numbers(result["distribution_type"].as<std::string>(), gen,
                            result["distribution_amplitude"].as<double>(),
                            result["distribution_center"].as<double>(),
                            result["standard_deviation"].as<double>(),
                            data__.at<device_t::GPU>(), data__.size());
    for (int num_atomic_ops = 4;
         num_atomic_ops < (max_number_of_elements + 128 - 1) / 128;
         num_atomic_ops *= 2) {
      for (int sample = 0; sample < data__.size(0); sample++) {
        det_sum__[sample] =
            reduce_step_det(stream_vector_[0], 64, num_atomic_ops,
                            data__.size(1), data__.at<device_t::GPU>(sample, 0),
                            static_cast<double *>(scratch_));
      }
      for (int it = 0; it < 10000; it += data__.size(0)) {
        for (int sample = 0; sample < data__.size(0); sample++) {
          const double result = reduce_step_non_det(
              stream_vector_[0], 64, num_atomic_ops, data__.size(1),
              data__.at<device_t::GPU>(sample, 0),
              static_cast<double *>(scratch_));
          results__[it + sample] =
              (1.0e16) * (det_sum__[sample] - result) / det_sum__[sample];
        }
      }
      stats_(round, 0, stat) = num_atomic_ops;
      stats_(round, 1, stat) = mean<double>(results__);
      stats_(round, 2, stat) = std_dev<double>(results__);
      stats_(round, 3, stat) = min<double>(results__);
      stats_(round, 4, stat) = max<double>(results__);

      fmt::print("{} {} {} {} {}\n", num_atomic_ops, mean<double>(results__),
                 std_dev<double>(results__), min<double>(results__),
                 max<double>(results__));
      round++;
    }
  }

  fmt::print("Final answer\n\n");

  for (int s = 0; s < stats_.size(0); s++) {
    std::vector<double> ts_(stats_.at<device_t::CPU>(s, 3, 0),
                            stats_.at<device_t::CPU>(s, 3, average_length - 1));
    fmt::print("{} {} {} {} {}\n", stats_(s, 0, 0), mean<double>(ts_),
               std_dev<double>(ts_), min(ts_), max(ts_));
  }
}

int main(int argc, char **argv) {
  generator_t gen;

  double *scratch_;
  mdarray<double, 2, CblasRowMajor> data_;
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
      "Compute the sum using only atomic operations. Only used to compute the "
      "distribution. Very slow",
      cxxopts::value<bool>()->default_value("false"))(
      "c,distribution_center",
      "center of the distribution. The default value for the uniform "
      "distribution will return )0, 1]",
      cxxopts::value<double>()->default_value("0.0"))(
      "d,distribution_type", "type of distribution, normal, uniform",
      cxxopts::value<std::string>()->default_value("uniform"))(
      "s,number_of_streams", "number of streams to use for the computations",
      cxxopts::value<int>()->default_value("4"))(
      "C,csv_file", "output results in the csv format",
      cxxopts::value<bool>()->default_value("true"))(
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
             "", "Information about the simulations", 60);

  fmt::print("- RNG: XORWOW\n");
  fmt::print("- Distribution: {}\n",
             result["distribution_type"].as<std::string>());
  fmt::print("- Distribution amplitude: {}\n",
             result["distribution_amplitude"].as<double>());
  if (result["distribution_type"].as<std::string>() == "normal")
    fmt::print("- Std dev.: {}\n", result["standard_deviation"].as<double>());
  fmt::print("- Distribution center: {}\n",
             result["distribution_center"].as<double>());
  fmt::print("- Max reduction size: {}\n",
             result["max_reduction_size"].as<int>());
  fmt::print("- Min reduction size: {}\n",
             result["min_reduction_size"].as<int>());
  fmt::print("- Number of samples: {}\n",
             result["number_of_samples"].as<int>());

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

  data_ = mdarray<double, 2, CblasRowMajor>(100, max_number_of_elements);
  whip::malloc(&scratch_, sizeof(double) * 64 * max_blocks);
  data_.allocate(memory_t::device);

  CreateGenerator(&gen, result["seed"].as<unsigned int>());

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

  timer_.start("test_2");
  test2(result, timer_, stream_vector_, gen, scratch_, data_);
  timer_.stop("test_2");

  timer_.start("test_3");
  test3(result, timer_, stream_vector_, gen, scratch_, data_);
  timer_.stop("test_3");

  auto timing_result = timer_.process();
  std::cout << timing_result.print(
      {rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
       rt_graph::Stat::Median, rt_graph::Stat::Min, rt_graph::Stat::Max});

  for (auto stream_ : stream_vector_)
    whip::stream_destroy(stream_);

  stream_vector_.clear();
  whip::free(scratch_);
  data_.clear();
#ifdef REDUCE_USE_CUDA
  curandDestroyGenerator(gen);
#else
  hiprandDestroyGenerator(gen);
#endif

  return 0;
}
