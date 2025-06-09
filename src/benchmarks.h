#include <std::string>

namespace benchmarks {

using benchmark_result_t = std::tuple<std::vector<double>, double>;
using all_results_t = std::unordered_map<std::string, benchmark_result_t>

enum benchmarks_types {
    level1,
    level2,
    level3,
    dpotrf,
    num_benchmarks
};

template <typename T>
std::string typeToString();

std::string benchmarkToString(const benchmarks& b);

template <typename T>
benchmark_result_t runBenchmarkLevel1(int N, int iters);

template <typename T>
benchmark_result_t runBenchmarkLevel2(int M, int N, int iters);

template <typename T>
benchmark_result_t runBenchmarkLevel3(int M, int N, int K, int iters);

template <typename T>
benchmark_result_t runBenchmarkDPOTRF(int N, int iters);

template <typename T>
benchmark_result_t runBenchmark(benchmarks type, int M, int N, int K, int iters);

all_results_t runAllBenchmarks(int M, int N, int K, int iters);

void printBenchmarkOutput(all_results_t benchmark_results);

} // end namespace benchmarks
