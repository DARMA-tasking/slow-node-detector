#ifndef SRC_BENCHMARKS_H
#define SRC_BENCHMARKS_H

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace benchmarks {

using benchmark_results_t = std::tuple<std::vector<double>, double>;
using all_results_t = std::unordered_map<std::string, benchmark_results_t>;

enum benchmark_types {
    level1,
    level2,
    level3,
    dpotrf,
    perf,
    num_benchmarks
};

template <typename T>
std::string typeToString();

std::string benchmarkToString(const benchmark_types& b);

template <typename T>
benchmark_results_t runBenchmarkLevel1(int N, int iters);

template <typename T>
benchmark_results_t runBenchmarkLevel2(int M, int N, int iters);

template <typename T>
benchmark_results_t runBenchmarkLevel3(int M, int N, int K, int iters);

template <typename T>
benchmark_results_t runBenchmarkDPOTRF(int N, int iters);

template <typename T>
benchmark_results_t runBenchmarkPerf(int iters);

template <typename T>
benchmark_results_t runBenchmark(benchmark_types b, std::vector<int> sizes, int iters);

all_results_t runSelectedBenchmark(benchmark_types b, std::vector<int> sizes, int iters);

all_results_t runAllBenchmarks(std::vector<int> sizes, int iters);

void printBenchmarkOutput(all_results_t benchmark_results, int iters);

} // end namespace benchmarks

#endif // SRC_BENCHMARKS_H
