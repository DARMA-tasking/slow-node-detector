#ifndef SRC_BENCHMARKS_H
#define SRC_BENCHMARKS_H

#include <string>
#include <unordered_map>

namespace benchmarks {

using benchmark_results_t = std::tuple<std::vector<double>, double>;
using all_results_t = std::unordered_map<std::string, benchmark_results_t>;

enum benchmark_types {
    level1,
    level2,
    level3,
    dpotrf,
    num_benchmarks
};

template <typename T>
std::string typeToString();

std::string benchmarkToString(const benchmark_types& b);

template <typename T>
benchmark_results_t runBenchmarkLevel1(std::size_t flops, int iters);

template <typename T>
benchmark_results_t runBenchmarkLevel2(std::size_t flops, int iters);

template <typename T>
benchmark_results_t runBenchmarkLevel3(std::size_t flops, int iters);

template <typename T>
benchmark_results_t runBenchmarkDPOTRF(std::size_t flops, int iters);

template <typename T>
benchmark_results_t runBenchmark(benchmark_types b, std::size_t flops, int iters);

all_results_t runAllBenchmarks(std::size_t flops, int iters);

void printBenchmarkOutput(all_results_t benchmark_results, int iters);

} // end namespace benchmarks

#endif // SRC_BENCHMARKS_H
