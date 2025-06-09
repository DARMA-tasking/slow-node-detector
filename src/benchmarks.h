#include <std::string>

enum benchmarks {
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
std::tuple<std::vector<double>, double> runBenchmarkLevel1(int N, int iters);

template <typename T>
std::tuple<std::vector<double>, double> runBenchmarkLevel2(int M, int N, int iters);

template <typename T>
std::tuple<std::vector<double>, double> runBenchmarkLevel3(int M, int N, int K, int iters);

template <typename T>
std::tuple<std::vector<double>, double> runBenchmarkDPOTRF(int N, int iters);

template <typename T>
std::tuple<std::vector<double>, double> runBenchmark(
    benchmarks type, int M, int N, int K, int iters);

void reduceAndPrintBenchmarkOutput(
    std::vector<double> iter_timings,
    double total_time,
    std::string benchmark);