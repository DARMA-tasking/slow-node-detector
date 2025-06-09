
#include "sensors.h"
#include "benchmarks.h"

#include <Kokkos_Random.hpp>

#include <iostream>

static int iters = 100;
static int M = 128;
static int N = 128;
static int K = 128;

int main(int argc, char** argv) {
  if (argc > 1) {
    iters = atoi(argv[1]) + 1; // add one iteration since we will drop the first one
    M = N = K = atoi(argv[2]);
  }
  std::cout << "iters: " << iters << ", M=N=K=" << M << std::endl;

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Loop through all available benchmarks
  sensors::runSensorsAndReduceOutput(processor_name, "pre");
  auto all_benchmark_output = runAllBenchmarks(M, N, K, iters);
  sensors::runSensorsAndReduceOutput(processor_name, "post");
  printBenchmarkOutput(all_benchmark_output);
  for (int i=0; i < benchmarks::num_benchmarks; i++) {
    auto benchmark_type = static_cast<benchmarks>(i);
    std::string benchmark_str = benchmarkToString(benchmark_type) + "_double";
    auto const& [iter_timings, total_time] results = runBenchmark<double>(benchmark_type, M, N, K, iters);
    reduceAndPrintBenchmarkOutput(iter_timings, total_time, benchmark_str);
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
