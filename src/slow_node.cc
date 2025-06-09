
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
  auto output = benchmarks::runAllBenchmarks(M, N, K, iters);
  sensors::runSensorsAndReduceOutput(processor_name, "post");
  benchmarks::printBenchmarkOutput(output);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
