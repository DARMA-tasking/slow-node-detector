
#include "sensors.h"
#include "benchmarks.h"

#include <Kokkos_Random.hpp>

#include <iostream>

static int iters = 100;
static std::size_t flops = 1000000;

/*
 * USAGE: ./slow_node <iters> <flops>
 */
int main(int argc, char** argv) {

  if (argc > 1) {
    iters = atoi(argv[1]);
    flops = atoi(argv[2]);
  }

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Loop through all available benchmarks
  sensors::runSensorsAndReduceOutput(processor_name, "pre");
  auto output = benchmarks::runAllBenchmarks(flops, iters + 1); // add one iteration since we drop the first one
  sensors::runSensorsAndReduceOutput(processor_name, "post");
  benchmarks::printBenchmarkOutput(output, iters);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
