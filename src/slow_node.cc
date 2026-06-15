#include "benchmarks.h"
#include "sensors.h"

#include <mpi.h>
#include <Kokkos_Core.hpp>

#ifdef VT_ENABLED
#include <vt/transport.h>
#endif

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int iters = 100;
int N1 = 128;
int M2 = 128;
int N2 = 128;
int M3 = 128;
int N3 = 128;
int K3 = 128;
int N4 = 128;

void printUsage(char const* exe) {
  std::cerr << "USAGE: " << exe
            << " [--benchmark all|level1|level2|level3|dpotrf|perf]"
            << " [iters [N1 M2 N2 M3 N3 K3 N4]]" << std::endl;
}

bool parseBenchmark(std::string const& name, benchmarks::benchmark_types& benchmark) {
  if (name == "all") {
    benchmark = benchmarks::num_benchmarks;
  } else if (name == "level1") {
    benchmark = benchmarks::level1;
  } else if (name == "level2") {
    benchmark = benchmarks::level2;
  } else if (name == "level3") {
    benchmark = benchmarks::level3;
  } else if (name == "dpotrf") {
    benchmark = benchmarks::dpotrf;
  } else if (name == "perf") {
    benchmark = benchmarks::perf;
  } else {
    return false;
  }
  return true;
}

} // namespace

/*
 * USAGE: ./slow_node [--benchmark all|level1|level2|level3|dpotrf|perf]
 *                    [iters [N1 M2 N2 M3 N3 K3 N4]]
 */
int main(int argc, char** argv) {
  std::vector<int> sizes = {N1, M2, N2, M3, N3, K3, N4};
  benchmarks::benchmark_types benchmark = benchmarks::num_benchmarks;

  int arg = 1;
  if (argc > arg && std::string(argv[arg]) == "--benchmark") {
    if (argc <= arg + 1 || !parseBenchmark(argv[arg + 1], benchmark)) {
      printUsage(argv[0]);
      return 1;
    }
    arg += 2;
  }

  int const remaining = argc - arg;
  if (remaining != 0) {
    if (remaining != 1 && remaining != 8) {
      printUsage(argv[0]);
      return 1;
    }

    iters = std::atoi(argv[arg++]);
    if (remaining == 8) {
      sizes[0] = std::atoi(argv[arg++]);
      sizes[1] = std::atoi(argv[arg++]);
      sizes[2] = std::atoi(argv[arg++]);
      sizes[3] = std::atoi(argv[arg++]);
      sizes[4] = std::atoi(argv[arg++]);
      sizes[5] = std::atoi(argv[arg++]);
      sizes[6] = std::atoi(argv[arg++]);
    }
  }

  std::cout << "iters: " << iters
            << ", sizes=" << sizes[0] << "," << sizes[1] << "," << sizes[2]
            << "," << sizes[3] << "," << sizes[4] << "," << sizes[5]
            << "," << sizes[6] << std::endl;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
#ifdef VT_ENABLED
  vt::initialize(argc, argv, &comm);
#endif
  Kokkos::initialize(argc, argv);

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  sensors::runSensorsAndReduceOutput(processor_name, "pre");
  auto output = benchmark == benchmarks::num_benchmarks ?
    benchmarks::runAllBenchmarks(sizes, iters + 1) :
    benchmarks::runSelectedBenchmark(benchmark, sizes, iters + 1);
  sensors::runSensorsAndReduceOutput(processor_name, "post");

  benchmarks::printBenchmarkOutput(output, iters);

  Kokkos::finalize();
#ifdef VT_ENABLED
  vt::finalize();
#endif
  MPI_Finalize();
  return 0;
}
