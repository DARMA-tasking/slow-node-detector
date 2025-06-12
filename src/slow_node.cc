
#include "sensors.h"
#include "benchmarks.h"

#include <Kokkos_Random.hpp>

#include <iostream>

static int iters = 100;
static int M = 128;
static int N1 = 128;
static int N2 = 128;
static int N3 = 128;
static int K = 128;

/*
 * USAGE: ./slow_node <iters> <M/K> | <N1> <N2> <N3>
 *
 *    M and K share the same value, given by the <M/K> slot.
 *
 *    If <N1> <N2> and <N3> are not provided, the value given for
 *    M and K is also used for all N.
 *
 *    Similarly, if <N2> is not provided, the value given for <N1>
 *    is used for all N, and so on with <N3>.
 */

int main(int argc, char** argv) {

  if (argc > 1) {
    iters = atoi(argv[1]);
    M = N1 = N2 = N3 = K = atoi(argv[2]);
    if (argc > 3) {
      N1 = N2 = N3 = atoi(argv[3]);
      if (argc > 4) {
        N2 = N3 = atoi(argv[4]);
        if (argc > 5) {
          N3 = atoi(argv[5]);
        }
      }
    }
  }

  std::cout << "iters: " << iters << ", M=K=" << M << std::endl;
  std::cout << "N1=" << N1 << ", N2=" << N2 << ", N3=" << N3 << std::endl;

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Loop through all available benchmarks
  sensors::runSensorsAndReduceOutput(processor_name, "pre");
  auto output = benchmarks::runAllBenchmarks(M, N1, N2, N3, K, iters + 1); // add one iteration since we'll drop the first one
  sensors::runSensorsAndReduceOutput(processor_name, "post");
  benchmarks::printBenchmarkOutput(output, iters);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
