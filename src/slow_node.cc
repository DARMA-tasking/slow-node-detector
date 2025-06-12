
#include "sensors.h"
#include "benchmarks.h"

#include <Kokkos_Random.hpp>

#include <iostream>

static int iters = 100;
static int N1 = 128;
static int M2 = 128;
static int N2 = 128;
static int M3 = 128;
static int N3 = 128;
static int K3 = 128;
static int N4 = 128;

/*
 * USAGE: ./slow_node <iters> <N1> <M2> <N2> <M3> <N3> <K3> <N4>
 */
int main(int argc, char** argv) {

  std::vector<int> sizes;

  if (argc > 1) {
    iters = atoi(argv[1]);
    sizes.push_back(atoi(argv[2])); /* N1 */
    sizes.push_back(atoi(argv[3])); /* M2 */
    sizes.push_back(atoi(argv[4])); /* N2 */
    sizes.push_back(atoi(argv[5])); /* M3 */
    sizes.push_back(atoi(argv[6])); /* N3 */
    sizes.push_back(atoi(argv[7])); /* K3 */
    sizes.push_back(atoi(argv[8])); /* N4 */
  }

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Loop through all available benchmarks
  sensors::runSensorsAndReduceOutput(processor_name, "pre");
  auto output = benchmarks::runAllBenchmarks(sizes, iters + 1); // add one iteration since we'll drop the first one
  sensors::runSensorsAndReduceOutput(processor_name, "post");
  benchmarks::printBenchmarkOutput(output, iters);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
