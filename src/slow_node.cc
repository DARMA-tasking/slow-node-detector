
#include "sensors.h"
#include "benchmarks.h"

#include <mkl.h>
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

  int rank = -1;
  int num_ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  sensors::runSensorsAndReduceOutput(processor_name, "pre");

  // Simple for loop just for now
  for (int i=0; i < benchmarks::num_benchmarks; i++) {
    auto benchmark_type = static_cast<benchmarks>(i);
    try {
      auto const& [iter_timings, total_time] results = runBenchmark<T>(benchmark_type, M, N, K, iters);
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error running benchmark type " << i << ": " << e.what() << std::endl;
    }
  }
  // auto const& [iter_timings, total_time] = runBenchmark<double>(level3, M, N, K, iters);
  sensors::runSensorsAndReduceOutput(processor_name, "post");

  std::vector<double> all_times;
  all_times.resize(num_ranks);

  std::vector<double> all_iter_times;
  all_iter_times.resize(num_ranks * iters);

  std::vector<char> all_processor_names;
  all_processor_names.resize(num_ranks * MPI_MAX_PROCESSOR_NAME);

  if (rank == 0) {
    std::cout << "num_ranks: " << num_ranks << std::endl;
  }

  MPI_Gather(
    &total_time, 1, MPI_DOUBLE,
    &all_times[0], 1, MPI_DOUBLE, 0,
    MPI_COMM_WORLD
  );

  MPI_Gather(
    &iter_timings[0], iters, MPI_DOUBLE,
    &all_iter_times[0], iters, MPI_DOUBLE, 0,
    MPI_COMM_WORLD
  );

  MPI_Gather(
    &processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
    &all_processor_names[0], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0,
    MPI_COMM_WORLD
  );

  if (rank == 0) {
    int cur_rank = 0;
    int cur = 0;
    for (auto&& time : all_times) {
      std::cout << "gather: " << cur_rank << " ("
        << std::string(&all_processor_names[cur_rank * MPI_MAX_PROCESSOR_NAME])
        << "): " << time << ": breakdown: ";
      for (int i = cur; i < iters + cur; i++) {
        std::cout << all_iter_times[i] << " ";
      }
      std::cout << std::endl;
      cur += iters;
      cur_rank++;
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
