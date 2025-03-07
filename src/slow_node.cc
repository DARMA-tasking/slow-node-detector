
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <mpi.h>

#include <Kokkos_Random.hpp>
#include <KokkosBlas3_gemm.hpp>

static int iters = 100;
static int M = 128;
static int N = 128;
static int K = 128;

std::tuple<std::vector<double>, double> runBenchmark() {
  Kokkos::View<double**> A("A", M, N);
  Kokkos::View<double**> B("B", N, K);
  Kokkos::View<double**> C("C", M, K);

  Kokkos::Random_XorShift64_Pool pool(123);
  Kokkos::fill_random(A, pool, 10.0);
  Kokkos::fill_random(B, pool, 10.0);

  std::vector<double> iter_timings;

  double total_time = 0.0;

  for (int i = 0; i < iters; i++) {
    Kokkos::Timer timer;
    KokkosBlas::gemm("N", "N", 1.0, A, B, 0.0, C);
    Kokkos::fence();
    double time = timer.seconds();
    total_time += time;
    iter_timings.push_back(time);
  }

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "rank: " << rank << ", total_time=" << total_time << std::endl;

  return std::make_tuple(iter_timings, total_time);
}

void runSensors(int rank, const std::string& proc_name) {
  // Set up log files
  std::filesystem::path output_dir = "sensors_output";
  std::filesystem::path filename = output_dir / ("sensors_" + proc_name + "_" + std::to_string(rank) + ".log");
  std::filesystem::create_directories(output_dir);
  std::ofstream log_file(filename);
  if (!log_file) {
      std::cerr << "Error: Unable to open " << filename << " for writing.\n";
      return;
  }

  // Get output from `sensors`
  FILE* pipe = popen("sensors", "r");
  if (!pipe) {
      std::cerr << "Error: Unable to run sensors command\n";
      return;
  }
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      log_file << buffer;
  }
  pclose(pipe);
  log_file.close();
}

int main(int argc, char** argv) {
  if (argc > 1) {
    iters = atoi(argv[1]);
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

  if (rank == 0) system("mkdir -p sensors_output");
  MPI_Barrier(MPI_COMM_WORLD);

  auto const& [iter_timings, total_time] = runBenchmark();
  runSensors(rank, processor_name);

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
        std::cout << all_iter_times[cur] << " ";
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
