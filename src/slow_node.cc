
#include <iostream>
#include <vector>

#include <mpi.h>

#include <Kokkos_Random.hpp>
#include <KokkosBlas3_gemm.hpp>

static int iters = 100;
static int M = 128;
static int N = 128;
static int K = 128;

std::tuple<std::vector<double>, double> runBenchmark() {
  Kokkos::View<double**> A(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), M, N);
  Kokkos::View<double**> B(Kokkos::view_alloc(Kokkos::WithoutInitializing, "B"), N, K);
  Kokkos::View<double**> C(Kokkos::view_alloc(Kokkos::WithoutInitializing, "C"), M, K);

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

int main(int argc, char** argv) {
  if (argc > 1) {
    iters = atoi(argv[1]);
    M = N = K = atoi(argv[2]);
  }
  std::cout << "iters: " << iters << ", M=N=K=" << M << std::endl;

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  auto const& [iter_timings, total_time] = runBenchmark();

  int rank = -1;
  int num_ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  std::vector<double> all_times;
  all_times.resize(num_ranks);

  std::vector<double> all_iter_times;
  all_iter_times.resize(num_ranks * iters);

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

  if (rank == 0) {
    int cur_rank = 0;
    int cur = 0;
    for (auto&& time : all_times) {
      std::cout << "gather: " << cur_rank << ": " << time << ": breakdown: ";
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

