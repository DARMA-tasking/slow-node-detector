#include <mpi.h>
#include <Kokkos_Random.hpp>
#include <Kokkos_Complex.hpp>

#include "ops.h"
#include "benchmarks.h"

#include <iostream>

namespace benchmarks {

template <>
std::string typeToString<double>() {
    return "double ";
}

template <>
std::string typeToString<Kokkos::complex<double>>() {
    return "complex";
}

std::string benchmarkToString(const benchmark_types& b) {
    switch (b) {
        case level1: return "level1";
        case level2: return "level2";
        case level3: return "level3";
        case dpotrf: return "dpotrf";
        default: return "unknown";
    }
}

template <typename T>
benchmark_results_t runBenchmarkLevel1(int N, int iters) {
    Kokkos::View<T*> x("x", N);
    Kokkos::View<T*> y("y", N);

    T* x_ptr = x.data();
    T* y_ptr = y.data();

    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(x, pool, 10.0);
    Kokkos::fill_random(y, pool, 10.0);

    std::vector<double> iter_timings;
    double total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        Kokkos::Timer timer;
        ops::level1<T>(N, x_ptr, y_ptr);
        Kokkos::fence();

        // Skip the first iteration
        if (i > 0) {
            double time = timer.seconds();
            total_time += time;
            iter_timings.push_back(time);
        }
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[level1 " << typeToString<T>() << "] (N=" << N << ") rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkLevel2(int M, int N, int iters) {
    Kokkos::View<T*> x("x", M);
    Kokkos::View<T*> y("y", N);
    Kokkos::View<T**> A("A", M, N);

    T* x_ptr = x.data();
    T* y_ptr = y.data();
    T* A_ptr = A.data();

    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(x, pool, 10.0);
    Kokkos::fill_random(y, pool, 10.0);
    Kokkos::fill_random(A, pool, 10.0);

    std::vector<double> iter_timings;
    double total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        Kokkos::Timer timer;
        ops::level2<T>(M, N, T(1.0), x_ptr, y_ptr, A_ptr, N);
        Kokkos::fence();

        // Skip the first iteration
        if (i > 0) {
            double time = timer.seconds();
            total_time += time;
            iter_timings.push_back(time);
        }
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[level2 " << typeToString<T>() << "] (M=" << M << ", N=" << N << ") rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkLevel3(int M, int N, int K, int iters) {
    Kokkos::View<T**> A("A", M, N);
    Kokkos::View<T**> B("B", N, K);
    Kokkos::View<T**> C("C", M, K);

    T* A_ptr = A.data();
    T* B_ptr = B.data();
    T* C_ptr = C.data();

    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(A, pool, 10.0);
    Kokkos::fill_random(B, pool, 10.0);
    Kokkos::fill_random(C, pool, 10.0);

    std::vector<double> iter_timings;

    double total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        Kokkos::Timer timer;
        ops::level3<T>(M, K, N, T(1.0), A_ptr, N, B_ptr, K, T(0.0), C_ptr, K);
        Kokkos::fence();

        // Skip the first iteration
        if (i > 0) {
            double time = timer.seconds();
            total_time += time;
            iter_timings.push_back(time);
        }
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "[level3 " << typeToString<T>() << "] (M=" << M << ", N=" << N << ", K=" << K << ") rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkDPOTRF(int N, int iters) {
    Kokkos::View<T**> A("A", N, N);

    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(A, pool, 10.0);

    // Make A symmetric positive definite
    Kokkos::parallel_for("MakeSPD", N, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < N; j++) {
            A(i, j) = A(i, j) + A(j, i);
        }
    });

    Kokkos::fence();

    T* A_ptr = A.data();

    std::vector<double> iter_timings;
    double total_time = 0.0;

    long long N_ll = static_cast<long long>(N);
    char uplo = 'L';

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        long long info;
        Kokkos::Timer timer;
        ops::dpotrf<T>(uplo, N_ll, A_ptr, N_ll, &info);

        Kokkos::fence();

        // Skip the first iteration
        if (i > 0) {
            double time = timer.seconds();
            total_time += time;
            iter_timings.push_back(time);
        }
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "[dpotrf " << typeToString<T>() << "] (N=" << N << ") rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmark(benchmark_types b, std::vector<int> sizes, int iters) {
    switch (b) {
        case level1:
            return runBenchmarkLevel1<T>(sizes[0], iters);
        case level2:
            return runBenchmarkLevel2<T>(sizes[1], sizes[2], iters);
        case level3:
            return runBenchmarkLevel3<T>(sizes[3], sizes[4], sizes[5], iters);
        case dpotrf:
            return runBenchmarkDPOTRF<T>(sizes[6], iters);
        default:
            throw std::invalid_argument("Unsupported benchmark type");
    }
}

all_results_t runAllBenchmarks(std::vector<int> sizes, int iters) {
    all_results_t all_results;
    for (int i=0; i < benchmark_types::num_benchmarks; i++) {
        auto b = static_cast<benchmark_types>(i);
        std::string benchmark_str = benchmarkToString(b);
        all_results[benchmark_str + "_double"] = runBenchmark<double>(b, sizes, iters);
        MPI_Barrier(MPI_COMM_WORLD);
        all_results[benchmark_str + "_complex"] = runBenchmark<Kokkos::complex<double>>(b, sizes, iters);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return all_results;
}

void printBenchmarkOutput(all_results_t benchmark_results, int iters)
{
    int rank = -1;
    int num_ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rank == 0) {
        std::cout << "num_ranks: " << num_ranks << std::endl;
    }

    for (const auto& [benchmark_str, benchmark_results]: benchmark_results) {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        auto const& [iter_timings, total_time] = benchmark_results;

        std::vector<double> all_times;
        all_times.resize(num_ranks);

        std::vector<double> all_iter_times;
        all_iter_times.resize(num_ranks * iters);

        std::vector<char> all_processor_names;
        all_processor_names.resize(num_ranks * MPI_MAX_PROCESSOR_NAME);

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
            std::cout << std::endl << benchmark_str << std::endl;
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
    }
}
} // end namespace benchmarks
