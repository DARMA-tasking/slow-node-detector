#include <mkl.h>
#include <mpi.h>
#include <Kokkos_Random.hpp>

#include <unordered_map>

using benchmark_result_t = std::tuple<std::vector<double>, double>;
using all_results_t = std::unordered_map<std::string, benchmark_result_t>

template <>
std::string typeToString<double>() {
    return "double";
}

template <>
std::string typeToString<std::complex<double>>() {
    return "complex";
}

std::string benchmarkToString(const benchmarks& b) {
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
    std::cout << "-- level 1 benchmark [" << typeToString<T>() << "] -- " << std::endl;

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
        cblas_ddot(N, x_ptr, 1, y_ptr, 1);
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
    std::cout << "rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkLevel2(int M, int N, int iters) {
    std::cout << "-- level 2 benchmark [" << typeToString<T>() << "] -- " << std::endl;

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
        cblas_dger(CblasRowMajor, M, N,
                   1.0,   // alpha
                   x_ptr, // vector x
                   1,     // increment for x
                   y_ptr, // vector y
                   1,     // increment for y
                   A_ptr, // matrix A
                   N);    // leading dimension of A
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
    std::cout << "rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkLevel3() {
    std::cout << "-- level 3 benchmark [" << typeToString<T>() << "] -- " << std::endl;
    Kokkos::View<T**> A("A", M, N);
    Kokkos::View<T**> B("B", N, K);
    Kokkos::View<T**> C("C", M, K);

    T* A_ptr = A.data();
    T* B_ptr = B.data();
    T* C_ptr = C.data();

    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(A, pool, 10.0);
    Kokkos::fill_random(B, pool, 10.0);

    std::vector<double> iter_timings;

    double total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        Kokkos::Timer timer;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M,         // number of rows in C (and A)
            K,         // number of columns in C (and B)
            N,         // shared inner dimension (columns of A, rows of B)
            1.0,       // alpha
            A_ptr,     // matrix A pointer
            N,         // leading dimension of A (A is M×N)
            B_ptr,     // matrix B pointer
            K,         // leading dimension of B (B is N×K)
            0.0,       // beta
            C_ptr,     // matrix C pointer
            K);        // leading dimension of C (C is M×K)
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

    std::cout << "rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkLevel3() {
    std::cout << "-- level 3 benchmark [" << typeToString<T>() << "] -- " << std::endl;
    Kokkos::View<T**> A("A", M, N);
    Kokkos::View<T**> B("B", N, K);
    Kokkos::View<T**> C("C", M, K);

    T* A_ptr = A.data();
    T* B_ptr = B.data();
    T* C_ptr = C.data();

    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(A, pool, 10.0);
    Kokkos::fill_random(B, pool, 10.0);

    std::vector<double> iter_timings;

    double total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iters; i++) {
        Kokkos::Timer timer;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M,         // number of rows in C (and A)
            K,         // number of columns in C (and B)
            N,         // shared inner dimension (columns of A, rows of B)
            1.0,       // alpha
            A_ptr,     // matrix A pointer
            N,         // leading dimension of A (A is M×N)
            B_ptr,     // matrix B pointer
            K,         // leading dimension of B (B is N×K)
            0.0,       // beta
            C_ptr,     // matrix C pointer
            K);        // leading dimension of C (C is M×K)
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

    std::cout << "rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmarkDPOTRF(int N, int iters) {
    std::cout << "-- dpotrf benchmark [" << typeToString<T>() << "] -- " << std::endl;

    // Define matrix size
    Kokkos::View<T**> A("A", N, N);

    // Fill matrix A with random values
    Kokkos::Random_XorShift64_Pool pool(123);
    Kokkos::fill_random(A, pool, 10.0);

    // Make A symmetric positive definite
    Kokkos::parallel_for("MakeSPD", N, KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < N; j++) {
            A(i, j) = A(i, j) + A(j, i); // Symmetrize
        }
    });
    Kokkos::fence();

    // Prepare for MKL DPOTRF
    T* A_ptr = A.data();
    std::vector<double> iter_timings;
    double total_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform the benchmark iterations
    for (int i = 0; i < iters; i++) {
        Kokkos::Timer timer;

        // Call MKL DPOTRF
        int info;
        dpotrf_("L", &N, A_ptr, &N, &info); // 'L' for lower triangular

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

    std::cout << "rank: " << rank << ", total_time=" << total_time << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

template <typename T>
benchmark_results_t runBenchmark(benchmarks type, int M, int N, int K, int iters) {
    switch (type) {
        case level1:
            return runBenchmarkLevel1<T>(N, iters);
        case level2:
            return runBenchmarkLevel2<T>(M, N, iters);
        case level3:
            return runBenchmarkLevel3<T>(M, N, K, iters);
        case dpotrf:
            return runBenchmarkDPOTRF<T>(N, iters);
        default:
            throw std::invalid_argument("Unsupported benchmark type");
    }
}

all_results_t runAllBenchmarks(int M, int N, int K, int iters) {
    all_results_t all_results;
    for (int i=0; i < benchmarks::num_benchmarks; i++) {
        auto b = static_cast<benchmarks>(i);
        std::string benchmark_str = benchmarkToString(b) + "_double";
        all_results[benchmark_str] = runBenchmark<T>(b, M, N, K, iters);
    }
    return all_results;
}

void reduceAndPrintBenchmarkOutput(all_results_t benchmark_results)
{
    int rank = -1;
    int num_ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    for (const auto& [benchmark_str, benchmark_results]: benchmark_results) {
        auto const& [iter_timings, total_time] = benchmark_results;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

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
            std::cout << "=== " << benchmark << " ===" << std::endl;
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
