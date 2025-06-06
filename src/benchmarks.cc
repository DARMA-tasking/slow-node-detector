#include <mkl.h>
#include <Kokkos_Random.hpp>

enum benchmarks {
    level1,
    level2,
    level3,
    dpotrf,
    num_benchmarks
};

template <typename T>
std::string typeToString();

template <>
std::string typeToString<double>() {
    return "double";
}

template <>
std::string typeToString<std::complex<double>>() {
    return "complex";
}

template <typename T>
std::tuple<std::vector<double>, double> runBenchmarkLevel1(int N, int iters) {
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
std::tuple<std::vector<double>, double> runBenchmarkLevel2(int M, int N, int iters) {
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
std::tuple<std::vector<double>, double> runBenchmarkLevel3() {
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
            N,         // leading dimension of A (because A is M×N)
            B_ptr,     // matrix B pointer
            K,         // leading dimension of B (because B is N×K)
            0.0,       // beta
            C_ptr,     // matrix C pointer
            K);        // leading dimension of C (because C is M×K)
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
std::tuple<std::vector<double>, double> runBenchmark(benchmarks type, int M, int N, int K, int iters) {
    switch (type) {
        case level1:
            return runBenchmarkLevel1<T>(M, N, K, iters);
        case level2:
            return runBenchmarkLevel2<T>(M, N, K, iters);
        case level3:
            return runBenchmarkLevel3<T>(M, N, K, iters);
        case dpotrf:
            return runBenchmarkDPOTRF<T>(M, N, K, iters);
        default:
            throw std::invalid_argument("Unsupported benchmark type");
    }
}
