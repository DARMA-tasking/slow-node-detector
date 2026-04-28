#ifndef SRC_OPS_IMPL_H
#define SRC_OPS_IMPL_H

#include <mkl.h>
#include <cassert>
#include <iostream>

#include <Kokkos_Complex.hpp>

#include "ops.h"
#include "benchmarks.h"

namespace ops {

///////////////////////////////////////////////////////////////////////////////
// Helpers

template <typename T>
constexpr bool isComplex() {
    if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
        return true;
    }
    return false;
}

template <typename T>
constexpr bool isDouble() {
    if constexpr (std::is_same_v<T, double>) {
        return true;
    }
    assert(isComplex<T>() && "Type must be either double or Kokkos::complex<double>.");
    return false;
}

static void warnWrongType(const std::string& benchmark) {
    std::cout << "Warning: Type must be either double or Kokkos::complex. Skipping " << benchmark << " benchmark." << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
// LEVEL 1

template <typename T>
void level1(int N, const T* x, const T* y) {
    if constexpr (isDouble<T>()) {
        cblas_ddot(N, x, 1, y, 1);
    } else if constexpr (isComplex<T>()) {
        T result;
        cblas_zdotu_sub(N, x, 1, y, 1, &result);
    } else {
        warnWrongType("level1");
    }
}

///////////////////////////////////////////////////////////////////////////////
// LEVEL 2

template <typename T>
void level2(
    int M,
    int N,
    T alpha,
    const T* x,
    const T* y,
    T* A,
    int lda)
{
    if constexpr (isDouble<T>()) {
        cblas_dger(CblasRowMajor, M, N, alpha, x, 1, y, 1, A, lda);
    } else if constexpr (isComplex<T>()) {
        cblas_zgeru(
            CblasRowMajor,
            M, N,
            reinterpret_cast<const void*>(&alpha),
            reinterpret_cast<const void*>(x), 1,
            reinterpret_cast<const void*>(y), 1,
            reinterpret_cast<void*>(A), lda
        );
    } else {
        warnWrongType("level2");
    }
}

///////////////////////////////////////////////////////////////////////////////
// LEVEL 3

template <typename T>
void level3(
    int M,
    int N,
    int K,
    T alpha,
    const T* A,
    int lda,
    const T* B,
    int ldb,
    T beta,
    T* C,
    int ldc)
{
    if constexpr (isDouble<T>()) {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc
        );
    } else if constexpr (isComplex<T>()) {
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            reinterpret_cast<const void*>(&alpha),
            reinterpret_cast<const void*>(A), lda,
            reinterpret_cast<const void*>(B), ldb,
            reinterpret_cast<const void*>(&beta),
            reinterpret_cast<void*>(C), ldc);
    } else {
        warnWrongType("level3");
    }


}

///////////////////////////////////////////////////////////////////////////////
// DPOTRF

template <typename T>
void dpotrf(char uplo, long long n, T* a, long long lda, long long* info) {
        long long local_info = 0;

    if constexpr (isDouble<T>()) {
                local_info = static_cast<long long>(
                    LAPACKE_dpotrf(
                        LAPACK_ROW_MAJOR,
                        uplo,
                        static_cast<lapack_int>(n),
                        a,
                        static_cast<lapack_int>(lda)
                    )
                );
    } else if constexpr (isComplex<T>()) {
                local_info = static_cast<long long>(
                    LAPACKE_zpotrf(
                        LAPACK_ROW_MAJOR,
                        uplo,
                        static_cast<lapack_int>(n),
                        reinterpret_cast<MKL_Complex16*>(a),
                        static_cast<lapack_int>(lda)
                    )
                );
    } else {
                local_info = -1;
        warnWrongType("dpotrf");
    }

        if (info != nullptr) {
                *info = local_info;
        }
}

} // end namespace ops

#endif // SRC_OPS_IMPL_H
