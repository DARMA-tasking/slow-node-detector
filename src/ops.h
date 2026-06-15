#ifndef SRC_OPS_H
#define SRC_OPS_H

namespace ops {

template <typename T>
constexpr bool isDouble();

template <typename T>
constexpr bool isComplex();

template <typename T>
void level1(int N, const T* x, const T* y);

template <typename T>
void level2(int M, int N, T alpha, const T* x, const T* y, T* A, int lda);

template <typename T>
void level3(
    int M, int N, int K,
    T alpha, const T* A, int lda,
    const T* B, int ldb,
    T beta,  T* C, int ldc);

template <typename T>
void dpotrf(char uplo, long long n, T* a, long long lda, long long* info);

} // end namespace ops

#include "ops_impl.h"

#endif // SRC_OPS_H
