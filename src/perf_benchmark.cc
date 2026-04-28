#include <mpi.h>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "benchmarks.h"
#include "perf.h"

#include <array>
#include <iostream>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define SLOW_NODE_X86_SIMD 1
#else
#define SLOW_NODE_X86_SIMD 0
#endif

namespace benchmarks {

namespace {

constexpr int kPerfWarmupIters = 1;
constexpr int kPerfLoopCount = 512;
constexpr int kPerfVectorLength = 256;

template <typename T>
inline void doNotOptimize(T const& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#else
    (void)value;
#endif
}

inline int perfLoopCount() {
    return kPerfLoopCount;
}

#if SLOW_NODE_X86_SIMD && (defined(__GNUC__) || defined(__clang__))

struct SimdCapabilities {
    bool sse2 = false;
    bool avx = false;
    bool avx2_fma = false;
    bool avx512f = false;
};

SimdCapabilities detectSimdCapabilities() {
    __builtin_cpu_init();

    SimdCapabilities capabilities;
    capabilities.sse2 = __builtin_cpu_supports("sse2");
    capabilities.avx = __builtin_cpu_supports("avx");
    capabilities.avx2_fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    capabilities.avx512f = __builtin_cpu_supports("avx512f");
    return capabilities;
}

__attribute__((target("sse2"), noinline))
double runSseScalarKernel(
  double* lhs, double* rhs, double* out,
  float* lhs_f, float* rhs_f, float* out_f,
  int count, int loops
) {
    // Primarily drives:
    // - fp_arith_inst_retired_scalar_double
    // - fp_arith_inst_retired_scalar_single
    __m128d accum_d = _mm_set_sd(0.25);
    __m128 accum_f = _mm_set_ss(0.5f);
    __m128d damp_d = _mm_set_sd(0.99999988079071044921875);
    __m128 damp_f = _mm_set_ss(0.99999988f);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; ++i) {
            __m128d lhs_d = _mm_load_sd(lhs + i);
            __m128d rhs_d = _mm_load_sd(rhs + i);
            __m128 lhs_s = _mm_load_ss(lhs_f + i);
            __m128 rhs_s = _mm_load_ss(rhs_f + i);

            accum_d = _mm_add_sd(_mm_mul_sd(accum_d, damp_d), _mm_mul_sd(lhs_d, rhs_d));
            accum_f = _mm_add_ss(_mm_mul_ss(accum_f, damp_f), _mm_mul_ss(lhs_s, rhs_s));

            _mm_store_sd(out + i, accum_d);
            _mm_store_ss(out_f + i, accum_f);
        }
    }

    return _mm_cvtsd_f64(accum_d) + static_cast<double>(_mm_cvtss_f32(accum_f))
      + out[count - 1] + static_cast<double>(out_f[count - 1]);
}

__attribute__((target("sse2"), noinline))
double runSsePackedKernel(
  double* lhs, double* rhs, double* out,
  float* lhs_f, float* rhs_f, float* out_f,
  int count, int loops
) {
    // Primarily drives:
    // - fp_arith_inst_retired_128b_packed_double
    // - fp_arith_inst_retired_128b_packed_single
    __m128d accum0 = _mm_set1_pd(0.25);
    __m128d accum1 = _mm_set1_pd(0.75);
    __m128 accum_f = _mm_set1_ps(0.5f);
    __m128d scale_a = _mm_set1_pd(1.00000011920928955078125);
    __m128d scale_b = _mm_set1_pd(0.9999997615814208984375);
    __m128 scale_f = _mm_set1_ps(1.0000001f);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 4) {
            __m128d lhs0 = _mm_loadu_pd(lhs + i);
            __m128d rhs0 = _mm_loadu_pd(rhs + i);
            __m128d lhs1 = _mm_loadu_pd(lhs + i + 2);
            __m128d rhs1 = _mm_loadu_pd(rhs + i + 2);
            __m128 lhs_s = _mm_loadu_ps(lhs_f + i);
            __m128 rhs_s = _mm_loadu_ps(rhs_f + i);

            __m128d mix0 = _mm_add_pd(_mm_mul_pd(lhs0, scale_a), rhs0);
            __m128d mix1 = _mm_sub_pd(_mm_mul_pd(lhs1, scale_b), rhs1);
            __m128 mix_s = _mm_add_ps(_mm_mul_ps(lhs_s, scale_f), rhs_s);

            accum0 = _mm_add_pd(accum0, _mm_mul_pd(mix0, mix0));
            accum1 = _mm_add_pd(accum1, _mm_mul_pd(mix1, mix1));
            accum_f = _mm_add_ps(accum_f, _mm_mul_ps(mix_s, mix_s));

            _mm_storeu_pd(out + i, _mm_add_pd(mix0, accum0));
            _mm_storeu_pd(out + i + 2, _mm_sub_pd(mix1, accum1));
            _mm_storeu_ps(out_f + i, _mm_add_ps(mix_s, accum_f));
        }
    }

    alignas(16) std::array<double, 2> lane0{};
    alignas(16) std::array<double, 2> lane1{};
    alignas(16) std::array<float, 4> lanes_f{};
    _mm_store_pd(lane0.data(), accum0);
    _mm_store_pd(lane1.data(), accum1);
    _mm_store_ps(lanes_f.data(), accum_f);
    return lane0[0] + lane0[1] + lane1[0] + lane1[1]
      + lanes_f[0] + lanes_f[1] + lanes_f[2] + lanes_f[3]
      + out[count - 1] + static_cast<double>(out_f[count - 1]);
}

__attribute__((target("avx"), noinline))
double runAvxKernel(double* lhs, double* rhs, double* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_256b_packed_double
    __m256d accum0 = _mm256_set1_pd(0.5);
    __m256d accum1 = _mm256_set1_pd(1.5);
    __m256d bias = _mm256_setr_pd(1.0, -1.0, 0.5, -0.5);
    __m256d scale = _mm256_set1_pd(1.0000002384185791015625);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 8) {
            __m256d lhs0 = _mm256_loadu_pd(lhs + i);
            __m256d rhs0 = _mm256_loadu_pd(rhs + i);
            __m256d lhs1 = _mm256_loadu_pd(lhs + i + 4);
            __m256d rhs1 = _mm256_loadu_pd(rhs + i + 4);

            __m256d mix0 = _mm256_add_pd(_mm256_mul_pd(lhs0, scale), _mm256_add_pd(rhs0, bias));
            __m256d mix1 = _mm256_sub_pd(_mm256_mul_pd(rhs1, scale), _mm256_sub_pd(lhs1, bias));

            accum0 = _mm256_add_pd(accum0, _mm256_mul_pd(mix0, mix0));
            accum1 = _mm256_add_pd(accum1, _mm256_mul_pd(mix1, mix1));

            _mm256_storeu_pd(out + i, _mm256_add_pd(mix0, accum0));
            _mm256_storeu_pd(out + i + 4, _mm256_sub_pd(mix1, accum1));
        }
    }

    alignas(32) std::array<double, 4> lanes0{};
    alignas(32) std::array<double, 4> lanes1{};
    _mm256_store_pd(lanes0.data(), accum0);
    _mm256_store_pd(lanes1.data(), accum1);
    return lanes0[0] + lanes0[1] + lanes0[2] + lanes0[3]
      + lanes1[0] + lanes1[1] + lanes1[2] + lanes1[3] + out[count - 2];
}

__attribute__((target("avx"), noinline))
double runAvxSingleKernel(float* lhs, float* rhs, float* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_256b_packed_single
    __m256 accum0 = _mm256_set1_ps(0.5f);
    __m256 accum1 = _mm256_set1_ps(1.5f);
    __m256 bias = _mm256_setr_ps(1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 3.0f, -3.0f);
    __m256 scale = _mm256_set1_ps(1.0000002f);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 16) {
            __m256 lhs0 = _mm256_loadu_ps(lhs + i);
            __m256 rhs0 = _mm256_loadu_ps(rhs + i);
            __m256 lhs1 = _mm256_loadu_ps(lhs + i + 8);
            __m256 rhs1 = _mm256_loadu_ps(rhs + i + 8);

            __m256 mix0 = _mm256_add_ps(_mm256_mul_ps(lhs0, scale), _mm256_add_ps(rhs0, bias));
            __m256 mix1 = _mm256_sub_ps(_mm256_mul_ps(rhs1, scale), _mm256_sub_ps(lhs1, bias));

            accum0 = _mm256_add_ps(accum0, _mm256_mul_ps(mix0, mix0));
            accum1 = _mm256_add_ps(accum1, _mm256_mul_ps(mix1, mix1));

            _mm256_storeu_ps(out + i, _mm256_add_ps(mix0, accum0));
            _mm256_storeu_ps(out + i + 8, _mm256_sub_ps(mix1, accum1));
        }
    }

    alignas(32) std::array<float, 8> lanes0{};
    alignas(32) std::array<float, 8> lanes1{};
    _mm256_store_ps(lanes0.data(), accum0);
    _mm256_store_ps(lanes1.data(), accum1);
    double sum = static_cast<double>(out[count - 2]);
    for (float value : lanes0) {
        sum += value;
    }
    for (float value : lanes1) {
        sum += value;
    }
    return sum;
}

__attribute__((target("avx"), noinline))
double runAvxComplexKernel(double* lhs, double* rhs, double* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_256b_packed_double
    // This path is only used for complex<double> runs.
    __m256d accum = _mm256_setr_pd(0.125, -0.125, 0.25, -0.25);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 4) {
            __m256d lhs_vec = _mm256_loadu_pd(lhs + i);
            __m256d rhs_vec = _mm256_loadu_pd(rhs + i);
            __m256d lhs_real = _mm256_movedup_pd(lhs_vec);
            __m256d lhs_imag = _mm256_permute_pd(lhs_vec, 0xF);
            __m256d rhs_swapped = _mm256_permute_pd(rhs_vec, 0x5);

            __m256d prod0 = _mm256_mul_pd(lhs_real, rhs_vec);
            __m256d prod1 = _mm256_mul_pd(lhs_imag, rhs_swapped);
            __m256d complex_value = _mm256_addsub_pd(prod0, prod1);

            accum = _mm256_add_pd(accum, complex_value);
            _mm256_storeu_pd(out + i, _mm256_add_pd(complex_value, accum));
        }
    }

    alignas(32) std::array<double, 4> lanes{};
    _mm256_store_pd(lanes.data(), accum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] + out[count - 1];
}

__attribute__((target("avx2,fma"), noinline))
double runAvx2FmaKernel(double* lhs, double* rhs, double* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_256b_packed_double
    // Uses FMA instructions; counts are still reflected in retired FP arithmetic events.
    __m256d accum = _mm256_set1_pd(0.125);
    __m256d shift = _mm256_setr_pd(0.25, 0.5, 0.75, 1.0);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 8) {
            __m256d lhs0 = _mm256_loadu_pd(lhs + i);
            __m256d rhs0 = _mm256_loadu_pd(rhs + i);
            __m256d lhs1 = _mm256_loadu_pd(lhs + i + 4);
            __m256d rhs1 = _mm256_loadu_pd(rhs + i + 4);

            __m256d fused0 = _mm256_fmadd_pd(lhs0, rhs0, shift);
            __m256d fused1 = _mm256_fnmadd_pd(lhs1, rhs1, shift);

            accum = _mm256_add_pd(accum, _mm256_add_pd(fused0, fused1));

            _mm256_storeu_pd(out + i, _mm256_add_pd(fused0, accum));
            _mm256_storeu_pd(out + i + 4, _mm256_sub_pd(fused1, accum));
        }
    }

    alignas(32) std::array<double, 4> lanes{};
    _mm256_store_pd(lanes.data(), accum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] + out[count - 3];
}

__attribute__((target("avx2,fma"), noinline))
double runAvx2FmaSingleKernel(float* lhs, float* rhs, float* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_256b_packed_single
    // Uses FMA instructions; counts are still reflected in retired FP arithmetic events.
    __m256 accum = _mm256_set1_ps(0.125f);
    __m256 shift = _mm256_setr_ps(0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 16) {
            __m256 lhs0 = _mm256_loadu_ps(lhs + i);
            __m256 rhs0 = _mm256_loadu_ps(rhs + i);
            __m256 lhs1 = _mm256_loadu_ps(lhs + i + 8);
            __m256 rhs1 = _mm256_loadu_ps(rhs + i + 8);

            __m256 fused0 = _mm256_fmadd_ps(lhs0, rhs0, shift);
            __m256 fused1 = _mm256_fnmadd_ps(lhs1, rhs1, shift);

            accum = _mm256_add_ps(accum, _mm256_add_ps(fused0, fused1));

            _mm256_storeu_ps(out + i, _mm256_add_ps(fused0, accum));
            _mm256_storeu_ps(out + i + 8, _mm256_sub_ps(fused1, accum));
        }
    }

    alignas(32) std::array<float, 8> lanes{};
    _mm256_store_ps(lanes.data(), accum);
    double sum = static_cast<double>(out[count - 3]);
    for (float value : lanes) {
        sum += value;
    }
    return sum;
}

__attribute__((target("avx512f"), noinline))
double runAvx512Kernel(double* lhs, double* rhs, double* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_512b_packed_double
    __m512d accum = _mm512_set1_pd(0.0625);
    __m512d blend = _mm512_setr_pd(1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0);

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 16) {
            __m512d lhs0 = _mm512_loadu_pd(lhs + i);
            __m512d rhs0 = _mm512_loadu_pd(rhs + i);
            __m512d lhs1 = _mm512_loadu_pd(lhs + i + 8);
            __m512d rhs1 = _mm512_loadu_pd(rhs + i + 8);

            __m512d mix0 = _mm512_add_pd(_mm512_mul_pd(lhs0, rhs0), blend);
            __m512d mix1 = _mm512_sub_pd(_mm512_mul_pd(lhs1, rhs1), blend);

            accum = _mm512_add_pd(accum, _mm512_add_pd(mix0, mix1));

            _mm512_storeu_pd(out + i, _mm512_add_pd(mix0, accum));
            _mm512_storeu_pd(out + i + 8, _mm512_sub_pd(mix1, accum));
        }
    }

    alignas(64) std::array<double, 8> lanes{};
    _mm512_store_pd(lanes.data(), accum);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3]
      + lanes[4] + lanes[5] + lanes[6] + lanes[7] + out[count - 4];
}

__attribute__((target("avx512f"), noinline))
double runAvx512SingleKernel(float* lhs, float* rhs, float* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_512b_packed_single
    __m512 accum = _mm512_set1_ps(0.0625f);
    __m512 blend = _mm512_setr_ps(
      1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f,
      5.0f, -5.0f, 6.0f, -6.0f, 7.0f, -7.0f, 8.0f, -8.0f
    );

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; i += 32) {
            __m512 lhs0 = _mm512_loadu_ps(lhs + i);
            __m512 rhs0 = _mm512_loadu_ps(rhs + i);
            __m512 lhs1 = _mm512_loadu_ps(lhs + i + 16);
            __m512 rhs1 = _mm512_loadu_ps(rhs + i + 16);

            __m512 mix0 = _mm512_add_ps(_mm512_mul_ps(lhs0, rhs0), blend);
            __m512 mix1 = _mm512_sub_ps(_mm512_mul_ps(lhs1, rhs1), blend);

            accum = _mm512_add_ps(accum, _mm512_add_ps(mix0, mix1));

            _mm512_storeu_ps(out + i, _mm512_add_ps(mix0, accum));
            _mm512_storeu_ps(out + i + 16, _mm512_sub_ps(mix1, accum));
        }
    }

    alignas(64) std::array<float, 16> lanes{};
    _mm512_store_ps(lanes.data(), accum);
    double sum = static_cast<double>(out[count - 4]);
    for (float value : lanes) {
        sum += value;
    }
    return sum;
}

#endif

double runScalarPerfKernel(double* lhs, double* rhs, double* out, int count, int loops) {
    // Primarily drives:
    // - fp_arith_inst_retired_scalar_double
    // Also contributes to generic events such as instructions/cycles.
    double accum0 = 0.25;
    double accum1 = 0.75;
    double accum2 = -0.5;

    for (int loop = 0; loop < loops; ++loop) {
        for (int i = 0; i < count; ++i) {
            double value0 = lhs[i] * rhs[i] + accum0;
            double value1 = lhs[i] - rhs[i] * accum1;
            accum2 += (value0 * value0) - (value1 * value1);
            out[i] = value0 + value1 + accum2;
            accum0 += value0 * 1.0e-8;
            accum1 += value1 * 1.0e-8;
        }
    }

    return accum0 + accum1 + accum2 + out[count - 1];
}

template <typename T>
double runPerfInstructionMix(
    double* lhs, double* rhs, double* out,
    float* lhs_f, float* rhs_f, float* out_f
) {
    // This intentionally mixes scalar/SSE/AVX/AVX2/FMA/AVX-512 kernels so that
    // a wide VT_EVENTS list can be exercised in one short benchmark window.
    // ISA-specific counters remain near zero when that ISA is unavailable.
    double checksum = 0.0;
    int const count = kPerfVectorLength;
    int const loops = perfLoopCount();

    checksum += runScalarPerfKernel(lhs, rhs, out, count, loops / 8);

#if SLOW_NODE_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
    SimdCapabilities const capabilities = detectSimdCapabilities();

    if (capabilities.sse2) {
        checksum += runSseScalarKernel(lhs, rhs, out, lhs_f, rhs_f, out_f, count, loops / 8);
        checksum += runSsePackedKernel(lhs, rhs, out, lhs_f, rhs_f, out_f, count, loops / 8);
    }

    if (capabilities.avx) {
        checksum += runAvxKernel(lhs, rhs, out, count, loops / 8);
        checksum += runAvxSingleKernel(lhs_f, rhs_f, out_f, count, loops / 8);
        if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
            checksum += runAvxComplexKernel(lhs, rhs, out, count, loops / 8);
        }
    }

    if (capabilities.avx2_fma) {
        checksum += runAvx2FmaKernel(lhs, rhs, out, count, loops / 8);
        checksum += runAvx2FmaSingleKernel(lhs_f, rhs_f, out_f, count, loops / 8);
    }

    if (capabilities.avx512f) {
        checksum += runAvx512Kernel(lhs, rhs, out, count, loops / 8);
        checksum += runAvx512SingleKernel(lhs_f, rhs_f, out_f, count, loops / 8);
    }
#endif

    return checksum;
}

template <typename T>
benchmark_results_t runTimedPerfBenchmark(char const* benchmark_name, int iters) {
    std::vector<double> lhs(kPerfVectorLength);
    std::vector<double> rhs(kPerfVectorLength);
    std::vector<double> out(kPerfVectorLength, 0.0);
    std::vector<float> lhs_f(kPerfVectorLength);
    std::vector<float> rhs_f(kPerfVectorLength);
    std::vector<float> out_f(kPerfVectorLength, 0.0f);

    for (int i = 0; i < kPerfVectorLength; ++i) {
        lhs[i] = 1.0 + static_cast<double>((i % 17) + 1) * 0.125;
        rhs[i] = 0.5 + static_cast<double>((i % 11) + 1) * 0.0625;
        lhs_f[i] = 1.0f + static_cast<float>((i % 13) + 1) * 0.125f;
        rhs_f[i] = 0.5f + static_cast<float>((i % 7) + 1) * 0.0625f;
        if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
            if ((i % 2) == 1) {
                lhs[i] *= -1.0;
                rhs[i] *= 0.75;
                lhs_f[i] *= -1.0f;
                rhs_f[i] *= 0.75f;
            }
        }
    }

    std::vector<double> iter_timings;
    iter_timings.reserve(iters > kPerfWarmupIters ? iters - kPerfWarmupIters : 0);
    double total_time = 0.0;
    double checksum = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    perf::startMeasurements();

    for (int i = 0; i < iters; ++i) {
        Kokkos::Timer timer;
        checksum += runPerfInstructionMix<T>(
          lhs.data(), rhs.data(), out.data(), lhs_f.data(), rhs_f.data(), out_f.data()
        );

        if (i >= kPerfWarmupIters) {
            double const elapsed = timer.seconds();
            total_time += elapsed;
            iter_timings.push_back(elapsed);
        }
    }

    perf::stopMeasurements(benchmark_name);
    doNotOptimize(checksum);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "[perf " << benchmark_name << "] rank: " << rank
              << ", total_time=" << total_time
              << ", checksum=" << checksum << std::endl;

    return std::make_tuple(iter_timings, total_time);
}

} // namespace

template <>
benchmark_results_t runBenchmarkPerf<double>(int iters) {
    return runTimedPerfBenchmark<double>("double", iters);
}

template <>
benchmark_results_t runBenchmarkPerf<Kokkos::complex<double>>(int iters) {
    return runTimedPerfBenchmark<Kokkos::complex<double>>("complex", iters);
}

} // namespace benchmarks
