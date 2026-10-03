#ifndef DGEMM_KERNELS_H
#define DGEMM_KERNELS_H

#include <immintrin.h>
#include <stddef.h>
#include <omp.h>

static inline void dgemm_scalar(size_t n, const double *a, const double *b, double *c)
{
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = c[i + j * n];
            for (size_t k = 0; k < n; ++k)
                sum += a[i + k * n] * b[k + j * n];
            c[i + j * n] = sum;
        }
}

static inline void dgemm_avx2(size_t n, const double *a, const double *b, double *c)
{
    for (size_t i = 0; i < n; i += 4)
        for (size_t j = 0; j < n; ++j) {
            __m256d accumulator = _mm256_load_pd(c + i + j * n);
            for (size_t k = 0; k < n; ++k) {
                __m256d values_a = _mm256_load_pd(a + i + k * n);
                __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                accumulator = _mm256_add_pd(accumulator, _mm256_mul_pd(values_a, value_b));
            }
            _mm256_store_pd(c + i + j * n, accumulator);
        }
}

static inline void dgemm_fma(size_t n, const double *a, const double *b, double *c)
{
    for (size_t i = 0; i < n; i += 4)
        for (size_t j = 0; j < n; ++j) {
            __m256d accumulator = _mm256_load_pd(c + i + j * n);
            for (size_t k = 0; k < n; ++k) {
                __m256d values_a = _mm256_load_pd(a + i + k * n);
                __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                accumulator = _mm256_fmadd_pd(values_a, value_b, accumulator);
            }
            _mm256_store_pd(c + i + j * n, accumulator);
        }
}

static inline void dgemm_fma_unroll(size_t n, const double *a, const double *b, double *c)
{
    for (size_t i = 0; i < n; i += 16)
        for (size_t j = 0; j < n; ++j) {
            __m256d accumulators[4];
            for (size_t r = 0; r < 4; ++r)
                accumulators[r] = _mm256_load_pd(c + i + r * 4 + j * n);
            for (size_t k = 0; k < n; ++k) {
                __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                for (size_t r = 0; r < 4; ++r) {
                    __m256d values_a = _mm256_load_pd(a + i + r * 4 + k * n);
                    accumulators[r] = _mm256_fmadd_pd(values_a, value_b, accumulators[r]);
                }
            }
            for (size_t r = 0; r < 4; ++r)
                _mm256_store_pd(c + i + r * 4 + j * n, accumulators[r]);
        }
}

static inline void dgemm_blocked(size_t n, size_t block_size,
                                 const double *a, const double *b, double *c)
{
    for (size_t sj = 0; sj < n; sj += block_size)
        for (size_t si = 0; si < n; si += block_size)
            for (size_t sk = 0; sk < n; sk += block_size)
                for (size_t i = si; i < si + block_size; i += 4)
                    for (size_t j = sj; j < sj + block_size; ++j) {
                        __m256d accumulator = _mm256_load_pd(c + i + j * n);
                        for (size_t k = sk; k < sk + block_size; ++k) {
                            __m256d values_a = _mm256_load_pd(a + i + k * n);
                            __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                            accumulator = _mm256_fmadd_pd(values_a, value_b, accumulator);
                        }
                        _mm256_store_pd(c + i + j * n, accumulator);
                    }
}

static inline void dgemm_openmp(size_t n, size_t block_size, int threads,
                                const double *a, const double *b, double *c)
{
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t sj = 0; sj < n; sj += block_size)
        for (size_t si = 0; si < n; si += block_size)
            for (size_t sk = 0; sk < n; sk += block_size)
                for (size_t i = si; i < si + block_size; i += 4)
                    for (size_t j = sj; j < sj + block_size; ++j) {
                        __m256d accumulator = _mm256_load_pd(c + i + j * n);
                        for (size_t k = sk; k < sk + block_size; ++k) {
                            __m256d values_a = _mm256_load_pd(a + i + k * n);
                            __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                            accumulator = _mm256_fmadd_pd(values_a, value_b, accumulator);
                        }
                        _mm256_store_pd(c + i + j * n, accumulator);
                    }
}

#endif