#include <immintrin.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 32
#define UNROLL 4

static void fill_matrix(size_t n, double *matrix, unsigned int seed)
{
    srand(seed);
    for (size_t index = 0; index < n * n; ++index)
        matrix[index] = (double)rand() / RAND_MAX;
}

static void clear_matrix(size_t n, double *matrix)
{
    for (size_t index = 0; index < n * n; ++index)
        matrix[index] = 0.0;
}

static void reference_dgemm(size_t n, const double *a, const double *b, double *c)
{
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                c[i + j * n] += a[i + k * n] * b[k + j * n];
}

static void avx2_dgemm(size_t n, const double *a, const double *b, double *c)
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

static void unrolled_dgemm(size_t n, const double *a, const double *b, double *c)
{
    for (size_t i = 0; i < n; i += UNROLL * 4)
        for (size_t j = 0; j < n; ++j) {
            __m256d accumulators[UNROLL];
            for (int register_index = 0; register_index < UNROLL; ++register_index)
                accumulators[register_index] = _mm256_load_pd(c + i + register_index * 4 + j * n);
            for (size_t k = 0; k < n; ++k) {
                __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                for (int register_index = 0; register_index < UNROLL; ++register_index) {
                    __m256d values_a = _mm256_load_pd(a + k * n + i + register_index * 4);
                    accumulators[register_index] = _mm256_fmadd_pd(values_a, value_b, accumulators[register_index]);
                }
            }
            for (int register_index = 0; register_index < UNROLL; ++register_index)
                _mm256_store_pd(c + i + register_index * 4 + j * n, accumulators[register_index]);
        }
}

static void blocking_dgemm(size_t n, const double *a, const double *b, double *c)
{
    for (size_t sj = 0; sj < n; sj += BLOCKSIZE)
        for (size_t si = 0; si < n; si += BLOCKSIZE)
            for (size_t sk = 0; sk < n; sk += BLOCKSIZE)
                for (size_t i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
                    for (size_t j = sj; j < sj + BLOCKSIZE; ++j) {
                        __m256d accumulators[UNROLL];
                        for (int register_index = 0; register_index < UNROLL; ++register_index)
                            accumulators[register_index] = _mm256_load_pd(c + i + register_index * 4 + j * n);
                        for (size_t k = sk; k < sk + BLOCKSIZE; ++k) {
                            __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                            for (int register_index = 0; register_index < UNROLL; ++register_index) {
                                __m256d values_a = _mm256_load_pd(a + k * n + i + register_index * 4);
                                accumulators[register_index] = _mm256_fmadd_pd(values_a, value_b, accumulators[register_index]);
                            }
                        }
                        for (int register_index = 0; register_index < UNROLL; ++register_index)
                            _mm256_store_pd(c + i + register_index * 4 + j * n, accumulators[register_index]);
                    }
}

static void openmp_dgemm(size_t n, const double *a, const double *b, double *c)
{
    #pragma omp parallel for
    for (size_t sj = 0; sj < n; sj += BLOCKSIZE)
        for (size_t si = 0; si < n; si += BLOCKSIZE)
            for (size_t sk = 0; sk < n; sk += BLOCKSIZE)
                for (size_t i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
                    for (size_t j = sj; j < sj + BLOCKSIZE; ++j) {
                        __m256d accumulators[UNROLL];
                        for (int register_index = 0; register_index < UNROLL; ++register_index)
                            accumulators[register_index] = _mm256_load_pd(c + i + register_index * 4 + j * n);
                        for (size_t k = sk; k < sk + BLOCKSIZE; ++k) {
                            __m256d value_b = _mm256_broadcast_sd(b + k + j * n);
                            for (int register_index = 0; register_index < UNROLL; ++register_index) {
                                __m256d values_a = _mm256_load_pd(a + k * n + i + register_index * 4);
                                accumulators[register_index] = _mm256_fmadd_pd(values_a, value_b, accumulators[register_index]);
                            }
                        }
                        for (int register_index = 0; register_index < UNROLL; ++register_index)
                            _mm256_store_pd(c + i + register_index * 4 + j * n, accumulators[register_index]);
                    }
}

static double max_error(size_t n, const double *expected, const double *actual)
{
    double largest_error = 0.0;
    for (size_t index = 0; index < n * n; ++index) {
        double error = fabs(expected[index] - actual[index]);
        if (error > largest_error)
            largest_error = error;
    }
    return largest_error;
}

static int check_variant(const char *name, size_t n,
                        void (*multiply)(size_t, const double *, const double *, double *),
                        const double *a, const double *b, const double *reference)
{
    double *result = _aligned_malloc(n * n * sizeof(double), 32);
    if (result == NULL)
        return 1;
    clear_matrix(n, result);
    multiply(n, a, b, result);
    double error = max_error(n, reference, result);
    printf("%s: max_error=%.3e %s\n", name, error, error <= 1e-12 ? "PASS" : "FAIL");
    _aligned_free(result);
    return error <= 1e-12 ? 0 : 1;
}

int main(void)
{
    const size_t n = 32;
    double *a = _aligned_malloc(n * n * sizeof(double), 32);
    double *b = _aligned_malloc(n * n * sizeof(double), 32);
    double *reference = _aligned_malloc(n * n * sizeof(double), 32);
    if (a == NULL || b == NULL || reference == NULL)
        return 1;

    fill_matrix(n, a, 1234);
    fill_matrix(n, b, 5678);
    clear_matrix(n, reference);
    reference_dgemm(n, a, b, reference);

    int failures = 0;
    failures += check_variant("AVX2", n, avx2_dgemm, a, b, reference);
    failures += check_variant("AVX2 + FMA + unrolling", n, unrolled_dgemm, a, b, reference);
    failures += check_variant("Blocking", n, blocking_dgemm, a, b, reference);
    failures += check_variant("OpenMP", n, openmp_dgemm, a, b, reference);

    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(reference);
    return failures == 0 ? 0 : 1;
}
