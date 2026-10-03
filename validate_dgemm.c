#define _POSIX_C_SOURCE 200809L
#include "dgemm_cli.h"

#include <math.h>
#include <stdint.h>

static double random_value(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return (double)(*state >> 8) / 16777216.0 - 0.5;
}

static void fill_matrices(size_t elements, double *a, double *b, uint32_t seed)
{
    for (size_t index = 0; index < elements; ++index) {
        a[index] = random_value(&seed);
        b[index] = random_value(&seed);
    }
}

static void reference_dgemm(size_t n, const double *a, const double *b, double *c)
{
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < n; ++k) {
            double value_b = b[k + j * n];
            for (size_t i = 0; i < n; ++i)
                c[i + j * n] += a[i + k * n] * value_b;
        }
}

static double max_error(size_t elements, const double *expected, const double *actual)
{
    double error = 0.0;
    for (size_t index = 0; index < elements; ++index) {
        double current = fabs(expected[index] - actual[index]);
        if (current > error)
            error = current;
    }
    return error;
}

static int check_variant(const char *name, size_t n, size_t block_size, int threads,
                         void (*kernel)(size_t, size_t, int, const double *,
                                        const double *, double *),
                         const double *a, const double *b, const double *reference)
{
    size_t elements = n * n;
    double *result = dgemm_aligned_allocate(elements * sizeof(*result));
    if (result == NULL)
        return 1;
    memset(result, 0, elements * sizeof(*result));
    kernel(n, block_size, threads, a, b, result);
    double error = max_error(elements, reference, result);
    int passed = error <= 1e-10;
    printf("N=%zu,%s,block=%zu,threads=%d,max_abs_error=%.12e,%s\n",
           n, name, block_size, threads, error, passed ? "PASS" : "FAIL");
    dgemm_aligned_release(result);
    return passed ? 0 : 1;
}

static void run_blocked(size_t n, size_t block_size, int threads,
                        const double *a, const double *b, double *c)
{
    (void)threads;
    dgemm_blocked(n, block_size, a, b, c);
}

static void run_openmp(size_t n, size_t block_size, int threads,
                       const double *a, const double *b, double *c)
{
    dgemm_openmp(n, block_size, threads, a, b, c);
}

static int check_simple(const char *name, size_t n,
                        void (*kernel)(size_t, const double *, const double *, double *),
                        const double *a, const double *b, const double *reference)
{
    size_t elements = n * n;
    double *result = dgemm_aligned_allocate(elements * sizeof(*result));
    if (result == NULL)
        return 1;
    memset(result, 0, elements * sizeof(*result));
    kernel(n, a, b, result);
    double error = max_error(elements, reference, result);
    int passed = error <= 1e-10;
    printf("N=%zu,%s,max_abs_error=%.12e,%s\n",
           n, name, error, passed ? "PASS" : "FAIL");
    dgemm_aligned_release(result);
    return passed ? 0 : 1;
}

int main(void)
{
    const size_t sizes[] = {32, 128, 256};
    const size_t block_sizes[] = {8, 16, 32, 64};
    const int thread_counts[] = {1, 2, 4};
    int failures = 0;

    for (size_t size_index = 0; size_index < sizeof(sizes) / sizeof(sizes[0]); ++size_index) {
        size_t n = sizes[size_index];
        size_t elements = n * n;
        double *a = dgemm_aligned_allocate(elements * sizeof(*a));
        double *b = dgemm_aligned_allocate(elements * sizeof(*b));
        double *reference = dgemm_aligned_allocate(elements * sizeof(*reference));
        if (a == NULL || b == NULL || reference == NULL) {
            fprintf(stderr, "Falha ao alocar matrizes para N=%zu\n", n);
            dgemm_aligned_release(a);
            dgemm_aligned_release(b);
            dgemm_aligned_release(reference);
            return 2;
        }
        fill_matrices(elements, a, b, 20261003u + (uint32_t)n);
        memset(reference, 0, elements * sizeof(*reference));
        reference_dgemm(n, a, b, reference);

        failures += check_simple("scalar", n, dgemm_scalar, a, b, reference);
        failures += check_simple("avx2", n, dgemm_avx2, a, b, reference);
        failures += check_simple("avx2_fma", n, dgemm_fma, a, b, reference);
        failures += check_simple("avx2_fma_unroll4", n, dgemm_fma_unroll,
                                 a, b, reference);
        for (size_t block_index = 0;
             block_index < sizeof(block_sizes) / sizeof(block_sizes[0]); ++block_index) {
            size_t block_size = block_sizes[block_index];
            if (block_size > n)
                break;
            failures += check_variant("blocked_fma", n, block_size, 1,
                                      run_blocked, a, b, reference);
            for (size_t thread_index = 0;
                 thread_index < sizeof(thread_counts) / sizeof(thread_counts[0]);
                 ++thread_index)
                failures += check_variant("openmp_blocked_fma", n, block_size,
                                          thread_counts[thread_index], run_openmp,
                                          a, b, reference);
        }
        dgemm_aligned_release(a);
        dgemm_aligned_release(b);
        dgemm_aligned_release(reference);
    }

    printf("Resultado geral: %s (tolerancia absoluta 1e-10)\n",
           failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}