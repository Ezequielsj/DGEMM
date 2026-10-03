#define _POSIX_C_SOURCE 200809L
#include "dgemm_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
static double wall_time(void)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}
static void *aligned_allocate(size_t bytes)
{
    return _aligned_malloc(bytes, 32);
}
static void aligned_release(void *memory)
{
    _aligned_free(memory);
}
#else
static double wall_time(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}
static void *aligned_allocate(size_t bytes)
{
    void *memory = NULL;
    return posix_memalign(&memory, 32, bytes) == 0 ? memory : NULL;
}
static void aligned_release(void *memory)
{
    free(memory);
}
#endif

typedef void (*kernel_fn)(size_t, const double *, const double *, double *);

typedef struct {
    const char *name;
    kernel_fn run;
} simple_variant;

static uint32_t rng_state = 20261003u;

static double random_value(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (double)(rng_state >> 8) / 16777216.0 - 0.5;
}

static void fill_inputs(size_t elements, double *a, double *b)
{
    for (size_t index = 0; index < elements; ++index) {
        a[index] = random_value();
        b[index] = random_value();
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

static void reset_result(size_t elements, double *result)
{
    memset(result, 0, elements * sizeof(*result));
}

static int record_run(FILE *csv, const char *variant, size_t n, size_t block_size,
                      int threads, int repetition, uint32_t seed, size_t elements,
                      const double *reference, double *result,
                      void (*run)(void), double *seconds_out)
{
    reset_result(elements, result);
    double start = wall_time();
    run();
    double seconds = wall_time() - start;
    double error = max_error(elements, reference, result);
    double gflops = (2.0 * (double)n * (double)n * (double)n) / (seconds * 1e9);
    int passed = error <= 1e-10;
        fprintf(csv, "%s,%zu,%zu,%d,%d,%u,%.9f,%.6f,%.12e,%s\n",
            variant, n, block_size, threads, repetition, seed, seconds, gflops,
            error, passed ? "PASS" : "FAIL");
    *seconds_out = seconds;
    return passed ? 0 : 1;
}

static size_t active_n;
static size_t active_block;
static int active_threads;
static const double *active_a;
static const double *active_b;
static double *active_c;
static kernel_fn active_simple;

static void run_simple(void)
{
    active_simple(active_n, active_a, active_b, active_c);
}

static void run_blocked(void)
{
    dgemm_blocked(active_n, active_block, active_a, active_b, active_c);
}

static void run_openmp(void)
{
    dgemm_openmp(active_n, active_block, active_threads,
                 active_a, active_b, active_c);
}

static int benchmark_one(FILE *csv, size_t n, size_t block_size, int threads,
                         int repetitions, int warmups, uint32_t seed,
                         const char *variant,
                         kernel_fn simple, size_t elements, const double *a,
                         const double *b, const double *reference, double *result,
                         void (*run)(void))
{
    active_n = n;
    active_block = block_size;
    active_threads = threads;
    active_a = a;
    active_b = b;
    active_c = result;
    active_simple = simple;

    int failures = 0;
    for (int warmup = 0; warmup < warmups; ++warmup) {
        reset_result(elements, result);
        run();
    }
    for (int repetition = 1; repetition <= repetitions; ++repetition) {
        double seconds = 0.0;
        failures += record_run(csv, variant, n, block_size, threads, repetition,
                               seed, elements, reference, result, run, &seconds);
    }
    return failures;
}

int main(int argc, char **argv)
{
    const size_t sizes[] = {128, 256};
    const size_t block_sizes[] = {8, 16, 32, 64};
    const int thread_counts[] = {1, 2, 4};
    const simple_variant simple_variants[] = {
        {"scalar", dgemm_scalar},
        {"avx2", dgemm_avx2},
        {"avx2_fma", dgemm_fma},
        {"avx2_fma_unroll4", dgemm_fma_unroll}
    };
    int repetitions = argc > 2 ? atoi(argv[2]) : 5;
    int warmups = argc > 3 ? atoi(argv[3]) : 1;
    const char *output_path = argc > 1 ? argv[1] : "resultados_campanha.csv";
    if (repetitions < 1 || warmups < 0) {
        fprintf(stderr, "Uso: %s [arquivo.csv] [repeticoes>=1] [aquecimentos>=0]\n", argv[0]);
        return 2;
    }

    FILE *csv = fopen(output_path, "w");
    if (csv == NULL) {
        perror(output_path);
        return 2;
    }
    fprintf(csv, "variant,n,block_size,threads,repetition,seed,seconds,gflops,max_abs_error,validation\n");

    int failures = 0;
    for (size_t size_index = 0; size_index < sizeof(sizes) / sizeof(sizes[0]); ++size_index) {
        size_t n = sizes[size_index];
        size_t elements = n * n;
        double *a = aligned_allocate(elements * sizeof(*a));
        double *b = aligned_allocate(elements * sizeof(*b));
        double *reference = aligned_allocate(elements * sizeof(*reference));
        double *result = aligned_allocate(elements * sizeof(*result));
        if (a == NULL || b == NULL || reference == NULL || result == NULL) {
            fprintf(stderr, "Falha ao alocar matrizes para N=%zu\n", n);
            aligned_release(a);
            aligned_release(b);
            aligned_release(reference);
            aligned_release(result);
            fclose(csv);
            return 2;
        }

        uint32_t seed = 20261003u + (uint32_t)n;
        rng_state = seed;
        fill_inputs(elements, a, b);
        reset_result(elements, reference);
        reference_dgemm(n, a, b, reference);

        for (size_t variant_index = 0;
             variant_index < sizeof(simple_variants) / sizeof(simple_variants[0]);
             ++variant_index) {
            failures += benchmark_one(csv, n, 0, 1, repetitions, warmups, seed,
                                      simple_variants[variant_index].name,
                                      simple_variants[variant_index].run,
                                      elements, a, b, reference, result, run_simple);
        }

        for (size_t block_index = 0;
             block_index < sizeof(block_sizes) / sizeof(block_sizes[0]); ++block_index) {
            size_t block_size = block_sizes[block_index];
            failures += benchmark_one(csv, n, block_size, 1, repetitions, warmups, seed,
                                      "blocked_fma", NULL, elements, a, b,
                                      reference, result, run_blocked);
            for (size_t thread_index = 0;
                 thread_index < sizeof(thread_counts) / sizeof(thread_counts[0]);
                 ++thread_index) {
                failures += benchmark_one(csv, n, block_size,
                                          thread_counts[thread_index], repetitions,
                                          warmups, seed, "openmp_blocked_fma", NULL,
                                          elements, a, b, reference, result, run_openmp);
            }
        }

        printf("N=%zu: campanha concluida; dados gravados em %s\n", n, output_path);
        aligned_release(a);
        aligned_release(b);
        aligned_release(reference);
        aligned_release(result);
    }

    fclose(csv);
    printf("Validacoes: %s; tolerancia absoluta 1e-10; aquecimento=%d; repeticoes=%d\n",
           failures == 0 ? "PASS" : "FAIL", warmups, repetitions);
    return failures == 0 ? 0 : 1;
}