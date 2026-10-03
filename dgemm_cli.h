#ifndef DGEMM_CLI_H
#define DGEMM_CLI_H

#define _POSIX_C_SOURCE 200809L
#include "dgemm_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
static double dgemm_wall_time(void)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}
static void *dgemm_aligned_allocate(size_t bytes)
{
    return _aligned_malloc(bytes, 32);
}
static void dgemm_aligned_release(void *memory)
{
    _aligned_free(memory);
}
#else
static double dgemm_wall_time(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}
static void *dgemm_aligned_allocate(size_t bytes)
{
    void *memory = NULL;
    return posix_memalign(&memory, 32, bytes) == 0 ? memory : NULL;
}
static void dgemm_aligned_release(void *memory)
{
    free(memory);
}
#endif

static inline int dgemm_cli_run(int argc, char **argv, const char *variant)
{
    size_t n = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 128;
    size_t block_size = (strcmp(variant, "blocked") == 0 || strcmp(variant, "openmp") == 0)
        ? (argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : 32) : 0;
    int threads = strcmp(variant, "openmp") == 0
        ? (argc > 3 ? atoi(argv[3]) : 1) : 1;
    if (n == 0 || n % 4 != 0 ||
        (strcmp(variant, "fma_unroll4") == 0 && n % 16 != 0) ||
        (block_size != 0 && (block_size % 4 != 0 || n % block_size != 0)) ||
        threads < 1) {
        fprintf(stderr, "Uso: %s N [bloco [threads]]; N deve ser multiplo de 4, "
                        "bloco multiplo de 4 e divisor de N.\n", argv[0]);
        return 2;
    }

    size_t elements = n * n;
    double *a = dgemm_aligned_allocate(elements * sizeof(*a));
    double *b = dgemm_aligned_allocate(elements * sizeof(*b));
    double *c = dgemm_aligned_allocate(elements * sizeof(*c));
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Falha ao alocar matrizes para N=%zu\n", n);
        dgemm_aligned_release(a);
        dgemm_aligned_release(b);
        dgemm_aligned_release(c);
        return 2;
    }
    for (size_t index = 0; index < elements; ++index) {
        a[index] = (double)((index * 17 + 11) % 101) / 101.0;
        b[index] = (double)((index * 29 + 7) % 103) / 103.0;
        c[index] = 0.0;
    }

    double start = dgemm_wall_time();
    if (strcmp(variant, "scalar") == 0)
        dgemm_scalar(n, a, b, c);
    else if (strcmp(variant, "avx2") == 0)
        dgemm_avx2(n, a, b, c);
    else if (strcmp(variant, "fma_unroll4") == 0)
        dgemm_fma_unroll(n, a, b, c);
    else if (strcmp(variant, "blocked") == 0)
        dgemm_blocked(n, block_size, a, b, c);
    else if (strcmp(variant, "openmp") == 0)
        dgemm_openmp(n, block_size, threads, a, b, c);
    else {
        fprintf(stderr, "Variante desconhecida: %s\n", variant);
        dgemm_aligned_release(a);
        dgemm_aligned_release(b);
        dgemm_aligned_release(c);
        return 2;
    }
    double seconds = dgemm_wall_time() - start;
    printf("variant=%s,N=%zu,block=%zu,threads=%d,seconds=%.9f,GFLOPS=%.6f\n",
           variant, n, block_size, threads, seconds,
           (2.0 * (double)n * (double)n * (double)n) / (seconds * 1e9));

    dgemm_aligned_release(a);
    dgemm_aligned_release(b);
    dgemm_aligned_release(c);
    return 0;
}

#endif