#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
#include <omp.h> // Include OpenMP header

#define UNROLL (4)
#define BLOCKSIZE 32

double dgemm (size_t n, double* A, double* B, double* C)
{
    double start_time = omp_get_wtime();
    // OpenMP parallelization on the outermost loop (sj)
    #pragma omp parallel for
    for (size_t sj = 0; sj < n; sj += BLOCKSIZE)
        for (size_t si = 0; si < n; si += BLOCKSIZE)
            for (size_t sk = 0; sk < n; sk += BLOCKSIZE)
                // Inlined do_block function
                for (size_t i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
                {
                    for (size_t j = sj; j < sj + BLOCKSIZE; j++)
                    {
                        __m256d c[UNROLL];
                        for (int r = 0; r < UNROLL; r++)
                        {
                            c[r] = _mm256_load_pd(C + i + r * 4 + j * n);
                        }

                        for (size_t k = sk; k < sk + BLOCKSIZE; k++)
                        {
                            __m256d bb = _mm256_broadcast_sd(B + k + j * n);
                            for (int r = 0; r < UNROLL; r++)
                            {
                                __m256d aa = _mm256_load_pd(A + k * n + r * 4 + i);
                                c[r] = _mm256_fmadd_pd(aa, bb, c[r]);
                            }
                        }

                        for (int r = 0; r < UNROLL; r++)
                        {
                            _mm256_store_pd(C + i + r * 4 + j * n, c[r]);
                        }
                    }
                }
    double end_time = omp_get_wtime();
    return end_time - start_time;
}



int main(int argc, char *argv[]) {
    double time_spent = 0.0;
    double end_time = argc > 2 ? atof(argv[2]) : 60.0;
    size_t n = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 512;
    int multiplication_count = 0;

    if (n == 0 || n % 32 != 0 || end_time <= 0.0) {
        fprintf(stderr, "Usage: %s [matrix_size_multiple_of_32] [seconds]\n", argv[0]);
        return 1;
    }

        printf("Performing matrix multiplications of size %I64ux%I64u for %f seconds.\n",
            (unsigned long long)n, (unsigned long long)n, end_time);

    srand(time(NULL));

    // Align memory for AVX
    double *A = _aligned_malloc(n*n*sizeof(double), 32);
    double *B = _aligned_malloc(n*n*sizeof(double), 32);
    double *C = _aligned_malloc(n*n*sizeof(double), 32);

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed for n = %I64u\n", (unsigned long long)n);
        if (A) _aligned_free(A);
        if (B) _aligned_free(B);
        if (C) _aligned_free(C);
        return 1;
    }

    // Initialize C to zeros and A, B with random values
    for (size_t i = 0; i < n*n; i++) {
        C[i] = 0.0;
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    while (time_spent < end_time) {
        time_spent += dgemm(n, A, B, C);
        multiplication_count++;
    }

    _aligned_free(A); 
    _aligned_free(B); 
    _aligned_free(C);

    printf("\n-------------------------------------------------\n");
    printf("Fixed N: %I64u\n", (unsigned long long)n);
    printf("Number of multiplications performed: %d\n", multiplication_count);
    printf("Total computation time: %.2f seconds\n", time_spent);
    printf("-------------------------------------------------\n");

    return 0;
}
