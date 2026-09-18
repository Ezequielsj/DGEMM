#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

#define UNROLL 4  // processes 32 columns at a time

double dgemm (size_t n, double* A, double* B, double* C)
{
    clock_t start = clock();
    // The outer loop advances by UNROLL * 4 (16 elements in the i dimension)
    // because each AVX2 register handles 4 doubles.
    for (size_t i = 0; i < n; i += UNROLL * 4)
    {
        for (size_t j = 0; j < n; ++j)
        {
            // Declare UNROLL (4) AVX2 registers for C[i..i+15][j]
            __m256d c[UNROLL];

            // Load the initial values of C[i..i+15][j]
            for (int r = 0; r < UNROLL; r++)
            {
                // Each load handles 4 doubles: C[i + r*4][j] through C[i + r*4 + 3][j]
                c[r] = _mm256_load_pd(C + i + r * 4 + j * n);
            }

            for (size_t k = 0; k < n; k++)
            {
                // Broadcast B[k][j] (B[k + j*n]) to all 4 lanes of a register
                // We use _mm256_broadcast_sd because B[k][j] is a single scalar.
                __m256d bb = _mm256_broadcast_sd(B + k + j * n);

                for (int r = 0; r < UNROLL; r++)
                {
                    // Load A[i + r*4 .. i + r*4 + 3][k]
                    __m256d aa = _mm256_load_pd(A + k * n + r * 4 + i);

                    // Multiply A * B and accumulate into C: c[r] += aa * bb
                    // AVX2 supports Fused Multiply-Add (FMA) for better performance.
                    c[r] = _mm256_fmadd_pd(aa, bb, c[r]);
                }
            }

            // Store the final accumulated values back into C
            for (int r = 0; r < UNROLL; r++)
            {
                _mm256_store_pd(C + i + r * 4 + j * n, c[r]);
            }
        }
    }
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}


int main(int argc, char *argv[]) {
    double time_spent = 0.0;
    double end_time = argc > 2 ? atof(argv[2]) : 60.0;
    size_t n = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 512;
    int multiplication_count = 0;

    if (n == 0 || n % 16 != 0 || end_time <= 0.0) {
        fprintf(stderr, "Usage: %s [matrix_size_multiple_of_16] [seconds]\n", argv[0]);
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
