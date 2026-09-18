#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

#define UNROLL (4)
#define BLOCKSIZE 32

void do_block (size_t n, size_t si, size_t sj, size_t sk,
               double *A, double *B, double *C)
{
    // The i-loop stride must be UNROLL * 4 = 16 for AVX2 (4 doubles per register)
    for (size_t i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
    {
        for (size_t j = sj; j < sj + BLOCKSIZE; j++)
        {
            // Declare UNROLL (4) AVX2 registers for C[i..i+15][j]
            __m256d c[UNROLL];

            // Load the initial values of C[i..i+15][j]
            for (int r = 0; r < UNROLL; r++)
            {
                // Each load handles 4 doubles: offset r * 4
                c[r] = _mm256_load_pd(C + i + r * 4 + j * n);
            }

            for (size_t k = sk; k < sk + BLOCKSIZE; k++)
            {
                // Broadcast B[k][j] (B[k + j*n])
                __m256d bb = _mm256_broadcast_sd(B + k + j * n);

                for (int r = 0; r < UNROLL; r++)
                {
                    // Load A[i + r*4 .. i + r*4 + 3][k]
                    __m256d aa = _mm256_load_pd(A + k * n + r * 4 + i);

                    // Fused Multiply-Add: c[r] += aa * bb
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
}

double dgemm (size_t n, double* A, double* B, double* C)
{
    clock_t start = clock();
    for (size_t sj = 0; sj < n; sj += BLOCKSIZE)
        for (size_t si = 0; si < n; si += BLOCKSIZE)
            for (size_t sk = 0; sk < n; sk += BLOCKSIZE)
                do_block(n, si, sj, sk, A, B, C);
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
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
