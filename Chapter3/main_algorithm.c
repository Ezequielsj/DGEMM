#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <malloc.h>

double dgemm (size_t n, double* A, double* B, double* C)
{
    clock_t start = clock();
    for ( size_t i = 0; i < n; i+=4 )
        for ( size_t j = 0; j < n; j++ ) {
            __m256d c0 = _mm256_load_pd(C+i+j*n); /* c0 = C[i][j] */
            for( size_t k = 0; k < n; k++ )
                c0 = _mm256_add_pd(c0, /* c0 += A[i][k]*B[k][j] */
                _mm256_mul_pd(_mm256_load_pd(A+i+k*n),
                _mm256_broadcast_sd(B+k+j*n)));
            _mm256_store_pd(C+i+j*n, c0); /* C[i][j] = c0 */
        }
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main(int argc, char *argv[]) {
    double time_spent = 0.0;
    double end_time = argc > 2 ? atof(argv[2]) : 60.0;
    size_t n = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 512;
    int multiplication_count = 0;

    if (n == 0 || n % 4 != 0 || end_time <= 0.0) {
        fprintf(stderr, "Usage: %s [matrix_size_multiple_of_4] [seconds]\n", argv[0]);
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
