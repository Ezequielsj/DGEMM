#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"

int main() {
    double time_spent = 0.0;
    double end_time = 1 * 60; // 1 minute in seconds
    int n = 512;
    int multiplication_count = 0;

    printf("Starting MKL DGEMM benchmark with fixed N = %d for %.f seconds.\n", n, end_time);

    double *A, *B, *C;
    double alpha = 1.0, beta = 0.0;

    // MKL recommends using mkl_malloc for better memory alignment
    A = (double *)mkl_malloc(n * n * sizeof(double), 64);
    B = (double *)mkl_malloc(n * n * sizeof(double), 64);
    C = (double *)mkl_malloc(n * n * sizeof(double), 64);

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed for n = %d\n", n);
        if (A) mkl_free(A);
        if (B) mkl_free(B);
        if (C) mkl_free(C);
        return 1;
    }

    // Initialize matrices A and B with random values
    for (int i = 0; i < (n * n); i++) {
        A[i] = (double)(i + 1);
        B[i] = (double)(n * n - i);
    }

    while (time_spent < end_time) {
        double start_dgemm, end_dgemm;
        
        // --- Time the dgemm operation ---
        start_dgemm = dsecnd();

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, alpha, A, n, B, n, beta, C, n);

        end_dgemm = dsecnd();
        // --------------------------------

        time_spent += end_dgemm - start_dgemm;
        multiplication_count++;
    }

    // Free memory
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf("\n-------------------------------------------------\n");
    printf("Fixed N: %d\n", n);
    printf("Number of multiplications performed: %d\n", multiplication_count);
    printf("Total computation time: %.2f seconds\n", time_spent);
    printf("-------------------------------------------------\n");

    return 0;
}