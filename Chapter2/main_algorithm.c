#include <stdio.h>
#include <time.h>
#include <stdlib.h>


double dgemm (int n, double* A, double* B, double* C)
{    
    clock_t start = clock();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i+j*n]; /* cij = C[i][j] */
            for (int k = 0; k < n; k++)
                cij += A[i+k*n] * B[k+j*n]; /* cij += A[i][k]*B[k][j] */
            C[i+j*n] = cij; /* C[i][j] = cij */
        }
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}



int main(int argc, char *argv[]) {
    double time_spent = 0.0;
    double end_time = argc > 2 ? atof(argv[2]) : 60.0;
    int n = argc > 1 ? atoi(argv[1]) : 512;
    int multiplication_count = 0;

    if (n <= 0 || end_time <= 0.0) {
        fprintf(stderr, "Usage: %s [matrix_size] [seconds]\n", argv[0]);
        return 1;
    }

    printf("Performing matrix multiplications of size %dx%d for %f seconds.\n", n, n, end_time);

    srand(time(NULL));

    double *A = malloc((long long)n*n*sizeof(double));
    double *B = malloc((long long)n*n*sizeof(double));
    double *C = calloc((long long)n*n,sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed for n = %d\n", n);
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        return 1;
    }

    for (int i = 0; i < n*n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    while (time_spent < end_time) {
        time_spent += dgemm(n, A, B, C);
        multiplication_count++;
    }

    free(A); free(B); free(C);

    printf("\n-------------------------------------------------\n");
    printf("Fixed N: %d\n", n);
    printf("Number of multiplications performed: %d\n", multiplication_count);
    printf("Total computation time: %.2f seconds\n", time_spent);
    printf("-------------------------------------------------\n");

    return 0;
}