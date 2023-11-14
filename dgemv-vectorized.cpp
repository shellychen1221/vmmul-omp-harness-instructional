#include <immintrin.h>
const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    alignas(32) double acc[4] = {0.0};  // Assuming AVX width of 4

    for (int i = 0; i < n; ++i) {
        double* A_row = &A[i * n];

        // Unroll the loop to better facilitate vectorization
        for (int j = 0; j < n; j += 4) {
            __m256d A_vec = _mm256_loadu_pd(&A_row[j]);
            __m256d x_vec = _mm256_loadu_pd(&x[j]);

            __m256d result = _mm256_mul_pd(A_vec, x_vec);
            acc[0] += result[0];
            acc[1] += result[1];
            acc[2] += result[2];
            acc[3] += result[3];
        }

        // Accumulate the results into the output vector y
        y[i] += acc[0] + acc[1] + acc[2] + acc[3];

        // Reset accumulator for the next iteration
        acc[0] = 0.0;
        acc[1] = 0.0;
        acc[2] = 0.0;
        acc[3] = 0.0;
    }
}