#include <immintrin.h>
const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
   // Ensure proper alignment for efficient memory access
    alignas(16) double acc[4] = {0.0};  // Assuming AVX width of 4

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            acc[0] += A[i * n + j] * x[j];
        }

        // Accumulate the results into the output vector y
        y[i] += acc[0];
        acc[0] = 0.0;  // Reset accumulator for the next iteration
    }
}
