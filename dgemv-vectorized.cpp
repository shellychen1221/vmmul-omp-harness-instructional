#include <immintrin.h>
const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    // Assuming AVX (256-bit) for illustration, adjust based on your architecture

    // Ensure the matrix size is a multiple of the SIMD width (e.g., 4 for AVX)
    int simd_width = 4;
    int aligned_n = (n / simd_width) * simd_width;

    // Vectorized loop
    for (int i = 0; i < aligned_n; i += simd_width) {
        __m256d acc = _mm256_setzero_pd();

        for (int j = 0; j < n; ++j) {
            acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(&A[i * n + j]), _mm256_broadcast_sd(&x[j])));
        }

        _mm256_storeu_pd(&y[i], _mm256_add_pd(_mm256_loadu_pd(&y[i]), acc));
    }

    // Handle the remaining elements
    for (int i = aligned_n; i < n; ++i) {
        y[i] += A[i * n] * x[0];
    }
}
