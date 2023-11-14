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
        __m256d acc_vec = _mm256_setzero_pd();  // Initialize accumulator vector

        for (int j = 0; j < n; j += 4) {
            __m256d A_vec = _mm256_loadu_pd(&A_row[j]);
            __m256d x_vec = _mm256_loadu_pd(&x[j]);
            acc_vec = _mm256_add_pd(acc_vec, _mm256_mul_pd(A_vec, x_vec));
        }

        // Horizontal sum of the accumulator vector
        __m128d acc128 = _mm_add_pd(_mm256_extractf128_pd(acc_vec, 0), _mm256_extractf128_pd(acc_vec, 1));
        acc[0] += acc128[0];
        acc[1] += acc128[1];
    }

    // Accumulate the results into the output vector y
    y[0] += acc[0] + acc[1];
}
