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
        __m256d sum = _mm256_setzero_pd();

        for (int j = 0; j < n; j += 4) {
            __m256d a_vec = _mm256_load_pd(&A[i * n + j]);
            __m256d x_vec = _mm256_loadu_pd(&x[j]);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(a_vec, x_vec));
        }

        // Horizontal sum using hadd
        sum = _mm256_hadd_pd(sum, sum);
        sum = _mm256_hadd_pd(sum, sum);

        // Store the result in the output vector y
        y[i] += _mm256_cvtsd_f64(sum);
    }
}
