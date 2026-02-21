#include "ops.hpp"

#include <stdexcept>

Matrix matmul_cpu(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows)
        throw std::invalid_argument("matmul_cpu: A.cols != B.rows");

    const std::size_t M = a.rows;
    const std::size_t K = a.cols;
    const std::size_t N = b.cols;

    Matrix c(M, N);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t p = 0; p < K; ++p) {
                sum += a(i, p) * b(p, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}
