#include "ops.hpp"

#include <stdexcept>

Matrix matmul_cpu(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows)
        throw std::invalid_argument("matmul: A.cols != B.rows");

    const std::size_t m = a.rows;
    const std::size_t k = a.cols;
    const std::size_t n = b.cols;

    Matrix c(m, n);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                sum += a(i, p) * b(p, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}
