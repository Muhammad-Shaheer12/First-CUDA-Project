#include "ops.hpp"

#include <stdexcept>

Matrix add_elementwise(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix c(a.rows, a.cols);
    const std::size_t n = c.data.size();
    for (std::size_t i = 0; i < n; ++i) {
        c.data[i] = a.data[i] + b.data[i];
    }
    return c;
}
