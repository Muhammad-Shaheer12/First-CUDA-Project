#pragma once

#include "matrix.hpp"

/// Naive CUDA matrix multiplication: C = A * B.
Matrix matmul_cuda(const Matrix& a, const Matrix& b);
