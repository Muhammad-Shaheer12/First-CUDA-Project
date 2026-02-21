#pragma once

#include "matrix.hpp"

/// Naive CUDA matrix multiplication: C = A * B.
/// Returns elapsed GPU time (H2D + kernel + D2H) in milliseconds via *elapsed_ms.
Matrix matmul_cuda(const Matrix& a, const Matrix& b, double* elapsed_ms);
