#pragma once

#include "matrix.hpp"

/// Naive CUDA matrix multiply.
Matrix matmul_cuda_naive(const Matrix& a, const Matrix& b, double* elapsed_ms);

/// Tiled CUDA matrix multiply using shared memory.
/// tile_size must be a power of 2 (8, 16, 32).
Matrix matmul_cuda_tiled(const Matrix& a, const Matrix& b, unsigned tile_size,
                         double* elapsed_ms);
