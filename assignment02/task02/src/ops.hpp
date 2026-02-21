#pragma once

#include "matrix.hpp"

/// CPU matrix multiplication: C = A * B (A is m×k, B is k×n → C is m×n).
Matrix matmul_cpu(const Matrix& a, const Matrix& b);
