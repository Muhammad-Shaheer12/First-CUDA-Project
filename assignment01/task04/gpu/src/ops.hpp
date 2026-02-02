#pragma once

#include "matrix.hpp"

// Returns result matrix; writes elapsed milliseconds (H2D + kernel + D2H) into time_ms.
Matrix add_elementwise_cuda_timed(const Matrix& a, const Matrix& b, double& time_ms);
