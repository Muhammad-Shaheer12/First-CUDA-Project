#pragma once

#include "matrix.hpp"

#include <istream>
#include <ostream>
#include <utility>

/// Reads two matrices A and B from the M2X format stream.
std::pair<Matrix, Matrix> read_two_matrices(std::istream& in);

/// Writes a single matrix labelled C in M2X format.
void write_matrix_as_c(std::ostream& out, const Matrix& c);
