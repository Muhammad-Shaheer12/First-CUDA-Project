#pragma once

#include "matrix.hpp"

#include <istream>
#include <ostream>
#include <utility>

std::pair<Matrix, Matrix> read_two_matrices(std::istream& in);
void write_matrix_as_c(std::ostream& out, const Matrix& c);
