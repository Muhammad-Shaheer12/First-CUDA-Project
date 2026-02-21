#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

struct Matrix {
    using value_type = double;

    std::size_t rows{};
    std::size_t cols{};
    std::vector<value_type> data{};

    Matrix() = default;

    Matrix(std::size_t r, std::size_t c)
        : rows(r), cols(c), data(r * c, 0.0) {}

    value_type& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    const value_type& operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
};
