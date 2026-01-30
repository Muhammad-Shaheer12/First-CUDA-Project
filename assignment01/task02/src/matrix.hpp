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
        : rows(r), cols(c), data(r * c) {}

    value_type& at(std::size_t r, std::size_t c) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[r * cols + c];
    }

    const value_type& at(std::size_t r, std::size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[r * cols + c];
    }
};
