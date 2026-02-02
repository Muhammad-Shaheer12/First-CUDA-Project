#include "io.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace {

bool read_token_skipping_comments(std::istream& in, std::string& token) {
    while (in >> token) {
        if (!token.empty() && token[0] == '#') {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        return true;
    }
    return false;
}

std::size_t read_size(std::istream& in) {
    std::string tok;
    if (!read_token_skipping_comments(in, tok)) {
        throw std::runtime_error("Unexpected end of file while reading size");
    }

    std::size_t value = 0;
    try {
        const unsigned long long parsed = std::stoull(tok);
        value = static_cast<std::size_t>(parsed);
    } catch (...) {
        throw std::runtime_error("Failed to parse size value: '" + tok + "'");
    }

    if (value == 0) {
        throw std::runtime_error("Matrix dimension must be > 0");
    }
    return value;
}

Matrix read_named_matrix(std::istream& in, const std::string& expected_name) {
    std::string name;
    if (!read_token_skipping_comments(in, name)) {
        throw std::runtime_error("Unexpected end of file while reading matrix name");
    }
    if (name != expected_name) {
        throw std::runtime_error("Expected matrix label '" + expected_name + "' but got '" + name + "'");
    }

    const std::size_t rows = read_size(in);
    const std::size_t cols = read_size(in);

    Matrix m(rows, cols);
    const std::size_t count = rows * cols;

    for (std::size_t i = 0; i < count; ++i) {
        std::string tok;
        if (!read_token_skipping_comments(in, tok)) {
            throw std::runtime_error("Unexpected end of file while reading matrix data");
        }
        try {
            m.data[i] = std::stod(tok);
        } catch (...) {
            throw std::runtime_error("Failed to parse numeric value: '" + tok + "'");
        }
    }

    return m;
}

} // namespace

std::pair<Matrix, Matrix> read_two_matrices(std::istream& in) {
    std::string magic;
    if (!read_token_skipping_comments(in, magic)) {
        throw std::runtime_error("Empty input");
    }
    if (magic != "M2X") {
        throw std::runtime_error("Bad header: expected 'M2X'");
    }

    std::string version;
    if (!read_token_skipping_comments(in, version)) {
        throw std::runtime_error("Missing version number");
    }
    if (version != "1") {
        throw std::runtime_error("Unsupported version: " + version);
    }

    Matrix a = read_named_matrix(in, "A");
    Matrix b = read_named_matrix(in, "B");

    return {a, b};
}

void write_matrix_as_c(std::ostream& out, const Matrix& c) {
    out << "M2X 1\n";
    out << "C " << c.rows << " " << c.cols << "\n";

    for (std::size_t r = 0; r < c.rows; ++r) {
        for (std::size_t col = 0; col < c.cols; ++col) {
            out << c.at(r, col);
            if (col + 1 < c.cols) {
                out << ' ';
            }
        }
        out << '\n';
    }
}
