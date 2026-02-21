#include "io.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace {

bool read_token(std::istream& in, std::string& token) {
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
    if (!read_token(in, tok))
        throw std::runtime_error("Unexpected EOF reading size");
    auto v = std::stoull(tok);
    if (v == 0) throw std::runtime_error("Dimension must be > 0");
    return static_cast<std::size_t>(v);
}

Matrix read_named_matrix(std::istream& in, const std::string& label) {
    std::string name;
    if (!read_token(in, name))
        throw std::runtime_error("Unexpected EOF reading matrix label");
    if (name != label)
        throw std::runtime_error("Expected '" + label + "', got '" + name + "'");

    const std::size_t rows = read_size(in);
    const std::size_t cols = read_size(in);

    Matrix m(rows, cols);
    for (std::size_t i = 0; i < rows * cols; ++i) {
        std::string tok;
        if (!read_token(in, tok))
            throw std::runtime_error("Unexpected EOF reading matrix data");
        m.data[i] = std::stod(tok);
    }
    return m;
}

} // namespace

std::pair<Matrix, Matrix> read_two_matrices(std::istream& in) {
    std::string magic;
    if (!read_token(in, magic)) throw std::runtime_error("Empty input");
    if (magic != "M2X") throw std::runtime_error("Bad header: expected 'M2X'");

    std::string ver;
    if (!read_token(in, ver)) throw std::runtime_error("Missing version");
    if (ver != "1") throw std::runtime_error("Unsupported version: " + ver);

    Matrix a = read_named_matrix(in, "A");
    Matrix b = read_named_matrix(in, "B");
    return {a, b};
}

void write_matrix_as_c(std::ostream& out, const Matrix& c) {
    out << "M2X 1\n";
    out << "C " << c.rows << " " << c.cols << "\n";
    for (std::size_t r = 0; r < c.rows; ++r) {
        for (std::size_t col = 0; col < c.cols; ++col) {
            out << c(r, col);
            if (col + 1 < c.cols) out << ' ';
        }
        out << '\n';
    }
}
