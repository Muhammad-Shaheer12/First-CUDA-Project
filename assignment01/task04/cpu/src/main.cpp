#include "io.hpp"
#include "ops.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <input_file> [output_file]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) {
            print_usage(argv[0]);
            return 2;
        }

        const std::string input_path = argv[1];
        std::ifstream in(input_path);
        if (!in) {
            throw std::runtime_error("Failed to open input file: " + input_path);
        }

        const auto [a, b] = read_two_matrices(in);

        const auto t0 = std::chrono::steady_clock::now();
        const Matrix c = add_elementwise(a, b);
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "TIME_MS " << ms << "\n";

        if (argc == 3) {
            const std::string output_path = argv[2];
            std::ofstream out(output_path);
            if (!out) {
                throw std::runtime_error("Failed to open output file: " + output_path);
            }
            write_matrix_as_c(out, c);
        } else {
            write_matrix_as_c(std::cout, c);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
