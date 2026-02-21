#include "io.hpp"
#include "ops.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
            return 2;
        }

        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error(std::string("Cannot open ") + argv[1]);

        const auto [a, b] = read_two_matrices(in);

        auto t0 = std::chrono::high_resolution_clock::now();
        const Matrix c = matmul_cpu(a, b);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "CPU matmul: " << elapsed_ms << " ms\n";
        std::cerr << "TIMING_MS " << elapsed_ms << "\n";

        if (argc == 3) {
            std::ofstream out(argv[2]);
            if (!out) throw std::runtime_error(std::string("Cannot open ") + argv[2]);
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
