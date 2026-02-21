#include "io.hpp"
#include "ops.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--naive | --tile SIZE] <input_file> [output_file]\n"
              << "  --naive        use naive (global-memory-only) kernel\n"
              << "  --tile SIZE    use tiled kernel with SIZE×SIZE tiles (default 32)\n"
              << "  If neither flag given, tiled with tile=32 is used.\n";
}

int main(int argc, char** argv) {
    try {
        bool naive = false;
        unsigned tile = 32;
        const char* input_path = nullptr;
        const char* output_path = nullptr;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--naive") == 0) {
                naive = true;
            } else if (std::strcmp(argv[i], "--tile") == 0) {
                if (i + 1 >= argc) { usage(argv[0]); return 2; }
                tile = static_cast<unsigned>(std::stoul(argv[++i]));
            } else if (argv[i][0] == '-') {
                usage(argv[0]);
                return 2;
            } else if (!input_path) {
                input_path = argv[i];
            } else if (!output_path) {
                output_path = argv[i];
            } else {
                usage(argv[0]);
                return 2;
            }
        }

        if (!input_path) { usage(argv[0]); return 2; }

        std::ifstream in(input_path);
        if (!in) throw std::runtime_error(std::string("Cannot open ") + input_path);

        const auto [a, b] = read_two_matrices(in);

        double gpu_ms = 0.0;
        Matrix c;

        if (naive) {
            c = matmul_cuda_naive(a, b, &gpu_ms);
            std::cerr << "GPU naive (H2D+kernel+D2H): " << gpu_ms << " ms\n";
        } else {
            c = matmul_cuda_tiled(a, b, tile, &gpu_ms);
            std::cerr << "GPU tiled (tile=" << tile
                      << ", H2D+kernel+D2H): " << gpu_ms << " ms\n";
        }
        std::cerr << "TIMING_MS " << gpu_ms << "\n";

        if (output_path) {
            std::ofstream out(output_path);
            if (!out) throw std::runtime_error(std::string("Cannot open ") + output_path);
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
