#include "ops.hpp"

#include "cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

namespace {

__global__ void add_kernel(const double* a, const double* b, double* c, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

} // namespace

Matrix add_elementwise_cuda(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix c(a.rows, a.cols);
    const std::size_t n = c.data.size();
    const std::size_t bytes = n * sizeof(Matrix::value_type);

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;

    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes), "cudaMalloc d_a");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes), "cudaMalloc d_b");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes), "cudaMalloc d_c");

        cuda_check(cudaMemcpy(d_a, a.data.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
        cuda_check(cudaMemcpy(d_b, b.data.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");

        const std::size_t threads = static_cast<std::size_t>(BLOCK_SIZE);
        const std::size_t blocks = (n + threads - 1) / threads;

        add_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads)>>>(d_a, d_b, d_c, n);
        cuda_device_synchronize_checked();

        cuda_check(cudaMemcpy(c.data.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");

        cuda_check(cudaFree(d_a), "cudaFree d_a");
        cuda_check(cudaFree(d_b), "cudaFree d_b");
        cuda_check(cudaFree(d_c), "cudaFree d_c");

        return c;
    } catch (...) {
        if (d_a != nullptr) {
            (void)cudaFree(d_a);
        }
        if (d_b != nullptr) {
            (void)cudaFree(d_b);
        }
        if (d_c != nullptr) {
            (void)cudaFree(d_c);
        }
        throw;
    }
}
