#include "ops.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

namespace {

__global__ void matmul_kernel(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C,
                              unsigned int M, unsigned int K, unsigned int N) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (unsigned int p = 0; p < K; ++p) {
            sum += A[row * K + p] * B[p * N + col];
        }
        C[row * N + col] = sum;
    }
}

} // namespace

Matrix matmul_cuda(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows)
        throw std::invalid_argument("matmul: A.cols != B.rows");

    const unsigned int M = static_cast<unsigned int>(a.rows);
    const unsigned int K = static_cast<unsigned int>(a.cols);
    const unsigned int N = static_cast<unsigned int>(b.cols);

    Matrix c(M, N);

    const std::size_t bytes_a = a.data.size() * sizeof(double);
    const std::size_t bytes_b = b.data.size() * sizeof(double);
    const std::size_t bytes_c = c.data.size() * sizeof(double);

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;

    try {
        cuda_check(cudaFree(nullptr), "warmup");

        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes_a), "malloc A");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes_b), "malloc B");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes_c), "malloc C");

        cuda_check(cudaMemcpy(d_a, a.data.data(), bytes_a, cudaMemcpyHostToDevice), "H2D A");
        cuda_check(cudaMemcpy(d_b, b.data.data(), bytes_b, cudaMemcpyHostToDevice), "H2D B");

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, M, K, N);
        cuda_device_synchronize_checked();

        cuda_check(cudaMemcpy(c.data.data(), d_c, bytes_c, cudaMemcpyDeviceToHost), "D2H C");

        cuda_check(cudaFree(d_a), "free A");
        cuda_check(cudaFree(d_b), "free B");
        cuda_check(cudaFree(d_c), "free C");

        return c;
    } catch (...) {
        if (d_a) (void)cudaFree(d_a);
        if (d_b) (void)cudaFree(d_b);
        if (d_c) (void)cudaFree(d_c);
        throw;
    }
}
