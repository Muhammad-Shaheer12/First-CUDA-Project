#include "ops.hpp"
#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

/* ── Naive kernel ────────────────────────────────────────────── */

namespace {

__global__ void matmul_naive_kernel(const double* __restrict__ A,
                                    const double* __restrict__ B,
                                    double* __restrict__ C,
                                    unsigned int M, unsigned int K,
                                    unsigned int N) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (unsigned int p = 0; p < K; ++p)
            sum += A[row * K + p] * B[p * N + col];
        C[row * N + col] = sum;
    }
}

/* ── Tiled kernel (dynamic shared memory) ────────────────────── */

__global__ void matmul_tiled_kernel(const double* __restrict__ A,
                                    const double* __restrict__ B,
                                    double* __restrict__ C,
                                    unsigned int M, unsigned int K,
                                    unsigned int N, unsigned int TILE) {
    extern __shared__ double smem[];
    double* As = smem;
    double* Bs = smem + TILE * TILE;

    const unsigned int row = blockIdx.y * TILE + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;

    const unsigned int numTiles = (K + TILE - 1) / TILE;

    for (unsigned int t = 0; t < numTiles; ++t) {
        unsigned int aCol = t * TILE + threadIdx.x;
        unsigned int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y * TILE + threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0;

        Bs[threadIdx.y * TILE + threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

        for (unsigned int p = 0; p < TILE; ++p)
            sum += As[threadIdx.y * TILE + p] * Bs[p * TILE + threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

/* ── Helper: allocate, time, free ────────────────────────────── */

struct GpuBuffers {
    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;
    cudaEvent_t ev_start{};
    cudaEvent_t ev_stop{};

    void alloc(std::size_t bytes_a, std::size_t bytes_b, std::size_t bytes_c) {
        cuda_check(cudaFree(nullptr), "warmup");
        cuda_check(cudaEventCreate(&ev_start), "event create");
        cuda_check(cudaEventCreate(&ev_stop), "event create");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes_a), "malloc A");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes_b), "malloc B");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes_c), "malloc C");
    }

    double finish_and_free(Matrix& c, std::size_t bytes_c) {
        cuda_check(cudaEventRecord(ev_stop), "event record stop");
        cuda_check(cudaEventSynchronize(ev_stop), "event sync");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, ev_start, ev_stop), "elapsed");
        cuda_check(cudaMemcpy(c.data.data(), d_c, bytes_c, cudaMemcpyDeviceToHost), "D2H C");
        cuda_check(cudaEventDestroy(ev_start), "ev destroy");
        cuda_check(cudaEventDestroy(ev_stop), "ev destroy");
        cuda_check(cudaFree(d_a), "free A");
        cuda_check(cudaFree(d_b), "free B");
        cuda_check(cudaFree(d_c), "free C");
        d_a = d_b = d_c = nullptr;
        return static_cast<double>(ms);
    }

    void cleanup() {
        if (d_a) (void)cudaFree(d_a);
        if (d_b) (void)cudaFree(d_b);
        if (d_c) (void)cudaFree(d_c);
        (void)cudaEventDestroy(ev_start);
        (void)cudaEventDestroy(ev_stop);
    }
};

} // namespace

/* ── Public API ──────────────────────────────────────────────── */

Matrix matmul_cuda_naive(const Matrix& a, const Matrix& b, double* elapsed_ms) {
    if (a.cols != b.rows)
        throw std::invalid_argument("matmul: A.cols != B.rows");

    const unsigned int M = static_cast<unsigned int>(a.rows);
    const unsigned int K = static_cast<unsigned int>(a.cols);
    const unsigned int N = static_cast<unsigned int>(b.cols);
    Matrix c(M, N);

    const std::size_t ba = a.data.size() * sizeof(double);
    const std::size_t bb = b.data.size() * sizeof(double);
    const std::size_t bc = c.data.size() * sizeof(double);

    GpuBuffers g;
    try {
        g.alloc(ba, bb, bc);

        cuda_check(cudaEventRecord(g.ev_start), "event record start");
        cuda_check(cudaMemcpy(g.d_a, a.data.data(), ba, cudaMemcpyHostToDevice), "H2D A");
        cuda_check(cudaMemcpy(g.d_b, b.data.data(), bb, cudaMemcpyHostToDevice), "H2D B");

        constexpr unsigned BLK = 16;
        dim3 block(BLK, BLK);
        dim3 grid((N + BLK - 1) / BLK, (M + BLK - 1) / BLK);
        matmul_naive_kernel<<<grid, block>>>(g.d_a, g.d_b, g.d_c, M, K, N);

        double ms = g.finish_and_free(c, bc);
        if (elapsed_ms) *elapsed_ms = ms;
        return c;
    } catch (...) { g.cleanup(); throw; }
}

Matrix matmul_cuda_tiled(const Matrix& a, const Matrix& b, unsigned tile_size,
                         double* elapsed_ms) {
    if (a.cols != b.rows)
        throw std::invalid_argument("matmul: A.cols != B.rows");

    const unsigned int M = static_cast<unsigned int>(a.rows);
    const unsigned int K = static_cast<unsigned int>(a.cols);
    const unsigned int N = static_cast<unsigned int>(b.cols);
    Matrix c(M, N);

    const std::size_t ba = a.data.size() * sizeof(double);
    const std::size_t bb = b.data.size() * sizeof(double);
    const std::size_t bc = c.data.size() * sizeof(double);

    GpuBuffers g;
    try {
        g.alloc(ba, bb, bc);

        cuda_check(cudaEventRecord(g.ev_start), "event record start");
        cuda_check(cudaMemcpy(g.d_a, a.data.data(), ba, cudaMemcpyHostToDevice), "H2D A");
        cuda_check(cudaMemcpy(g.d_b, b.data.data(), bb, cudaMemcpyHostToDevice), "H2D B");

        dim3 block(tile_size, tile_size);
        dim3 grid((N + tile_size - 1) / tile_size,
                  (M + tile_size - 1) / tile_size);
        std::size_t smem = 2ULL * tile_size * tile_size * sizeof(double);

        matmul_tiled_kernel<<<grid, block, smem>>>(g.d_a, g.d_b, g.d_c,
                                                    M, K, N, tile_size);

        double ms = g.finish_and_free(c, bc);
        if (elapsed_ms) *elapsed_ms = ms;
        return c;
    } catch (...) { g.cleanup(); throw; }
}
