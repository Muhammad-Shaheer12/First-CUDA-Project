#include "cuda_utils.hpp"

void cuda_device_synchronize_checked() {
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cuda_check(cudaGetLastError(), "CUDA kernel launch");
}
