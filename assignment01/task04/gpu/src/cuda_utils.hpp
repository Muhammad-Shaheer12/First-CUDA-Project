#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void cuda_device_synchronize_checked();
