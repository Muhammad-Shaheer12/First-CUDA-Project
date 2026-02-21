#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static int getAttr(cudaDeviceAttr attr, int dev) {
    int val = 0;
    cudaDeviceGetAttribute(&val, attr, dev);
    return val;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);

        int memClockKHz   = getAttr(cudaDevAttrMemoryClockRate, dev);
        int coreClockKHz  = getAttr(cudaDevAttrClockRate, dev);
        int computeMode   = getAttr(cudaDevAttrComputeMode, dev);
        int kernelTimeout = getAttr(cudaDevAttrKernelExecTimeout, dev);
        int deviceOverlap = getAttr(cudaDevAttrGpuOverlap, dev);

        printf("====================================================================\n");
        printf("Device %d: %s\n", dev, p.name);
        printf("====================================================================\n\n");

        printf("--- General Info ---\n");
        printf("  name:                          %s\n", p.name);
        printf("    // ASCII string identifying the device.\n\n");

        printf("  major.minor:                   %d.%d\n", p.major, p.minor);
        printf("    // Compute capability version of the device.\n\n");

        printf("  totalGlobalMem:                %zu bytes (%.2f GB)\n",
               p.totalGlobalMem, (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("    // Total amount of global memory on the device.\n\n");

        printf("  memoryClockRate:               %d kHz (%.2f GHz)\n",
               memClockKHz, memClockKHz / 1.0e6);
        printf("    // Peak memory clock frequency in kHz.\n\n");

        printf("  memoryBusWidth:                %d bits\n", p.memoryBusWidth);
        printf("    // Width of the memory bus (in bits).\n\n");

        printf("  l2CacheSize:                   %d bytes (%.2f MB)\n",
               p.l2CacheSize, p.l2CacheSize / (1024.0 * 1024.0));
        printf("    // Size of L2 cache in bytes.\n\n");

        printf("--- Multiprocessor Info ---\n");
        printf("  multiProcessorCount:           %d\n", p.multiProcessorCount);
        printf("    // Number of streaming multiprocessors (SMs) on the device.\n\n");

        printf("  clockRate:                     %d kHz (%.2f GHz)\n",
               coreClockKHz, coreClockKHz / 1.0e6);
        printf("    // GPU core clock rate in kHz.\n\n");

        printf("  warpSize:                      %d\n", p.warpSize);
        printf("    // Number of threads in a warp.\n\n");

        printf("  maxThreadsPerBlock:            %d\n", p.maxThreadsPerBlock);
        printf("    // Maximum number of threads per block.\n\n");

        printf("  maxThreadsDim[3]:              [%d, %d, %d]\n",
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("    // Max dimensions of a thread block (x, y, z).\n\n");

        printf("  maxGridSize[3]:                [%d, %d, %d]\n",
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf("    // Max dimensions of a grid of blocks (x, y, z).\n\n");

        printf("  maxThreadsPerMultiProcessor:   %d\n", p.maxThreadsPerMultiProcessor);
        printf("    // Max resident threads per SM.\n\n");

        printf("--- Memory Info ---\n");
        printf("  totalConstMem:                 %zu bytes (%.2f KB)\n",
               p.totalConstMem, p.totalConstMem / 1024.0);
        printf("    // Total amount of constant memory on the device.\n\n");

        printf("  sharedMemPerBlock:             %zu bytes (%.2f KB)\n",
               p.sharedMemPerBlock, p.sharedMemPerBlock / 1024.0);
        printf("    // Max shared memory per block.\n\n");

        printf("  sharedMemPerMultiprocessor:     %zu bytes (%.2f KB)\n",
               p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024.0);
        printf("    // Max shared memory per SM.\n\n");

        printf("  regsPerBlock:                  %d\n", p.regsPerBlock);
        printf("    // Total number of 32-bit registers per block.\n\n");

        printf("  regsPerMultiprocessor:         %d\n", p.regsPerMultiprocessor);
        printf("    // Total number of 32-bit registers per SM.\n\n");

        printf("  memPitch:                      %zu bytes\n", p.memPitch);
        printf("    // Max pitch (in bytes) for memory copies involving 2D arrays.\n\n");

        printf("--- Texture/Surface ---\n");
        printf("  maxTexture1D:                  %d\n", p.maxTexture1D);
        printf("    // Max 1D texture size.\n\n");

        printf("  maxTexture2D[2]:               [%d, %d]\n",
               p.maxTexture2D[0], p.maxTexture2D[1]);
        printf("    // Max 2D texture dimensions.\n\n");

        printf("  maxTexture3D[3]:               [%d, %d, %d]\n",
               p.maxTexture3D[0], p.maxTexture3D[1], p.maxTexture3D[2]);
        printf("    // Max 3D texture dimensions.\n\n");

        printf("  maxSurface1D:                  %d\n", p.maxSurface1D);
        printf("    // Max 1D surface size.\n\n");

        printf("  maxSurface2D[2]:               [%d, %d]\n",
               p.maxSurface2D[0], p.maxSurface2D[1]);
        printf("    // Max 2D surface dimensions.\n\n");

        printf("  textureAlignment:              %zu bytes\n", p.textureAlignment);
        printf("    // Alignment requirement for textures.\n\n");

        printf("  texturePitchAlignment:         %zu bytes\n", p.texturePitchAlignment);
        printf("    // Pitch alignment requirement for texture references.\n\n");

        printf("--- Execution Features ---\n");
        printf("  concurrentKernels:             %d\n", p.concurrentKernels);
        printf("    // Whether the device can execute multiple kernels concurrently.\n\n");

        printf("  asyncEngineCount:              %d\n", p.asyncEngineCount);
        printf("    // Number of asynchronous DMA engines (for overlap of copy & compute).\n\n");

        printf("  unifiedAddressing:             %d\n", p.unifiedAddressing);
        printf("    // Whether device & host share a unified address space.\n\n");

        printf("  managedMemory:                 %d\n", p.managedMemory);
        printf("    // Whether device supports CUDA managed memory.\n\n");

        printf("  cooperativeLaunch:             %d\n", p.cooperativeLaunch);
        printf("    // Whether device supports cooperative kernel launches.\n\n");

        printf("  computeMode:                   %d\n", computeMode);
        printf("    // Compute mode: 0=Default, 1=Exclusive, 2=Prohibited, 3=ExclusiveProcess.\n\n");

        printf("  canMapHostMemory:              %d\n", p.canMapHostMemory);
        printf("    // Whether the device can map host memory into CUDA address space.\n\n");

        printf("  isMultiGpuBoard:               %d\n", p.isMultiGpuBoard);
        printf("    // Whether the device is on a multi-GPU board.\n\n");

        printf("  multiGpuBoardGroupID:          %d\n", p.multiGpuBoardGroupID);
        printf("    // Unique ID for a group of devices on the same multi-GPU board.\n\n");

        printf("  ECCEnabled:                    %d\n", p.ECCEnabled);
        printf("    // Whether ECC memory is enabled.\n\n");

        printf("  pciBusID:                      %d\n", p.pciBusID);
        printf("    // PCI bus ID of the device.\n\n");

        printf("  pciDeviceID:                   %d\n", p.pciDeviceID);
        printf("    // PCI device ID.\n\n");

        printf("  pciDomainID:                   %d\n", p.pciDomainID);
        printf("    // PCI domain ID.\n\n");

        printf("  kernelExecTimeoutEnabled:      %d\n", kernelTimeout);
        printf("    // Whether there is a run-time limit on kernels (display GPU watchdog).\n\n");

        printf("  integrated:                    %d\n", p.integrated);
        printf("    // Whether the device is an integrated GPU.\n\n");

        printf("  deviceOverlap:                 %d\n", deviceOverlap);
        printf("    // Whether the device can overlap memory transfers with computation.\n\n");

        printf("--- Derived Performance Metrics ---\n\n");

        // Peak global memory bandwidth (GB/s)
        // BW = 2 * memoryClockRate * (memoryBusWidth / 8)
        // memoryClockRate is in kHz, so multiply by 1e3 to get Hz, divide by 1e9 for GB/s
        double bw_gbs = 2.0 * (memClockKHz * 1e3) * (p.memoryBusWidth / 8.0) / 1e9;
        printf("  Peak Global Memory Bandwidth:  %.2f GB/s\n", bw_gbs);
        printf("    // BW = 2 * memoryClockRate * (memoryBusWidth / 8)\n\n");

        // Peak compute performance (GFLOPS, single precision)
        // Need to know CUDA cores per SM for this compute capability.
        // Common mappings:
        int cuda_cores_per_sm = 0;
        switch (p.major) {
            case 2: cuda_cores_per_sm = (p.minor == 0) ? 32 : 48; break;   // Fermi
            case 3: cuda_cores_per_sm = 192; break;                         // Kepler
            case 5: cuda_cores_per_sm = 128; break;                         // Maxwell
            case 6: cuda_cores_per_sm = (p.minor == 0) ? 64 :              // Pascal GP100
                                        (p.minor == 1) ? 128 :             // Pascal GP10x
                                        (p.minor == 2) ? 128 : 64; break;
            case 7: cuda_cores_per_sm = (p.minor == 0) ? 64 : 64; break;   // Volta / Turing
            case 8: cuda_cores_per_sm = (p.minor == 0) ? 64 :              // Ampere GA100
                                        (p.minor == 6) ? 128 :             // Ampere GA10x
                                        (p.minor == 9) ? 128 : 128; break; // Ada Lovelace
            case 9: cuda_cores_per_sm = 128; break;                         // Hopper / Blackwell
            case 10: cuda_cores_per_sm = 128; break;                        // Blackwell
            case 12: cuda_cores_per_sm = 128; break;                        // Blackwell
            default: cuda_cores_per_sm = 128; break;
        }

        int total_cores = cuda_cores_per_sm * p.multiProcessorCount;
        // GFLOPS = cores * clockRate(kHz) * 1e3 * 2 (FMA) / 1e9
        double gflops_sp = (double)total_cores * (coreClockKHz * 1e3) * 2.0 / 1e9;

        printf("  CUDA cores per SM:             %d  (estimated for CC %d.%d)\n",
               cuda_cores_per_sm, p.major, p.minor);
        printf("  Total CUDA cores:              %d\n", total_cores);
        printf("  Peak FP32 Performance:         %.2f GFLOPS\n", gflops_sp);
        printf("    // GFLOPS = totalCores * clockRate * 2 (FMA)\n\n");
    }

    return 0;
}
