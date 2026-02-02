# Assignment 01 — Task 01: GPU Coding Environment Setup (CUDA)

## Goal
Set up a working CUDA development environment that provides:
- Access to a machine with an NVIDIA GPU
- CUDA toolkit installed (compiler + headers + libraries)
- Ability to compile and run a CUDA program on the GPU
- A GitHub repo for the project, shared with the instructor

This document describes the environment I set up and the exact commands I used to verify it works.

---

## Environment summary (my setup)
- **OS**: Windows
- **GPU**: NVIDIA GPU (local machine)
- **CUDA Toolkit**: CUDA Toolkit 13.1 (`nvcc` present)
- **Compiler toolchain**: Visual Studio 2022 Build Tools (MSVC) for Windows host compilation
- **Build system**: `mingw32-make` for Makefile builds (PowerShell)

---

## 1) Install prerequisites

### 1.1 NVIDIA driver
Install the latest NVIDIA driver for your GPU.
- Verification: open `nvidia-smi` in a terminal and confirm the GPU is detected.

### 1.2 Install CUDA Toolkit
Install the CUDA Toolkit (includes `nvcc`, headers, runtime libraries).
- Download from NVIDIA CUDA Toolkit page.
- Ensure `nvcc` is available on PATH.

**Verify** (PowerShell):
- `nvcc --version`

### 1.3 Install Visual Studio Build Tools (MSVC)
On Windows, `nvcc` uses MSVC (`cl.exe`) as the host compiler.
Install:
- Visual Studio 2022 Build Tools
- Workload: **Desktop development with C++**

**Verify** (in “x64 Native Tools Command Prompt for VS 2022”):
- `cl /Bv`

---

## 2) Make sure you can compile and run a CUDA program

### 2.1 Minimal CUDA test program
Create `hello.cu`:

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel() {}

int main() {
    kernel<<<1, 1>>>();
    auto st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(st) << "\n";
        return 1;
    }
    std::cout << "CUDA kernel ran successfully\n";
    return 0;
}
```

### 2.2 Build
In a VS Developer prompt (so `cl.exe` is on PATH):
- `nvcc -std=c++17 hello.cu -o hello.exe`

### 2.3 Run
- `./hello.exe`

Expected output:
- `CUDA kernel ran successfully`

---

## 3) Debugging / profiling (what to install)
Depending on your CUDA Toolkit version, install the NVIDIA tools you need:
- **Nsight Systems** (timeline profiling)
- **Nsight Compute** (kernel profiling)
- **Nsight Visual Studio Edition** (VS integration, if needed)

At minimum for this assignment, being able to compile (`nvcc`) and run CUDA code correctly is sufficient.

---

## 4) Build system approach used in this repo
This repo uses Makefiles plus a small Windows helper script for CUDA builds:

- For CUDA projects (Task 03 / Task 04), `nvcc` requires MSVC environment variables.
- The repo provides a `build_win64.bat` helper that:
  - Finds Visual Studio Build Tools using `vswhere`
  - Calls `vcvars64.bat`
  - Runs `mingw32-make`

This makes building CUDA projects possible from PowerShell by just running the batch file.

---

## 5) GitHub repo
- Repo: https://github.com/Muhammad-Shaheer12/First-CUDA-Project

This repo contains:
- Task 02 (CPU)
- Task 03 (CUDA)
- Task 04 (Timing + Benchmark)
- This Task 01 setup document

---

## Quick verification checklist
- `nvcc --version` works
- Visual Studio Build Tools installed (MSVC available after running `vcvars64.bat`)
- A CUDA kernel compiles and runs successfully on the GPU
- Code pushed to GitHub repo and shared with instructor
