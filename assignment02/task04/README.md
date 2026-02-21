# Assignment 02 — Task 04 (Timing & Benchmarking)

## Overview
Adds wall-clock timing to the CPU matrix-multiply (via `std::chrono`) and CUDA event timing to the GPU multiply (H2D + kernel + D2H).  A Python benchmark script sweeps N = 64 … 2048 and produces `bench/results.csv` + `bench/plot.svg`.

## Build

### CPU
```
cd cpu
mingw32-make
```

### GPU
```
cd gpu
.\build_win64.bat
```

## Run individually
```
cpu\matmul_cpu.exe  data\sample_input.txt
gpu\matmul_cuda.exe data\sample_input.txt
```
Timing printed to stderr.

## Full benchmark
```
py -3 bench\benchmark.py
```
Generates `bench/results.csv` and `bench/plot.svg`.
