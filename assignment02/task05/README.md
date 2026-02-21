# Assignment 02 — Task 05 (Tiled GPU Matrix Multiplication)

## Overview
Implements **tiled matrix multiplication** using CUDA shared memory alongside the naive GPU kernel.
The GPU executable supports both `--naive` and `--tile SIZE` modes. A benchmark script
compares CPU, GPU-Naive, and GPU-Tiled at tile sizes 8, 16, 32 across matrix sizes 64–2048.

## Maximum Tile Size Analysis
| Constraint | Limit |
|---|---|
| `sharedMemPerBlock` = 49 152 bytes | 2 × T² × 8 ≤ 49 152 → T ≤ 55 |
| `maxThreadsPerBlock` = 1024 | T² ≤ 1024 → T ≤ 32 |
| **Maximum tile size** | **32** |

## Computational Intensity (Tiled)
For square N×N matrices with tile size T:
- Each tile-pair load brings 2·T² doubles = 16·T² bytes into shared memory.
- Each tile-pair contributes T multiply-adds per output element → 2·T FLOPs × T² elements = 2·T³ FLOPs.
- Intensity = 2·T³ FLOPs / (16·T² bytes) = **T/8 FLOP/byte**.

| Tile | Intensity |
|------|-----------|
| 8 | 1.0 FLOP/byte |
| 16 | 2.0 FLOP/byte |
| 32 | 4.0 FLOP/byte |

Compare with naive: **0.125 FLOP/byte**. Tiling at T=32 gives a **32× improvement** in computational intensity.

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

## Run
```
gpu\matmul_cuda.exe --naive  data\sample_input.txt
gpu\matmul_cuda.exe --tile 32 data\sample_input.txt
```

## Full Benchmark
```
py -3 bench\benchmark.py
```
Produces `bench/results.csv` and `bench/plot.svg`.
