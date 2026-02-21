# Assignment 02 — Task 03 (CUDA Naive Matrix Multiplication)

## What it does
Reads two matrices A (m×k) and B (k×n), copies them to GPU, multiplies them naively (`C = A × B`), copies the result back, and writes C.

## CLI
```
matmul_cuda <input_file> [output_file]
```

## Build (Windows)
```
.\build_win64.bat
```

## Run
```
.\matmul_cuda.exe data\sample_input.txt
```

## Computational Intensity (Naive)
For square N×N matrices:
- **FLOPs**: Each element of C requires N multiply-add operations → 2·N FLOPs per element → 2·N³ total FLOPs.
- **Memory accesses**: Each thread reads one full row of A (N doubles) and one full column of B (N doubles) from global memory, and writes 1 element of C.
  Total global memory traffic ≈ N² × (N + N + 1) × 8 bytes = N²·(2N+1)·8 bytes.
- **Computational intensity** = FLOPs / Bytes ≈ 2N³ / (N²·2N·8) = **1/8 FLOP/byte ≈ 0.125 FLOP/byte**.

This is very low — the kernel is **memory-bound** and does not reuse data from global memory.
Tiling (Task 05) improves this by loading data into shared memory and reusing it across multiple output elements.
