# Assignment 01 — Task 04 (Timing + Benchmark)

This task adds **computation timing** to both programs from Task 02 (CPU) and Task 03 (GPU/CUDA), then benchmarks multiple matrix sizes and produces a graph.

## Timing rules
- **CPU time** measures only the element-wise addition computation.
- **GPU time** measures **H2D copy + kernel + D2H copy** (includes transfer + computation).

Timing is printed to **stderr** as:
```
TIME_MS <milliseconds>
```
So the matrix output format on stdout / output file remains unchanged.

## File format
Same as Task 02/03 (token-based, comments start with `#`).

## Build
### CPU
From `assignment01/task04/cpu`:
- `mingw32-make`

### GPU (Windows)
From `assignment01/task04/gpu`:
- Preferred: `./build_win64.bat`
- Or from a VS "x64 Native Tools Command Prompt": `mingw32-make`

## Benchmark + Plot
From `assignment01/task04/bench`:
- `python benchmark.py`

Outputs:
- `results.csv`
- `plot.svg` (always generated)
- `plot.png` (only if matplotlib is installed)

## What I observed (on my machine)
Using `sizes = [64, 128, 256, 512]` (NxN), the measured times were:
- CPU: ~0.008 ms → ~0.331 ms
- GPU (H2D + kernel + D2H): ~0.336 ms → ~0.693 ms

For these sizes the CPU is faster. This is expected because:
- GPU has non-trivial **transfer + launch overhead** that dominates at small/medium sizes.
- The CPU benefits from caches and very low per-element overhead for small matrices.

If you benchmark much larger matrices (where compute dominates transfers), the GPU may catch up or win, but text-file input generation becomes a bottleneck for very large sizes.

## Notes
- CUDA build requires `nvcc` and Visual Studio Build Tools (MSVC) installed.
