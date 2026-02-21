# Assignment 02 — Task 02 (CPU Matrix Multiplication)

## What it does
Reads **two matrices** (potentially non-square) from an input file and **multiplies** them on the CPU:
`C = A × B`  (A is m×k, B is k×n → C is m×n).

## CLI
```
matmul_cpu <input_file> [output_file]
```
If `output_file` is omitted → stdout.

## File format (M2X v1, text)
```
M2X 1
A <rows> <cols>
<values row-major, whitespace separated>
B <rows> <cols>
<values row-major, whitespace separated>
```
Output uses label `C` instead of `A`/`B`.

## Build
```
mingw32-make
```

## Run
```
.\matmul_cpu.exe data\sample_input.txt
```
