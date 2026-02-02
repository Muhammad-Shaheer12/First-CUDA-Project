# Assignment 01 — Task 03 (GPU / CUDA)

## What it does
Reads **two matrices** from an input file, adds them **element-wise on the GPU (CUDA)**, and writes the result in the **same file format**.

## CLI
- `matrix_add_cuda <input_file> [output_file]`
  - If `output_file` is omitted, output is written to **stdout**.

## File format (text)
Token-based format (whitespace separated). Lines starting with `#` are comments.

**Input format**
```
M2X 1
A <rows> <cols>
<rows*cols values in row-major order>
B <rows> <cols>
<rows*cols values in row-major order>
```

**Output format**
```
M2X 1
C <rows> <cols>
<rows*cols values in row-major order>
```

## Build (PowerShell / Windows)
From `assignment01/task03`:
- Preferred (sets up MSVC for nvcc automatically):
  - `./build_win64.bat`
- If you already opened an "x64 Native Tools Command Prompt for VS 2022":
  - `mingw32-make`

Notes:
- Requires CUDA Toolkit (nvcc) in PATH.
- You may need Visual Studio Build Tools installed (nvcc uses MSVC on Windows).

## Run
- `./matrix_add_cuda.exe data/sample_input.txt`
- `./matrix_add_cuda.exe data/sample_input.txt out.txt`

## Tuning (compile-time)
You can set CUDA block size at build time:
- `mingw32-make clean`
- `mingw32-make BLOCK=16`

(Default is `BLOCK=16`.)
