# Assignment 01 — Task 02 (CPU / C++)

## What it does
Reads **two matrices** from an input file, adds them **element-wise on the CPU**, and writes the result in the **same file format**.

## CLI
- `./matrix_add <input_file> [output_file]`
  - If `output_file` is omitted, output is written to **stdout**.

## File format (text)
A simple token-based format (whitespace separated). Lines starting with `#` are treated as comments.

**Input format**
```
M2X 1
A <rows> <cols>
<rows*cols values in row-major order>
B <rows> <cols>
<rows*cols values in row-major order>
```

Values can be split across lines arbitrarily.

**Output format**
```
M2X 1
C <rows> <cols>
<rows*cols values in row-major order>
```

## Example
See `data/sample_input.txt`.

## Build
From `assignment01/task02`:
- PowerShell / Windows: `mingw32-make`
- MSYS2 shell: `make`

## Run
- PowerShell / Windows:
  - `./matrix_add.exe data/sample_input.txt`
  - `./matrix_add.exe data/sample_input.txt out.txt`
- MSYS2 shell:
  - `./matrix_add.exe data/sample_input.txt`
  - `./matrix_add.exe data/sample_input.txt out.txt`
