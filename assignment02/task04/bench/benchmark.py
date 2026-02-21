#!/usr/bin/env python3
"""Benchmark CPU vs GPU (naive) matrix multiply across matrix sizes."""

import csv
import os
import random
import re
import subprocess
import sys

SIZES = [64, 128, 256, 512, 1024, 2048]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(SCRIPT_DIR)
CPU_EXE = os.path.join(TASK_DIR, "cpu", "matmul_cpu.exe")
GPU_EXE = os.path.join(TASK_DIR, "gpu", "matmul_cuda.exe")
TMP_DIR = os.path.join(SCRIPT_DIR, "tmp")
RESULTS = os.path.join(SCRIPT_DIR, "results.csv")
PLOT    = os.path.join(SCRIPT_DIR, "plot.svg")

TIMING_RE = re.compile(r"TIMING_MS\s+([\d.eE+\-]+)")


def gen_input(n: int, path: str) -> None:
    """Write an M2X file with two random n×n matrices."""
    with open(path, "w") as f:
        f.write("M2X 1\n")
        for label in ("A", "B"):
            f.write(f"{label} {n} {n}\n")
            for _ in range(n):
                row = " ".join(f"{random.uniform(-10, 10):.6f}" for _ in range(n))
                f.write(row + "\n")


def run_and_time(exe: str, inp: str) -> float:
    """Run exe with inp, parse TIMING_MS from stderr, return ms."""
    r = subprocess.run([exe, inp], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  ERROR running {exe}:\n{r.stderr}", file=sys.stderr)
        return float("nan")
    m = TIMING_RE.search(r.stderr)
    if not m:
        print(f"  WARNING: no TIMING_MS in stderr of {exe}", file=sys.stderr)
        return float("nan")
    return float(m.group(1))


def main() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)

    for exe, name in ((CPU_EXE, "CPU"), (GPU_EXE, "GPU")):
        if not os.path.isfile(exe):
            print(f"ERROR: {name} executable not found: {exe}", file=sys.stderr)
            sys.exit(1)

    rows: list[dict[str, object]] = []

    for n in SIZES:
        inp = os.path.join(TMP_DIR, f"input_{n}.txt")
        print(f"N={n}: generating input … ", end="", flush=True)
        gen_input(n, inp)

        print("CPU … ", end="", flush=True)
        cpu_ms = run_and_time(CPU_EXE, inp)

        print("GPU … ", end="", flush=True)
        gpu_ms = run_and_time(GPU_EXE, inp)

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("nan")
        print(f"CPU={cpu_ms:.2f} ms  GPU={gpu_ms:.2f} ms  Speedup={speedup:.2f}x")
        rows.append({"N": n, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": speedup})

    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "cpu_ms", "gpu_ms", "speedup"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {RESULTS}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns     = [r["N"] for r in rows]
        cpu_t  = [r["cpu_ms"] for r in rows]
        gpu_t  = [r["gpu_ms"] for r in rows]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ns, cpu_t, "o-", label="CPU")
        ax1.plot(ns, gpu_t, "s-", label="GPU (naive)")
        ax1.set_xlabel("Matrix size N (N×N)")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Matrix Multiplication: CPU vs GPU (Naive)")
        ax1.legend(loc="upper left")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(PLOT, dpi=150)
        print(f"Plot saved to {PLOT}")
    except ImportError:
        print("matplotlib not available — skipping plot generation.", file=sys.stderr)


if __name__ == "__main__":
    main()
