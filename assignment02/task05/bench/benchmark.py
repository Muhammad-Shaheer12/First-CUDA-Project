#!/usr/bin/env python3
"""Benchmark CPU vs GPU-Naive vs GPU-Tiled (multiple tile sizes) matrix multiply."""

import csv
import os
import random
import re
import subprocess
import sys

SIZES = [64, 128, 256, 512, 1024, 2048]
TILE_SIZES = [8, 16, 32]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(SCRIPT_DIR)
CPU_EXE = os.path.join(TASK_DIR, "cpu", "matmul_cpu.exe")
GPU_EXE = os.path.join(TASK_DIR, "gpu", "matmul_cuda.exe")
TMP_DIR = os.path.join(SCRIPT_DIR, "tmp")
RESULTS = os.path.join(SCRIPT_DIR, "results.csv")
PLOT    = os.path.join(SCRIPT_DIR, "plot.svg")

TIMING_RE = re.compile(r"TIMING_MS\s+([\d.eE+\-]+)")


def gen_input(n: int, path: str) -> None:
    with open(path, "w") as f:
        f.write("M2X 1\n")
        for label in ("A", "B"):
            f.write(f"{label} {n} {n}\n")
            for _ in range(n):
                row = " ".join(f"{random.uniform(-10, 10):.6f}" for _ in range(n))
                f.write(row + "\n")


def run_and_time(args: list[str]) -> float:
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  ERROR running {args}:\n{r.stderr}", file=sys.stderr)
        return float("nan")
    m = TIMING_RE.search(r.stderr)
    if not m:
        print(f"  WARNING: no TIMING_MS in stderr of {args}", file=sys.stderr)
        return float("nan")
    return float(m.group(1))


def main() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)

    for exe, name in ((CPU_EXE, "CPU"), (GPU_EXE, "GPU")):
        if not os.path.isfile(exe):
            print(f"ERROR: {name} executable not found: {exe}", file=sys.stderr)
            sys.exit(1)

    fieldnames = ["N", "cpu_ms", "gpu_naive_ms"] + [f"gpu_tile{t}_ms" for t in TILE_SIZES]
    rows: list[dict[str, object]] = []

    for n in SIZES:
        inp = os.path.join(TMP_DIR, f"input_{n}.txt")
        print(f"N={n}: gen ", end="", flush=True)
        gen_input(n, inp)

        print("CPU ", end="", flush=True)
        cpu_ms = run_and_time([CPU_EXE, inp])

        print("naive ", end="", flush=True)
        naive_ms = run_and_time([GPU_EXE, "--naive", inp])

        row: dict[str, object] = {"N": n, "cpu_ms": cpu_ms, "gpu_naive_ms": naive_ms}

        for ts in TILE_SIZES:
            print(f"tile{ts} ", end="", flush=True)
            t_ms = run_and_time([GPU_EXE, "--tile", str(ts), inp])
            row[f"gpu_tile{ts}_ms"] = t_ms

        best_tile = min(TILE_SIZES, key=lambda t: float(row[f"gpu_tile{t}_ms"]))  # type: ignore[arg-type]
        best_ms = float(row[f"gpu_tile{best_tile}_ms"])  # type: ignore[arg-type]
        speedup_naive = cpu_ms / naive_ms if naive_ms > 0 else float("nan")
        speedup_tiled = cpu_ms / best_ms if best_ms > 0 else float("nan")

        print(f"| CPU={cpu_ms:.2f} naive={naive_ms:.2f} best_tile({best_tile})={best_ms:.2f} "
              f"speedup_naive={speedup_naive:.1f}x tiled={speedup_tiled:.1f}x")

        rows.append(row)

    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {RESULTS}")

    # ── Max tile analysis ──
    # sharedMemPerBlock = 49152 bytes (48 KB)
    # Two tiles of TILE×TILE doubles: 2 * TILE^2 * 8 ≤ 49152
    # TILE^2 ≤ 3072 → TILE ≤ 55
    # maxThreadsPerBlock = 1024 → TILE ≤ 32  (32×32 = 1024)
    # ⇒ Maximum tile size = 32
    print("\n── Max tile size analysis ──")
    print("sharedMemPerBlock  = 49152 bytes")
    print("sizeof(double)     = 8")
    print("Two tiles: 2 × T² × 8  ≤ 49152 → T ≤ 55")
    print("maxThreadsPerBlock = 1024 → T ≤ 32  (32×32 = 1024)")
    print("⇒ Maximum tile size = 32\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [r["N"] for r in rows]
        cpu_t   = [r["cpu_ms"] for r in rows]
        naive_t = [r["gpu_naive_ms"] for r in rows]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ns, cpu_t, "o-", label="CPU")
        ax.plot(ns, naive_t, "s-", label="GPU Naive")

        for ts in TILE_SIZES:
            key = f"gpu_tile{ts}_ms"
            vals = [r[key] for r in rows]
            ax.plot(ns, vals, "^--", label=f"GPU Tiled ({ts}×{ts})")

        ax.set_xlabel("Matrix size N (N×N)")
        ax.set_ylabel("Time (ms)")
        ax.set_title("MatMul: CPU vs GPU-Naive vs GPU-Tiled")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(PLOT, dpi=150)
        print(f"Plot saved to {PLOT}")
    except ImportError:
        print("matplotlib not available — skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
