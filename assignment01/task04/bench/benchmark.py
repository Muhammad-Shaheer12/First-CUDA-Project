import csv
import os
import re
import subprocess
from pathlib import Path


def write_input(path: Path, n: int) -> None:
    # Two n x n matrices in row-major order
    # Deterministic values to keep generation fast.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("M2X 1\n")
        f.write(f"A {n} {n}\n")
        for r in range(n):
            base = r * n
            f.write(" ".join(str(base + c) for c in range(n)))
            f.write("\n")
        f.write(f"B {n} {n}\n")
        for r in range(n):
            base = r * n
            f.write(" ".join(str(2 * (base + c)) for c in range(n)))
            f.write("\n")


TIME_RE = re.compile(r"TIME_MS\s+([0-9]*\.?[0-9]+)")


def run_and_get_time_ms(exe: Path, input_file: Path) -> float:
    # Discard stdout (matrix output) to avoid huge console spam.
    proc = subprocess.run(
        [str(exe), str(input_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{exe.name} failed (rc={proc.returncode}):\n{proc.stderr}")
    m = TIME_RE.search(proc.stderr)
    if not m:
        raise RuntimeError(f"Timing not found in stderr for {exe.name}. stderr was:\n{proc.stderr}")
    return float(m.group(1))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cpu_exe = root / "cpu" / "matrix_add_cpu.exe"
    gpu_exe = root / "gpu" / "matrix_add_cuda.exe"

    if not cpu_exe.exists():
        raise RuntimeError(f"CPU executable not found: {cpu_exe}. Build task04/cpu first.")
    if not gpu_exe.exists():
        raise RuntimeError(f"GPU executable not found: {gpu_exe}. Build task04/gpu first.")

    # Keep sizes reasonable for text-file IO.
    sizes = [64, 128, 256, 512]
    repeats = 3

    rows = []
    tmp_dir = Path(__file__).resolve().parent / "tmp"
    for n in sizes:
        inp = tmp_dir / f"input_{n}.txt"
        write_input(inp, n)

        cpu_runs = [run_and_get_time_ms(cpu_exe, inp) for _ in range(repeats)]
        gpu_runs = [run_and_get_time_ms(gpu_exe, inp) for _ in range(repeats)]

        cpu_ms = min(cpu_runs)
        gpu_ms = min(gpu_runs)

        rows.append({"n": n, "elements": n * n, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms})
        print(f"n={n}: cpu_ms={cpu_ms:.3f}, gpu_ms={gpu_ms:.3f}")

    out_csv = Path(__file__).resolve().parent / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "elements", "cpu_ms", "gpu_ms"])
        w.writeheader()
        w.writerows(rows)

    # Plot (dependency-free SVG)
    xs = [int(r["n"]) for r in rows]
    cpu = [float(r["cpu_ms"]) for r in rows]
    gpu = [float(r["gpu_ms"]) for r in rows]

    width, height = 900, 520
    pad_l, pad_r, pad_t, pad_b = 70, 30, 40, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    x_min, x_max = min(xs), max(xs)
    y_min = 0.0
    y_max = max(max(cpu), max(gpu)) * 1.10
    if y_max <= 0.0:
        y_max = 1.0

    def x_to_px(x: int) -> float:
        if x_max == x_min:
            return float(pad_l + plot_w / 2)
        return pad_l + (float(x - x_min) / float(x_max - x_min)) * plot_w

    def y_to_px(y: float) -> float:
        return pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    def polyline(points, stroke: str) -> str:
        pts = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points)
        return f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}" />'

    def circles(points, color: str) -> str:
        return "\n".join(
            f'<circle cx="{x_to_px(x):.2f}" cy="{y_to_px(y):.2f}" r="4" fill="{color}" />'
            for x, y in points
        )

    cpu_pts = list(zip(xs, cpu))
    gpu_pts = list(zip(xs, gpu))

    # Simple ticks
    y_ticks = 5
    x_ticks = len(xs)

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white" />')
    svg_lines.append(f'<text x="{width/2:.0f}" y="24" font-family="Arial" font-size="18" text-anchor="middle">Matrix add: size vs time</text>')

    # Axes
    x0, y0 = pad_l, pad_t + plot_h
    x1, y1 = pad_l + plot_w, pad_t
    svg_lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" />')
    svg_lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" />')

    # Y grid + labels
    for i in range(y_ticks + 1):
        y = y_min + (y_max - y_min) * (i / y_ticks)
        py = y_to_px(y)
        svg_lines.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#ddd" />')
        svg_lines.append(f'<text x="{x0-10}" y="{py+4:.2f}" font-family="Arial" font-size="12" text-anchor="end">{y:.1f}</text>')

    # X labels
    for i, x in enumerate(xs):
        px = x_to_px(x)
        svg_lines.append(f'<text x="{px:.2f}" y="{y0+20}" font-family="Arial" font-size="12" text-anchor="middle">{x}</text>')

    # Axis titles
    svg_lines.append(f'<text x="{width/2:.0f}" y="{height-20}" font-family="Arial" font-size="14" text-anchor="middle">Matrix size N (NxN)</text>')
    svg_lines.append(f'<text x="20" y="{height/2:.0f}" font-family="Arial" font-size="14" text-anchor="middle" transform="rotate(-90 20 {height/2:.0f})">Time (ms)</text>')

    # Data
    svg_lines.append(polyline(cpu_pts, "#1f77b4"))
    svg_lines.append(polyline(gpu_pts, "#ff7f0e"))
    svg_lines.append(circles(cpu_pts, "#1f77b4"))
    svg_lines.append(circles(gpu_pts, "#ff7f0e"))

    # Legend
    lx, ly = x1 - 240, y1 + 10
    svg_lines.append(f'<rect x="{lx}" y="{ly}" width="230" height="48" fill="white" stroke="#ccc" />')
    svg_lines.append(f'<line x1="{lx+10}" y1="{ly+16}" x2="{lx+40}" y2="{ly+16}" stroke="#1f77b4" stroke-width="2" />')
    svg_lines.append(f'<text x="{lx+50}" y="{ly+20}" font-family="Arial" font-size="12">CPU (compute)</text>')
    svg_lines.append(f'<line x1="{lx+10}" y1="{ly+34}" x2="{lx+40}" y2="{ly+34}" stroke="#ff7f0e" stroke-width="2" />')
    svg_lines.append(f'<text x="{lx+50}" y="{ly+38}" font-family="Arial" font-size="12">GPU (H2D+kernel+D2H)</text>')

    svg_lines.append('</svg>')

    out_svg = Path(__file__).resolve().parent / "plot.svg"
    out_svg.write_text("\n".join(svg_lines), encoding="utf-8")

    # Optional PNG via matplotlib (if installed)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(xs, cpu, marker="o", label="CPU (compute)")
        plt.plot(xs, gpu, marker="o", label="GPU (H2D+kernel+D2H)")
        plt.xlabel("Matrix size N (NxN)")
        plt.ylabel("Time (ms)")
        plt.title("Matrix add: size vs time")
        plt.grid(True)
        plt.legend()

        out_png = Path(__file__).resolve().parent / "plot.png"
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        print(f"Wrote {out_csv}, {out_svg}, and {out_png}")
    except ImportError:
        print(f"Wrote {out_csv} and {out_svg}")


if __name__ == "__main__":
    main()
