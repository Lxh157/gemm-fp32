#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt

# 运行脚本：python3 scripts/plot.py

RAW_DIR = Path("results/raw")
candidates = sorted(RAW_DIR.glob("bench_fp32_*.txt"))
if not candidates:
    raise FileNotFoundError("No results/raw/bench_fp32_*.txt found. Run scripts/run_bench.sh first.")
INPUT = candidates[-1]

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 解析：
# ===== impl=xxx, M=N=K=1024 =====
# [perf] median=1234.567 GFLOP/s
pat_case = re.compile(r"===== impl=([\w_]+), M=N=K=(\d+) =====")
pat_perf = re.compile(r"\[perf\]\s+median=([0-9.]+)\s+GFLOP/s")

data = {}  # impl -> size -> gflops
cur_impl = None
cur_size = None

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat_case.search(line)
        if m:
            cur_impl = m.group(1)
            cur_size = int(m.group(2))
            data.setdefault(cur_impl, {})
            continue

        m = pat_perf.search(line)
        if m and cur_impl is not None and cur_size is not None:
            g = float(m.group(1))
            data[cur_impl][cur_size] = g
            cur_impl = None
            cur_size = None

FP32_IMPLS = [
    "naive",
    "tiled",
    "tiled_rb1x4",
    "tiled_rb2x4",
    "cublas",
    "cublaslt",
]

FP16_IMPLS = [
    "tiled_fp16acc",
    "tiled_fp16acc_rb1x4",
    "tiled_fp16acc_rb2x4",
    "wmma_fp16acc",
    "wmma_fp16acc_staged",
    "cublas_gemmex_fp16acc",
    "cublaslt_fp16acc",
]

def collect_sizes(impls):
    s = set()
    for impl in impls:
        if impl in data:
            s |= set(data[impl].keys())
    return sorted(s)

def plot_gflops(impls, sizes, title, out_path):
    plt.figure()
    for impl in impls:
        if impl not in data:
            continue
        ys = [data[impl].get(s, float("nan")) for s in sizes]
        plt.plot(sizes, ys, marker="o", label=impl)
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("GFLOP/s (median)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_relative(impls, baseline_impl, sizes, title, ylabel, out_path):
    if baseline_impl not in data:
        print(f"[WARN] baseline {baseline_impl} not found, skip {out_path.name}")
        return

    baseline = data[baseline_impl]

    plt.figure()
    for impl in impls:
        if impl not in data or impl == baseline_impl:
            continue

        ys = []
        for s in sizes:
            if s in data[impl] and s in baseline and baseline[s] != 0:
                ys.append(100.0 * data[impl][s] / baseline[s])
            else:
                ys.append(float("nan"))
        plt.plot(sizes, ys, marker="o", label=f"{impl}/{baseline_impl}")

    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# FP32 plots
fp32_sizes = collect_sizes(FP32_IMPLS)
out_fp32_gflops = OUT_DIR / "gflops_fp32.png"
out_fp32_rel = OUT_DIR / "rel_to_cublas_fp32.png"

plot_gflops(
    FP32_IMPLS,
    fp32_sizes,
    "FP32 GEMM Throughput (CUDA events, median)",
    out_fp32_gflops,
)

plot_relative(
    FP32_IMPLS,
    "cublas",
    fp32_sizes,
    "Relative Throughput vs cuBLAS (FP32)",
    "Percent of cuBLAS (%)",
    out_fp32_rel,
)

# FP16 / Tensor Core plots
fp16_sizes = collect_sizes(FP16_IMPLS)
out_fp16_gflops = OUT_DIR / "gflops_fp16.png"
out_fp16_rel = OUT_DIR / "rel_to_cublas_fp16.png"

plot_gflops(
    FP16_IMPLS,
    fp16_sizes,
    "FP16 input + FP32 accumulate Throughput (CUDA events, median)",
    out_fp16_gflops,
)

plot_relative(
    FP16_IMPLS,
    "wmma_fp16acc_staged",
    fp16_sizes,
    "Relative Throughput vs cuBLAS GemmEx FP16acc",
    "Percent of cuBLAS GemmEx FP16acc (%)",
    out_fp16_rel,
)

print(f"[OK] Parsed: {INPUT}")
print(f"[OK] Wrote: {out_fp32_gflops}")
print(f"[OK] Wrote: {out_fp32_rel}")
print(f"[OK] Wrote: {out_fp16_gflops}")
print(f"[OK] Wrote: {out_fp16_rel}")