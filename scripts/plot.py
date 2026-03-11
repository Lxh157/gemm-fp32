#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt

# 运行脚本：python3 scripts/plot.py

RAW_DIR = Path("results/raw")
OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP32_CANDIDATES = sorted(RAW_DIR.glob("bench_fp32_*.txt"))
FP16_CANDIDATES = sorted(RAW_DIR.glob("bench_fp16_*.txt"))

INPUT_FP32 = FP32_CANDIDATES[-1] if FP32_CANDIDATES else None
INPUT_FP16 = FP16_CANDIDATES[-1] if FP16_CANDIDATES else None

if INPUT_FP32 is None and INPUT_FP16 is None:
    raise FileNotFoundError(
        "No results/raw/bench_fp32_*.txt or results/raw/bench_fp16_*.txt found. "
        "Run scripts/run_bench.sh first."
    )

# 解析：
# ===== impl=xxx, M=N=K=1024 =====
# [perf] median=1234.567 GFLOP/s
PAT_CASE = re.compile(r"===== impl=([\w_]+), M=N=K=(\d+) =====")
PAT_PERF = re.compile(r"\[perf\]\s+median=([0-9.]+)\s+GFLOP/s")

FP32_IMPLS = [
    "naive",
    "tiled",
    "tiled_rb1x4",
    "tiled_rb2x4",
    # "cublas",
    "cublaslt",
]

FP16_IMPLS = [
    "tiled_fp16acc",
    "tiled_fp16acc_rb1x4",
    "tiled_fp16acc_rb2x4",
    "wmma_fp16acc",
    "wmma_fp16acc_staged",
    # 不放 staged_db 到主图里
    "wmma_fp16acc_staged_cpasync",
    "wmma_fp16acc_staged_cpasync_k32",
    # "cublas_gemmex_fp16acc",
    "cublaslt_fp16acc",
]


def parse_bench_file(path: Path):
    data = {}  # impl -> size -> gflops
    cur_impl = None
    cur_size = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PAT_CASE.search(line)
            if m:
                cur_impl = m.group(1)
                cur_size = int(m.group(2))
                data.setdefault(cur_impl, {})
                continue

            m = PAT_PERF.search(line)
            if m and cur_impl is not None and cur_size is not None:
                g = float(m.group(1))
                data[cur_impl][cur_size] = g
                cur_impl = None
                cur_size = None

    return data


def collect_sizes(data, impls):
    sizes = set()
    for impl in impls:
        if impl in data:
            sizes |= set(data[impl].keys())
    return sorted(sizes)


def plot_gflops(data, impls, sizes, title, out_path):
    if not sizes:
        print(f"[WARN] no sizes found, skip {out_path.name}")
        return

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


def plot_relative(data, impls, baseline_impl, sizes, title, ylabel, out_path):
    if not sizes:
        print(f"[WARN] no sizes found, skip {out_path.name}")
        return

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
if INPUT_FP32 is not None:
    data_fp32 = parse_bench_file(INPUT_FP32)
    fp32_sizes = collect_sizes(data_fp32, FP32_IMPLS)

    out_fp32_gflops = OUT_DIR / "gflops_fp32.png"
    out_fp32_rel = OUT_DIR / "rel_to_cublas_fp32.png"

    plot_gflops(
        data_fp32,
        FP32_IMPLS,
        fp32_sizes,
        "FP32 GEMM Throughput (CUDA events, median)",
        out_fp32_gflops,
    )

    plot_relative(
        data_fp32,
        FP32_IMPLS,
        "cublaslt",
        fp32_sizes,
        "Relative Throughput vs cuBLASLt (FP32)",
        "Percent of cuBLASLt (%)",
        out_fp32_rel,
    )

    print(f"[OK] Parsed FP32: {INPUT_FP32}")
    print(f"[OK] Wrote: {out_fp32_gflops}")
    print(f"[OK] Wrote: {out_fp32_rel}")
else:
    print("[WARN] No bench_fp32_*.txt found, skip FP32 plots.")


# FP16 / Tensor Core plots
if INPUT_FP16 is not None:
    data_fp16 = parse_bench_file(INPUT_FP16)
    fp16_sizes = collect_sizes(data_fp16, FP16_IMPLS)

    out_fp16_gflops = OUT_DIR / "gflops_fp16.png"
    out_fp16_rel = OUT_DIR / "rel_to_cublaslt_fp16.png"

    plot_gflops(
        data_fp16,
        FP16_IMPLS,
        fp16_sizes,
        "FP16 input + FP32 accumulate Throughput (CUDA events, median)",
        out_fp16_gflops,
    )

    plot_relative(
        data_fp16,
        FP16_IMPLS,
        "cublaslt_fp16acc",
        fp16_sizes,
        "Relative Throughput vs cuBLASLt FP16acc",
        "Percent of cuBLASLt FP16acc (%)",
        out_fp16_rel,
    )

    print(f"[OK] Parsed FP16: {INPUT_FP16}")
    print(f"[OK] Wrote: {out_fp16_gflops}")
    print(f"[OK] Wrote: {out_fp16_rel}")
else:
    print("[WARN] No bench_fp16_*.txt found, skip FP16 plots.")