#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/run_bench.sh
#   PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
#   PROFILE_SET=phase1_4060_all bash scripts/run_bench.sh
#   BUILD_DIR=build WARMUP=3 REPEAT=10 bash scripts/run_bench.sh
#   SIZES_OVERRIDE="1024 2048" PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh

BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BUILD_DIR}/bench_gemm"
OUT_DIR="${OUT_DIR:-results/raw}"
LOG_DIR="${LOG_DIR:-logs}"

WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"
PROFILE_SET="${PROFILE_SET:-phase2_4090_tc}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] binary not found: ${BIN}"
  echo "Please build first: cmake -S . -B build && cmake --build build -j"
  exit 1
fi

case "${PROFILE_SET}" in
  phase1_4060_all)
    OUT_PREFIX="bench_phase1_4060_all"
    if [[ -n "${SIZES_OVERRIDE:-}" ]]; then
      read -r -a SIZES <<< "${SIZES_OVERRIDE}"
    else
      SIZES=(256 512 1024)
    fi
    IMPLS=(
      naive
      tiled
      tiled_rb1x4
      tiled_rb2x4
      cublas
      cublaslt
      tiled_fp16acc
      tiled_fp16acc_rb1x4
      tiled_fp16acc_rb2x4
      wmma_fp16acc
      wmma_fp16acc_staged
      wmma_fp16acc_staged_db
      wmma_fp16acc_staged_cpasync
      cublas_gemmex_fp16acc
      cublaslt_fp16acc
    )
    ;;
  phase2_4090_tc)
    OUT_PREFIX="bench_phase2_4090_tc"
    if [[ -n "${SIZES_OVERRIDE:-}" ]]; then
      read -r -a SIZES <<< "${SIZES_OVERRIDE}"
    else
      SIZES=(1024 2048)
    fi
    IMPLS=(
      wmma_fp16acc_staged_cpasync
      wmma_fp16acc_staged_cpasync_k32
      wmma_fp16acc_staged_cpasync_k32_skew16
      cublas_gemmex_fp16acc
      cublaslt_fp16acc
    )
    ;;
  *)
    echo "[ERROR] unknown PROFILE_SET=${PROFILE_SET}"
    echo "Supported PROFILE_SET values: phase1_4060_all, phase2_4090_tc"
    exit 1
    ;;
esac

TS=$(date +%Y%m%d_%H%M%S)
OUT_TXT="${OUT_DIR}/${OUT_PREFIX}_${TS}.txt"

{
  echo "# bench_gemm batch run"
  echo "# time: $(date)"
  echo "# profile_set=${PROFILE_SET}"
  echo "# build_dir=${BUILD_DIR}"
  echo "# binary=${BIN}"
  echo "# warmup=${WARMUP}, repeat=${REPEAT}, metric=median"
  echo "# sizes=${SIZES[*]}"
  echo "# impls=${IMPLS[*]}"
  echo
} | tee "${OUT_TXT}"

for s in "${SIZES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    echo "===== impl=${impl}, M=N=K=${s} =====" | tee -a "${OUT_TXT}"
    "${BIN}" \
      --impl "${impl}" \
      --M "${s}" --N "${s}" --K "${s}" \
      --warmup "${WARMUP}" --repeat "${REPEAT}" \
      2>&1 | tee -a "${OUT_TXT}"
    echo | tee -a "${OUT_TXT}"
  done
done

echo "[DONE] results saved to ${OUT_TXT}"
