# CUDA GEMM Optimization Practice（FP32 / FP16 / Tensor Core）

## 项目目标

本项目用于练习 CUDA / GPU 性能优化与 profiling，围绕 GEMM（矩阵乘法）实现一条清晰的优化链路，并用 **benchmark 数据 + Nsight Compute 指标**形成可复现实验结论。

项目分成两个阶段：

- **Phase 1（RTX 4060 Laptop / WSL2）**：从 naive / tiled / register blocking，推进到 FP16 / Tensor Core / cuBLASLt baseline，形成完整的入门优化链路。
- **Phase 2（RTX 4090 Server / CUDA 11.8）**：围绕 Tensor Core 主干继续推进，重点优化 `wmma_fp16acc_staged_cpasync` → `k32` → `skew16` 这条 Ada 路线，并与 `cublas_gemmex_fp16acc` / `cublaslt_fp16acc` 做对比。


## 当前进展

### Phase 1（4060）
- 搭建 benchmark 框架：参数化 M/N/K，CUDA events 计时，输出 min / median / avg 与 GFLOP/s
- naive GEMM kernel（CPU reference correctness check）
- tiled GEMM kernel（shared memory tiling）
- tiled_rb1x4 GEMM kernel（thread coarsening）
- tiled_rb2x4 GEMM kernel（thread coarsening）
- 接入 cublasSgemm 做 FP32 baseline
- 接入 cublasLt（FP32）做 baseline
- tiled_fp16acc GEMM kernel（FP16 input + FP32 accumulate，non-Tensor-Core）
- tiled_fp16acc_rb1x4 GEMM kernel（FP16 input + FP32 accumulate，non-Tensor-Core）
- tiled_fp16acc_rb2x4 GEMM kernel（FP16 input + FP32 accumulate，non-Tensor-Core）
- wmma_fp16acc（minimal demo）
- wmma_fp16acc_staged（shared-memory staged WMMA）
- wmma_fp16acc_staged_cpasync（cp.async 流水化 staged WMMA）

### Phase 2（4090）
- 接入 `cublas_gemmex_fp16acc` 与 `cublaslt_fp16acc` 作为 FP16 Tensor Core baseline
- 补齐 `wmma_fp16acc_staged_cpasync_k32`
- 补齐 shared-layout / pitch sweep：
  - `wmma_fp16acc_staged_cpasync_k32_skew16`
  - `wmma_fp16acc_staged_cpasync_k32_skewA16_B8`
  - `wmma_fp16acc_staged_cpasync_k32_skewA8_B16`
- 通过额外 sweep 确认当前最强 custom kernel 仍为：
  - `wmma_fp16acc_staged_cpasync_k32_skew16`
- 已完成 4090 阶段的 Nsight Compute 对比分析：
  - `wmma_fp16acc_staged_cpasync_k32`
  - `wmma_fp16acc_staged_cpasync_k32_skew16`
  - `cublaslt_fp16acc`
- 下一阶段准备从当前 WMMA 路线进一步下沉到更低层的数据通路实现：
  - `src/gemm_mma_ldmatrix_fp16acc_stage2.cu`
  - `impl = mma_ldmatrix_fp16acc_stage2`


## 环境

### Phase 1（RTX 4060 Laptop）
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU（SM89 / Ada）
- OS: WSL2 Ubuntu
- CUDA Toolkit / nvcc: 12.8（V12.8.93）
- Build: CMake + make
- Tools: Nsight Compute / Nsight Systems

### Phase 2（RTX 4090 Server）
- GPU: NVIDIA GeForce RTX 4090（SM89 / Ada）
- OS: Ubuntu（remote server）
- CUDA Toolkit / nvcc: 11.8
- 推荐环境变量：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

- Build: CMake + make
- Tools: Nsight Compute / Nsight Systems


## 目录结构

```text
gemm-fp16/
  src/
    main_bench.cu            # benchmark 入口
    gemm_naive.cu
    gemm_tiled.cu
    gemm_tiled_rb1x4.cu
    gemm_tiled_rb2x4.cu
    gemm_cublas.cu
    cublaslt_baseline.cu

    gemm_tiled_fp16acc.cu
    gemm_tiled_fp16acc_rb1x4.cu
    gemm_tiled_fp16acc_rb2x4.cu

    gemm_wmma_fp16acc.cu
    gemm_wmma_fp16acc_staged.cu
    gemm_wmma_fp16acc_staged_db.cu
    gemm_wmma_fp16acc_staged_cpasync.cu
    gemm_wmma_fp16acc_staged_cpasync_k32.cu
    gemm_wmma_fp16acc_staged_cpasync_k32_bcol.cu
    gemm_wmma_fp16acc_staged_cpasync_k32_4x2.cu
    gemm_wmma_fp16acc_staged_cpasync_k32_split.cu
    gemm_wmma_fp16acc_staged_cpasync_k32_skew16.cu
    gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B8.cu
    gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B16.cu

    gemm_cublas_gemmex_fp16acc.cu
    cublaslt_fp16acc.cu

    # 下一阶段（planned）
    gemm_mma_ldmatrix_fp16acc_stage2.cu

    utils.cuh
  scripts/
    run_bench.sh
    collect_env.sh
    plot.py
  results/
    raw/
    plots/
  profiles/
    nsys/
    ncu/
  logs/
```


## 复现方式（Build / Run）

### 1) 构建

#### Phase 1（本地 / WSL2）
```bash
cd ~/gemm-fp16/build
cmake ..
make -j
```

#### Phase 2（4090 server）
```bash
cd ~/gemm-fp16/build
cmake ..
make -j
```

构建产物：`build/bench_gemm`

### 2) 单点运行

```bash
./bench_gemm --impl wmma_fp16acc_staged_cpasync_k32_skew16 --M 1024 --N 1024 --K 1024 --warmup 3 --repeat 10
./bench_gemm --impl wmma_fp16acc_staged_cpasync_k32_skew16 --M 2048 --N 2048 --K 2048 --warmup 3 --repeat 10
```

### 3) 批量 benchmark

#### Phase 1（4060 全链路）
```bash
PROFILE_SET=phase1_4060_all bash scripts/run_bench.sh
```

默认测试：
- 尺寸：`256 / 512 / 1024`
- 实现：Phase 1 全链路（FP32 + FP16 + WMMA early chain）

#### Phase 2（4090 Tensor Core 主线）
```bash
PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

默认测试：
- 尺寸：`1024 / 2048`
- 实现：
  - `wmma_fp16acc_staged_cpasync`
  - `wmma_fp16acc_staged_cpasync_k32`
  - `wmma_fp16acc_staged_cpasync_k32_skew16`
  - `cublas_gemmex_fp16acc`
  - `cublaslt_fp16acc`

也可以覆盖尺寸：

```bash
PROFILE_SET=phase2_4090_tc SIZES_OVERRIDE="1024 2048" bash scripts/run_bench.sh
```

### 4) 画图

```bash
python3 scripts/plot.py
```

输出图表：

- `results/plots/gflops_phase1_fp32.png`
- `results/plots/gflops_phase1_fp16.png`
- `results/plots/rel_to_cublaslt_phase1_fp16.png`
- `results/plots/gflops_phase2_4090_tc.png`
- `results/plots/rel_to_cublaslt_phase2_4090_tc.png`


## 实验口径说明

### 1) 性能对比
- 计时方式：CUDA events
- `warmup >= 3`，`repeat >= 10`
- 使用 **median** 作为稳定性能指标
- correctness check：默认对 CPU reference 做校验
  - FP32 kernels：`atol=1e-3, rtol=1e-3`
  - FP16 input kernels：`atol=2e-2, rtol=2e-2`

### 2) NCU profiling
- NCU 会显著扰动运行时间，因此 **NCU 输出的 ms/GFLOP/s 不用于最终性能结论**
- 建议口径：

```bash
ncu --set full --target-processes all --force-overwrite \
./build/bench_gemm --impl <impl> --M 2048 --N 2048 --K 2048 --warmup 0 --repeat 1 --no-check
```

## 当前结果

### Phase 1 — RTX 4060 Laptop（完整优化链路）

> Phase 1 使用 4060 Laptop + WSL2 环境，主要目的是把从 FP32 / non-TC 到 WMMA / cp.async 的完整优化链条跑通。该阶段结果不与 4090 阶段混图展示。

#### 表 A：FP32 路线（4060）

| Impl            |     256³ |     512³ |   1024³ | 1024³ 相对 cublas |
| --------------- | -------: | -------: | ------: | ----------------: |
| naive           |  606.815 |  680.148 |  693.506 |            10.98% |
| tiled           |  661.980 |  897.753 |  697.888 |            11.05% |
| tiled_rb1x4     | 1310.720 | 1814.145 | 1860.827 |            29.46% |
| **tiled_rb2x4** | 1489.455 | 2803.679 | **3363.516** |     **53.25%** |
| cublas          | 2048.000 | 4861.552 | 6316.723 |           100.00% |
| cublaslt        | 2048.000 | 4606.594 | 6307.224 |            99.85% |

#### 表 B：FP16 / Tensor Core 路线（4060）

| impl                              |     256³ |     512³ |    1024³ | 1024³ 相对 cublaslt_fp16acc |
| --------------------------------- | -------: | -------: | --------: | ---------------------------: |
| tiled_fp16acc                     |  728.178 |  885.668 |   733.783 |                        4.17% |
| tiled_fp16acc_rb1x4               | 1459.396 | 2340.571 |  2207.528 |                       12.56% |
| tiled_fp16acc_rb2x4               | 1639.681 | 3015.858 |  2972.575 |                       16.91% |
| wmma_fp16acc                      | 1574.438 | 3912.597 |  5932.537 |                       33.76% |
| wmma_fp16acc_staged               | 1525.201 | 3887.214 |  6045.027 |                       34.40% |
| **wmma_fp16acc_staged_cpasync**   | **2056.031** | **8499.097** | **14734.629** |               **80%+** |
| cublaslt_fp16acc                  | 4120.141 | 10485.760 | 18236.104 |                      100.00% |

> Phase 1 的主结论是：  
> register blocking 在 FP32 / non-TC 路线上都有效；而在 Tensor Core 路线上，真正决定性能上限的是数据供给路径，单纯 WMMA minimal 不足以接近库实现，引入 staged + cp.async 后才出现显著跃迁。该阶段原始 README 与图表主要记录的是这条完整教学式优化链路。 

#### Phase 1 图表
![Phase 1 FP32 throughput](results/plots/gflops_phase1_fp32.png)

![Phase 1 FP16 / Tensor Core throughput](results/plots/gflops_phase1_fp16.png)

![Phase 1 relative to cuBLASLt FP16acc](results/plots/rel_to_cublaslt_phase1_fp16.png)


### Phase 2 — RTX 4090 Server（Tensor Core 主线推进）

> Phase 2 使用 4090 server + CUDA 11.8，目标不再是重复 Phase 1 的全链路，而是只围绕 Tensor Core 主干继续推进，因此不再更新早期 FP32 / non-Tensor-Core kernels。

#### Phase 2 主测试集合
- `wmma_fp16acc_staged_cpasync`
- `wmma_fp16acc_staged_cpasync_k32`
- `wmma_fp16acc_staged_cpasync_k32_skew16`
- `cublas_gemmex_fp16acc`
- `cublaslt_fp16acc`

#### 表 C：4090，1024³（FP16 input + FP32 accumulate）

| impl                                    | 1024³ TFLOP/s | 相对 cublaslt_fp16acc |
| --------------------------------------- | ------------: | --------------------: |
| wmma_fp16acc_staged_cpasync             | 47.66         | 70.6%                 |
| wmma_fp16acc_staged_cpasync_k32         | 56.66         | 83.9%                 |
| **wmma_fp16acc_staged_cpasync_k32_skew16** | **80.66**   | **119.4%**            |
| cublas_gemmex_fp16acc                   | 66.91         | 98.9%                  |
| cublaslt_fp16acc                        | 67.65         | 100.0%                |

#### 表 D：4090，2048³（FP16 input + FP32 accumulate）

| impl                                    | 2048³ TFLOP/s | 相对 cublaslt_fp16acc |
| --------------------------------------- | ------------: | --------------------: |
| wmma_fp16acc_staged_cpasync             | 46.25         | 31.6%                 |
| wmma_fp16acc_staged_cpasync_k32         | 60.58         | 41.3%                 |
| **wmma_fp16acc_staged_cpasync_k32_skew16** | **90.42**   | **62.0%**             |
| cublas_gemmex_fp16acc                   | 146.03        | 99.7%                  |
| cublaslt_fp16acc                        | 146.51        | 100.0%                |

> Phase 2 的主结论是： 
> 在 4090 上，`wmma_fp16acc_staged_cpasync_k32_skew16` 已经成为当前最强 custom kernel；它在 `1024^3` 上不仅显著高于 `k32`，也已超过当前仓库口径的 `cublaslt_fp16acc` baseline，但在 `2048^3` 上仍明显落后 vendor。因此，shared layout / pitch 是一个非常有效的杠杆，但还不足以把大方阵完全拉到 vendor 的实现层级。 

#### Phase 2 图表
![Phase 2 4090 Tensor Core throughput](results/plots/gflops_phase2_4090_tc_4x8.png)

![Phase 2 4090 relative to cuBLASLt FP16acc](results/plots/rel_to_cublaslt_phase2_4090_tc_4x8.png)


## 4090 Profiling 结论（围绕 `k32` / `skew16` / `cublaslt_fp16acc`）

### 1) `skew16` 相对 `k32` 的提升是真实的
在 2048³ 上，Nsight Compute 显示：

- `Issue Slots Busy`：**25.21% → 37.42%**
- `Eligible Warps / Scheduler`：**0.35 → 0.60**
- `Warp Cycles / Issued Instruction`：**31.16 → 20.05**
- `Barrier stall`：**12.3 cycles → 6.1 cycles**
- `Memory Throughput`：**53.73 GB/s → 76.53 GB/s**
- `Compute (SM)`：**29.51% → 43.84%**

这说明 `skew16` 的收益机制是：  
**同时压低 barrier stall 与 operand/data-ready 相关等待，从而提高 warp readiness、issue 连续性以及 memory feed。**

### 2) `skew16` 的收益不能简单解释成“shared bank conflict 下降”
`shared excessive wavefronts` 并没有随着 `skew16` 同步明显改善，因此更准确的表述是：

- `skew16` 改变了 shared layout 之后，整体上改善了 operand feeding / warp readiness / issue continuity
- 但这个收益机制不能被简化成单一的 bank-conflict 叙述

### 3) 当前 custom kernel 与 cublasLt 仍处在不同 operating point
`cublaslt_fp16acc` 对应的 CUTLASS-like kernel 在 2048³ 上表现出：

- 更重的资源占用（`228 registers/thread`）
- 更高的 dynamic shared memory（`73.73 KB/block`）
- 更低的 occupancy（`16.67%`）
- 更强的 tensor-pipeline 主导 stall（`math pipeline unavailable = 24.6 cycles / 79.9%`）

这说明 vendor kernel 已进入一种**以极重 tile / pipeline 设计来追求 tensor-pipeline 饱和**；而当前 `WMMA + cp.async + K32 + skew16` kernel 仍然更依赖较健康的 warp supply / issue efficiency 去喂 tensor core。二者并不在同一个微内核设计层级上。


## 当前最佳 kernel

当前最强 custom kernel：

- 文件：`src/gemm_wmma_fp16acc_staged_cpasync_k32_skew16.cu`
- impl：`wmma_fp16acc_staged_cpasync_k32_skew16`

当前定位：

- 它已经是本仓库在 4090 上的 best custom kernel 候选主干
- 它不是最终答案，也不意味着已经整体超越 cublasLt
- 它证明了 shared-layout sweet spot 的有效性，同时也证明了继续简单 sweep pitch 并不能自动把大方阵拉近到 cublasLt 的实现层级


## 下一步

下一阶段将不再继续围绕更多 A/B skew 点做大规模 sweep，而是：

1. 保留 `wmma_fp16acc_staged_cpasync_k32_skew16` 作为当前 best custom baseline
2. 从当前 WMMA 路线下沉到更低层的数据通路实现：

```text
src/gemm_mma_ldmatrix_fp16acc_stage2.cu
impl = mma_ldmatrix_fp16acc_stage2
```

目标是获得比当前 `wmma::load_matrix_sync + wmma::mma_sync` 更强的 operand delivery 与 instruction scheduling control，从而在 Ada（SM89）上继续缩小与 vendor kernel 的差距。
