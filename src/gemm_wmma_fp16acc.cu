// src/gemm_wmma_fp16acc.cu
#include "utils.cuh"

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// Minimal WMMA GEMM:
// A: [M,K] half, row-major
// B: [K,N] half, row-major
// C: [M,N] float, row-major
//
// Current limitation:
// - requires M, N, K to be multiples of 16
// - minimal demo, no boundary handling

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// block has 8 warps = 256 threads
// arrange as 2 x 4 warp tiles => block computes 32 x 64 C tile
constexpr int WARPS_PER_BLOCK_M = 2;
constexpr int WARPS_PER_BLOCK_N = 4;
constexpr int WARPS_PER_BLOCK   = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N; // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;                  // 256

__global__ void gemm_wmma_fp16acc_kernel(const half* A, const half* B, float* C,
                                         int M, int N, int K) {
  const int warp_id = threadIdx.x / 32;   // 0..7
  const int lane_id = threadIdx.x % 32;

  (void)lane_id; // silence warning

  const int warp_m = warp_id / WARPS_PER_BLOCK_N; // 0..1
  const int warp_n = warp_id % WARPS_PER_BLOCK_N; // 0..3

  const int c_row = (blockIdx.y * WARPS_PER_BLOCK_M + warp_m) * WMMA_M;
  const int c_col = (blockIdx.x * WARPS_PER_BLOCK_N + warp_n) * WMMA_N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

    const half* A_ptr = A + c_row * K + k0;   // A tile: [c_row : c_row+16, k0 : k0+16]
    const half* B_ptr = B + k0 * N + c_col;   // B tile: [k0 : k0+16, c_col : c_col+16]

    wmma::load_matrix_sync(a_frag, A_ptr, K);
    wmma::load_matrix_sync(b_frag, B_ptr, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* C_ptr = C + c_row * N + c_col;
  wmma::store_matrix_sync(C_ptr, c_frag, N, wmma::mem_row_major);
}

void launch_gemm_wmma_fp16acc(const half* dA, const half* dB, float* dC,
                              int M, int N, int K, cudaStream_t stream = nullptr) {
  // minimal demo: only support multiples of 16
  if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
    std::fprintf(stderr,
                 "gemm_wmma_fp16acc: currently requires M,N,K to be multiples of 16. "
                 "Got M=%d N=%d K=%d\n",
                 M, N, K);
    std::exit(EXIT_FAILURE);
  }

  dim3 block(THREADS_PER_BLOCK);
  dim3 grid(N / (WMMA_N * WARPS_PER_BLOCK_N),
            M / (WMMA_M * WARPS_PER_BLOCK_M));

  gemm_wmma_fp16acc_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}