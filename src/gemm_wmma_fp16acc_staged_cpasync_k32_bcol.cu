#include "utils.cuh"

#include <mma.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// WMMA tile
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// block = 2 x 4 warps => 32 x 64 output tile
constexpr int WARPS_PER_BLOCK_M = 2;
constexpr int WARPS_PER_BLOCK_N = 4;
constexpr int WARPS_PER_BLOCK   = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N; // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;                  // 256

constexpr int BLOCK_M = WARPS_PER_BLOCK_M * WMMA_M; // 32
constexpr int BLOCK_N = WARPS_PER_BLOCK_N * WMMA_N; // 64
constexpr int BLOCK_K = 2 * WMMA_K;                 // 32

// A shared layout: [BLOCK_M][BLOCK_K + skew]
constexpr int SKEW_HALF_A = 8;
constexpr int SMEM_STRIDE_A = BLOCK_K + SKEW_HALF_A; // 40

// B shared layout: store as [BLOCK_N][BLOCK_K + skew]
// i.e. transpose the staged tile into shared
constexpr int SKEW_HALF_B = 8;
constexpr int SMEM_STRIDE_B_T = BLOCK_K + SKEW_HALF_B; // 40

constexpr int CHUNK_BYTES = 16;
constexpr int CHUNK_HALF  = CHUNK_BYTES / sizeof(half); // 8 half

// A tile: [32, 32]
constexpr int A_ROW_CHUNKS = BLOCK_K / CHUNK_HALF;      // 4
constexpr int A_NUM_CHUNKS = BLOCK_M * A_ROW_CHUNKS;    // 128

// For B global tile [32, 64], we still enumerate by original row-major chunks
constexpr int B_ROW_CHUNKS = BLOCK_N / CHUNK_HALF;      // 8
constexpr int B_NUM_CHUNKS = BLOCK_K * B_ROW_CHUNKS;    // 256

__device__ __forceinline__ void cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// A: same as k32 baseline
__device__ __forceinline__ void load_A_stage_cpasync_k32(
    const half* __restrict__ A,
    half* __restrict__ smemA_stage,
    int block_row, int k0, int K, int tid) {

    for (int chunk = tid; chunk < A_NUM_CHUNKS; chunk += THREADS_PER_BLOCK) {
        const int row = chunk / A_ROW_CHUNKS;          // 0..31
        const int chunk_in_row = chunk % A_ROW_CHUNKS; // 0..3
        const int col = chunk_in_row * CHUNK_HALF;     // 0,8,16,24

        half* smem_dst = &smemA_stage[row * SMEM_STRIDE_A + col];
        const half* gmem_src = &A[(block_row + row) * K + (k0 + col)];

        cp_async_cg_16B(smem_dst, gmem_src);
    }
}

// B: load from global row-major [k][n], but stage into shared as transposed [n][k]
__device__ __forceinline__ void load_B_stage_transposed_sync_k32(
    const half* __restrict__ B,
    half* __restrict__ smemB_stage_t,
    int block_col, int k0, int N, int tid) {

    // each thread handles strided scalar copies for simplicity/correctness first
    for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
        const int row = idx / BLOCK_N; // k dimension: 0..31
        const int col = idx % BLOCK_N; // n dimension: 0..63

        const half v = B[(k0 + row) * N + (block_col + col)];
        smemB_stage_t[col * SMEM_STRIDE_B_T + row] = v;
    }
}

__global__ void gemm_wmma_fp16acc_staged_cpasync_k32_bcol_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ half smemA[2][BLOCK_M * SMEM_STRIDE_A];
    __shared__ half smemB_t[2][BLOCK_N * SMEM_STRIDE_B_T];

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    (void)lane_id;

    const int warp_m = warp_id / WARPS_PER_BLOCK_N; // 0..1
    const int warp_n = warp_id % WARPS_PER_BLOCK_N; // 0..3

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int c_row = block_row + warp_m * WMMA_M;
    const int c_col = block_col + warp_n * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int read_buf = 0;

    // preload stage 0
    load_A_stage_cpasync_k32(A, smemA[read_buf], block_row, 0, K, tid);
    cp_async_commit();
    load_B_stage_transposed_sync_k32(B, smemB_t[read_buf], block_col, 0, N, tid);
    cp_async_wait_all();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        const int next_k0   = k0 + BLOCK_K;
        const int write_buf = read_buf ^ 1;

        if (next_k0 < K) {
            load_A_stage_cpasync_k32(A, smemA[write_buf], block_row, next_k0, K, tid);
            cp_async_commit();
            load_B_stage_transposed_sync_k32(B, smemB_t[write_buf], block_col, next_k0, N, tid);
        }

        // kk = 0
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            const half* A_smem_ptr = &smemA[read_buf][(warp_m * WMMA_M) * SMEM_STRIDE_A + 0];
            const half* B_smem_ptr = &smemB_t[read_buf][(warp_n * WMMA_N) * SMEM_STRIDE_B_T + 0];

            wmma::load_matrix_sync(a_frag, A_smem_ptr, SMEM_STRIDE_A);
            wmma::load_matrix_sync(b_frag, B_smem_ptr, SMEM_STRIDE_B_T);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // kk = 16
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            const half* A_smem_ptr = &smemA[read_buf][(warp_m * WMMA_M) * SMEM_STRIDE_A + WMMA_K];
            const half* B_smem_ptr = &smemB_t[read_buf][(warp_n * WMMA_N) * SMEM_STRIDE_B_T + WMMA_K];

            wmma::load_matrix_sync(a_frag, A_smem_ptr, SMEM_STRIDE_A);
            wmma::load_matrix_sync(b_frag, B_smem_ptr, SMEM_STRIDE_B_T);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        if (next_k0 < K) {
            cp_async_wait_all();
        }
        __syncthreads();

        read_buf ^= 1;
    }

    float* C_ptr = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_ptr, c_frag, N, wmma::mem_row_major);
}

void launch_gemm_wmma_fp16acc_staged_cpasync_k32_bcol(
    const half* dA, const half* dB, float* dC,
    int M, int N, int K, cudaStream_t stream) {

    if (M % 16 != 0 || N % 16 != 0 || K % 32 != 0) {
        std::fprintf(stderr,
            "gemm_wmma_fp16acc_staged_cpasync_k32_bcol: currently requires M,N to be multiples of 16 and K to be a multiple of 32. "
            "Got M=%d N=%d K=%d\n",
            M, N, K);
        std::exit(EXIT_FAILURE);
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(
        N / BLOCK_N,
        M / BLOCK_M
    );

    gemm_wmma_fp16acc_staged_cpasync_k32_bcol_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}