// Copyright 2026 Raul Mc
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  ternary_gemm.cu — Group-scaled ternary GEMV / GEMM kernels
//
//  Kernels exported (name-exact for PTX symbol lookup):
//    ternary_gemv  — y = scale(W) @ x   (M×K ternary × K vector)
//    ternary_gemm  — C = scale(W) @ B   (M×K ternary × K×N dense)
//
//  Weight encoding (see common.cuh / bitpacking.rs):
//    16 ternary values per u32; bits (2*i, 2*i+1):
//      00 → 0,  01 → +1,  10 → -1,  11 → 0
//
//  Scales: per-row groups along K, row-major
//    scales[m * n_groups + (k / group_size)], n_groups = ceil(K / group_size)
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

// ════════════════════════════════════════════════════════════════════
//  ternary_gemv
//
//  y = scale(W) @ x
//
//  packed_w : M rows × ternary_words(K) u32, row-major
//  scales   : M × ceil(K/group_size) f32, row-major groups along K
//  x        : K f32
//  y        : M f32
//  skip_zeros: if non-zero, skip w==0 (optional sparse path)
//
//  Launch: any 1-D grid; threads stride over M rows (one row per thread
//  step). Prefer enough threads to cover M.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void ternary_gemv(
    const unsigned int* __restrict__ packed_w,
    const float*        __restrict__ scales,
    const float*        __restrict__ x,
    float*              __restrict__ y,
    int M,
    int K,
    int group_size,
    int skip_zeros)
{
    if (M <= 0 || K <= 0 || group_size <= 0) return;
    if (packed_w == nullptr || scales == nullptr || x == nullptr || y == nullptr)
        return;

    const unsigned int words_k = ternary_words((unsigned int)K);
    const int n_groups = (K + group_size - 1) / group_size;

    // Block-strided over output rows.
    for (int m = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         m < M;
         m += (int)(gridDim.x * blockDim.x))
    {
        const unsigned int* __restrict__ row_w =
            packed_w + (size_t)m * words_k;
        const float* __restrict__ row_s =
            scales + (size_t)m * (size_t)n_groups;

        float acc = 0.0f;

        for (int k = 0; k < K; ++k) {
            const int w = unpack_ternary(row_w, (unsigned int)k);
            if (skip_zeros != 0 && w == 0) continue;

            const float s = row_s[k / group_size];
            // w ∈ {-1, 0, +1}: multiply as float
            acc = fmaf(s * (float)w, x[k], acc);
        }

        y[m] = acc;
    }
}

// ════════════════════════════════════════════════════════════════════
//  ternary_gemm
//
//  C = scale(W) @ B
//
//  packed_w : M rows × ternary_words(K) u32, row-major
//  scales   : M × ceil(K/group_size) f32, row-major groups along K
//  B        : K × N f32, row-major
//  C        : M × N f32, row-major
//  skip_zeros: if non-zero, skip w==0 (optional sparse path)
//
//  Launch: 1-D grid over flattened M*N output elements (one thread per
//  (m, n) with early return for out-of-range threads).
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void ternary_gemm(
    const unsigned int* __restrict__ packed_w,
    const float*        __restrict__ scales,
    const float*        __restrict__ B,
    float*              __restrict__ C,
    int M,
    int K,
    int N,
    int group_size,
    int skip_zeros)
{
    if (M <= 0 || K <= 0 || N <= 0 || group_size <= 0) return;
    if (packed_w == nullptr || scales == nullptr || B == nullptr || C == nullptr)
        return;

    // Explicit 64-bit product (avoid LLP64 `long` width surprises).
    const int64_t total = (int64_t)M * (int64_t)N;
    const int64_t tid =
        (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (tid >= total) return;

    const int m = (int)(tid / (int64_t)N);
    const int n = (int)(tid % (int64_t)N);

    const unsigned int words_k = ternary_words((unsigned int)K);
    const int n_groups = (K + group_size - 1) / group_size;

    const unsigned int* __restrict__ row_w =
        packed_w + (size_t)m * words_k;
    const float* __restrict__ row_s =
        scales + (size_t)m * (size_t)n_groups;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        const int w = unpack_ternary(row_w, (unsigned int)k);
        if (skip_zeros != 0 && w == 0) continue;

        const float s = row_s[k / group_size];
        // B is K×N row-major: B[k, n] = B[k * N + n]
        acc = fmaf(s * (float)w, B[(size_t)k * (size_t)N + (size_t)n], acc);
    }

    C[(size_t)m * (size_t)N + (size_t)n] = acc;
}
