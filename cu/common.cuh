// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  common.cuh — Shared definitions for all neuro-spike CUDA kernels
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>
#include <cstdint>

// ── Numeric constants ──────────────────────────────────────────────
#define SHIP_PI       3.14159265358979323846f
#define SHIP_E        2.71828182845904523536f
#define SHIP_EPS      1e-8f

// ── Warp utility ──────────────────────────────────────────────────
#define WARP_SIZE 32

// Fast warp reduction (sum)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

// Fast warp reduction (max)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    return val;
}

// Fast warp reduction (int sum)
__device__ __forceinline__ int warp_reduce_sum_int(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

// Fast warp reduction (int min)
__device__ __forceinline__ int warp_reduce_min_int(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = min(val, __shfl_down_sync(0xffffffffu, val, offset));
    return val;
}

// ── Simple LCG random (inline, no cuRAND dependency) ──────────────
__device__ __forceinline__ unsigned int lcg_next(unsigned int state) {
    return state * 1664525u + 1013904223u;
}

// Convert an already-advanced LCG state to a float in [0, 1).
// Callers must call lcg_next() before this function to advance
// the RNG state; lcg_float itself is a pure conversion.
__device__ __forceinline__ float lcg_float(unsigned int state) {
    return (float)(state >> 8) * (1.0f / 16777216.0f);
}

// ── Bitpacking utilities ────────────────────────────────────────────
//
// Binary layout:  32 values per u32, 1 bit each.
//   Bit i of word w encodes element (w * 32 + i).
//   0 = false/zero, 1 = true/one.
//
// Ternary layout: 16 values per u32, 2 bits each.
//   Bits (2*i, 2*i+1) of word w encode element (w * 16 + i).
//   0b00 = 0, 0b01 = +1, 0b10 = -1, 0b11 = 0 (reserved, decoded as 0).
//
// Alignment: packed buffers should be 4-byte aligned for vectorized loads.
// Warp-specialized and tensor-core friendly layouts are future work.

#define BINARY_VALUES_PER_WORD  32
#define TERNARY_VALUES_PER_WORD 16

// Number of u32 words needed to pack n binary values.
__host__ __device__ __forceinline__ unsigned int binary_words(unsigned int n) {
    return (n + BINARY_VALUES_PER_WORD - 1) / BINARY_VALUES_PER_WORD;
}

// Number of u32 words needed to pack n ternary values.
__host__ __device__ __forceinline__ unsigned int ternary_words(unsigned int n) {
    return (n + TERNARY_VALUES_PER_WORD - 1) / TERNARY_VALUES_PER_WORD;
}

// ── Binary unpack (device) ──────────────────────────────────────────
// Unpack 1-bit values from packed[] into out[].
// out[i] = 1 if bit i is set, 0 otherwise.
__device__ __forceinline__ int unpack_binary(const unsigned int* packed, unsigned int idx) {
    unsigned int word = idx / BINARY_VALUES_PER_WORD;
    unsigned int bit  = idx % BINARY_VALUES_PER_WORD;
    return (packed[word] >> bit) & 1u;
}

// ── Ternary unpack (device) ─────────────────────────────────────────
// Unpack 2-bit values from packed[] into out[] as signed char {-1, 0, +1}.
// Encoding: 0b00 -> 0, 0b01 -> +1, 0b10 -> -1, 0b11 -> 0.
__device__ __forceinline__ int unpack_ternary(const unsigned int* packed, unsigned int idx) {
    unsigned int word = idx / TERNARY_VALUES_PER_WORD;
    unsigned int bit  = (idx % TERNARY_VALUES_PER_WORD) * 2;
    unsigned int code = (packed[word] >> bit) & 3u;
    // Decode: 0b01 -> +1, 0b10 -> -1, else 0.
    return (code == 1) ? 1 : (code == 2) ? -1 : 0;
}

// ── Vectorized 4-word load helper ───────────────────────────────────
// Loads 4 consecutive u32 values. Uses 128-bit vectorized load when
// the effective address is 16-byte aligned, falls back to scalar loads
// otherwise. Callers must ensure at least 4 words beyond offset.
__device__ __forceinline__ void load_u32x4(const unsigned int* base, unsigned int offset,
                                            unsigned int& a, unsigned int& b,
                                            unsigned int& c, unsigned int& d) {
    // Guard: effective address must be 16-byte aligned for 128-bit load.
    if ((reinterpret_cast<uintptr_t>(base + offset) & 0xFu) != 0) {
        // Fallback: scalar loads for unaligned access.
        a = base[offset]; b = base[offset + 1];
        c = base[offset + 2]; d = base[offset + 3];
        return;
    }
    const uint4 v = *reinterpret_cast<const uint4*>(base + offset);
    a = v.x; b = v.y; c = v.z; d = v.w;
}

// ── Bulk binary unpack ──────────────────────────────────────────────
// Unpack n binary values from packed[] into out[].
// Each thread handles one element; caller must launch with >= n threads.
__device__ void unpack_binary_bulk(const unsigned int* packed, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = unpack_binary(packed, idx);
    }
}

// ── Bulk ternary unpack ─────────────────────────────────────────────
// Unpack n ternary values from packed[] into out[] as int {-1, 0, +1}.
__device__ void unpack_ternary_bulk(const unsigned int* packed, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = unpack_ternary(packed, idx);
    }
}
