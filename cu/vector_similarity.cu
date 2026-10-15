// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  vector_similarity.cu — GPU cosine-similarity kernels
//
//  Kernels exported:
//    cosine_similarity_batched  — compute similarity of Q query vectors
//                                 against K key vectors (Q×K matrix)
//    cosine_similarity_top_k    — like above but also write top-k indices
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

#define MAX_ROUTING_TOP_K 32
#define MAX_BLOCK_WARPS   32
// FTZ can produce finite scores below -1; every finite score must be eligible.
#define COSINE_SENTINEL   -INFINITY

// Canonical score for both entry points. SHIP_EPS is an additive denominator
// regularizer (1e-8f), NOT a per-vector squared-norm clamp. Zero vectors have
// dot == 0 and score 0; positive finite norms follow this same equation all
// the way down to the build's fast-math flush-to-zero boundary.
__device__ __forceinline__
float cosine_score(float dot, float norm_q, float norm_k)
{
    float denom = sqrtf(norm_q) * sqrtf(norm_k) + SHIP_EPS;
    return dot / denom;
}

// Small pairs use a scalar FMA chain; larger pairs use all 32 lanes, with
// lane 0 receiving the score. Both entry points use this same dimension-based
// arithmetic order, independent of block size. Sharing only the denominator
// is insufficient: different dot/norm summation orders can reverse near ties.
__device__ __forceinline__
float cosine_pair_score(const float* qv, const float* kv, int dim, int lane)
{
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_k = 0.0f;
    const bool scalar = dim <= WARP_SIZE;
    for (int i = scalar ? 0 : lane; i < dim; i += scalar ? 1 : WARP_SIZE) {
        float qi = qv[i];
        float ki = kv[i];
        dot = fmaf(qi, ki, dot);
        norm_q = fmaf(qi, qi, norm_q);
        norm_k = fmaf(ki, ki, norm_k);
    }
    if (!scalar) {
        dot = warp_reduce_sum(dot);
        norm_q = warp_reduce_sum(norm_q);
        norm_k = warp_reduce_sum(norm_k);
    }
    return cosine_score(dot, norm_q, norm_k);
}

__device__ __forceinline__
bool topk_better(
    float lhs_score,
    int lhs_idx,
    float rhs_score,
    int rhs_idx)
{
    return lhs_score > rhs_score ||
           (lhs_score == rhs_score && lhs_idx < rhs_idx);
}

__device__ __forceinline__
void topk_insert(
    float score,
    int idx,
    float (&scores)[MAX_ROUTING_TOP_K],
    int (&indices)[MAX_ROUTING_TOP_K],
    int top_k)
{
    if (top_k <= 0) return;

    int tail = top_k - 1;
    if (!topk_better(score, idx, scores[tail], indices[tail])) return;

    scores[tail] = score;
    indices[tail] = idx;

    for (int pos = tail; pos > 0; --pos) {
        if (!topk_better(scores[pos], indices[pos], scores[pos - 1], indices[pos - 1]))
            break;

        float tmp_score = scores[pos - 1];
        int tmp_idx = indices[pos - 1];
        scores[pos - 1] = scores[pos];
        indices[pos - 1] = indices[pos];
        scores[pos] = tmp_score;
        indices[pos] = tmp_idx;
    }
}

__device__ __forceinline__
void warp_reduce_best(
    float& score,
    int& idx,
    int& owner_lane)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_score = __shfl_down_sync(0xffffffffu, score, offset);
        int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
        int other_owner = __shfl_down_sync(0xffffffffu, owner_lane, offset);

        if (topk_better(other_score, other_idx, score, idx)) {
            score = other_score;
            idx = other_idx;
            owner_lane = other_owner;
        }
    }

    score = __shfl_sync(0xffffffffu, score, 0);
    idx = __shfl_sync(0xffffffffu, idx, 0);
    owner_lane = __shfl_sync(0xffffffffu, owner_lane, 0);
}

// ════════════════════════════════════════════════════════════════════
//  cosine_similarity_batched
//
//  Computes the cosine similarity between every (query, key) pair:
//    out[q * n_keys + k] = dot(Q[q], K[k]) / (|Q[q]| * |K[k]| + eps)
//
//  Grid: (n_keys, n_queries), one block per pair. Blocks must contain
//        32–1024 threads in full warps; the first warp computes the pair.
//
//  Params
//    queries    [n_queries × dim]  — query matrix (row-major)
//    keys       [n_keys   × dim]  — key matrix
//    out        [n_queries × n_keys] — output similarity matrix
//    n_queries, n_keys, dim
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void cosine_similarity_batched(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float*       __restrict__ out,
    int n_queries,
    int n_keys,
    int dim)
{
    // Block index identifies (query, key) pair
    int q = blockIdx.y;
    int k = blockIdx.x;
    if (q >= n_queries || k >= n_keys) return;

    const float* qv = queries + (long)q * dim;
    const float* kv = keys    + (long)k * dim;

    // The first full warp computes the pair. Additional warps need not
    // participate; there are no block barriers in this kernel.
    if (threadIdx.x >= WARP_SIZE) return;
    if (dim <= WARP_SIZE && threadIdx.x != 0) return;
    float score = cosine_pair_score(qv, kv, dim, threadIdx.x);
    if (threadIdx.x == 0)
        out[(long)q * n_keys + k] = score;
}

// ════════════════════════════════════════════════════════════════════
//  cosine_similarity_top_k
//
//  Like cosine_similarity_batched but for each query also writes
//  the top-K key indices (by similarity score) into `top_k_indices`.
//  Equal scores select the lower key index first. Both entry points use
//  cosine_pair_score with identical accumulation and normalization order.
//
//  Implementation: short vectors retain one key per thread; otherwise each
//  warp computes strided keys and keeps its candidates in lane 0 registers.
//  Both paths use cosine_pair_score, then the existing
//  warp-participating k-way merges to produce the final top-k indices.
//  Shared-memory usage stays bounded; no single-thread O(N · K) tail.
//
//  Shared memory: static only (8 KiB), no dynamic allocation needed.
//  The host should pass 0 for the dynamic shared-memory bytes parameter.
//
//  Params
//    queries         [n_queries × dim]
//    keys            [n_keys   × dim]
//    similarities    [n_queries × n_keys]  — full similarity matrix output
//    top_k_indices   [n_queries × top_k]   — indices of top-k keys per query
//    n_queries, n_keys, dim, top_k
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void cosine_similarity_top_k(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    float*       __restrict__ similarities,
    int*         __restrict__ top_k_indices,
    int n_queries,
    int n_keys,
    int dim,
    int top_k)
{
    int q = blockIdx.x;
    if (q >= n_queries) return;

    if (top_k <= 0) return;

    int requested_k = (top_k < n_keys) ? top_k : n_keys;
    int actual_k = (requested_k < MAX_ROUTING_TOP_K) ? requested_k : MAX_ROUTING_TOP_K;
    int* out_idx = top_k_indices + (long)q * top_k;

    if (requested_k <= 0) {
        for (int t = threadIdx.x; t < top_k; t += blockDim.x)
            out_idx[t] = -1;
        return;
    }

    const float* qv = queries + (long)q * dim;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_warp_scores[MAX_BLOCK_WARPS][MAX_ROUTING_TOP_K];
    __shared__ int s_warp_indices[MAX_BLOCK_WARPS][MAX_ROUTING_TOP_K];

    float local_scores[MAX_ROUTING_TOP_K];
    int local_indices[MAX_ROUTING_TOP_K];

#pragma unroll
    for (int i = 0; i < MAX_ROUTING_TOP_K; ++i) {
        local_scores[i] = COSINE_SENTINEL;
        local_indices[i] = INT_MAX;
    }

    const bool scalar = dim <= WARP_SIZE;
    int first_key = scalar ? threadIdx.x : warp_id;
    int key_stride = scalar ? blockDim.x : n_warps;
    for (int k = first_key; k < n_keys; k += key_stride) {
        const float* kv = keys + (long)k * dim;
        float similarity = cosine_pair_score(qv, kv, dim, lane);
        if (scalar || lane == 0) {
            similarities[(long)q * n_keys + k] = similarity;
            topk_insert(similarity, k, local_scores, local_indices, actual_k);
        }
    }

    int local_cursor = 0;
    float candidate_score = local_scores[0];
    int candidate_idx = local_indices[0];

    for (int slot = 0; slot < actual_k; ++slot) {
        float best_score = candidate_score;
        int best_idx = candidate_idx;
        int best_lane = lane;
        warp_reduce_best(best_score, best_idx, best_lane);

        if (lane == 0) {
            s_warp_scores[warp_id][slot] = best_score;
            s_warp_indices[warp_id][slot] = (best_score > COSINE_SENTINEL) ? best_idx : -1;
        }

        if (lane == best_lane) {
            ++local_cursor;
            candidate_score = (local_cursor < actual_k) ? local_scores[local_cursor] : COSINE_SENTINEL;
            candidate_idx = (local_cursor < actual_k) ? local_indices[local_cursor] : INT_MAX;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        int warp_cursor = 0;
        float warp_score = (lane < n_warps) ? s_warp_scores[lane][0] : COSINE_SENTINEL;
        int warp_index = (lane < n_warps) ? s_warp_indices[lane][0] : INT_MAX;

        for (int slot = 0; slot < actual_k; ++slot) {
            float best_score = warp_score;
            int best_idx = warp_index;
            int best_lane = lane;
            warp_reduce_best(best_score, best_idx, best_lane);

            if (lane == 0)
                out_idx[slot] = (best_score > COSINE_SENTINEL) ? best_idx : -1;

            if (lane == best_lane && lane < n_warps) {
                ++warp_cursor;
                warp_score = (warp_cursor < actual_k) ? s_warp_scores[lane][warp_cursor] : COSINE_SENTINEL;
                warp_index = (warp_cursor < actual_k) ? s_warp_indices[lane][warp_cursor] : INT_MAX;
            }
        }
    }

    for (int t = actual_k + threadIdx.x; t < top_k; t += blockDim.x)
        out_idx[t] = -1;
}
