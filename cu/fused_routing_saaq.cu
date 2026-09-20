// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  fused_routing_saaq.cu — fused routing / entropy / SAAQ selection
//
//  Kernels exported (name-exact for PTX symbol lookup in kernel.rs):
//    routing_softmax              — unfused baseline: logits → probabilities
//    saaq_select_fused            — single-block SAAQ argmax (no partials)
//    routing_saaq_fused_pass1    — softmax + entropy + top-k + SAAQ partials
//    fused_telemetry_reduce_pass2 — final on-device entropy + walker reduce
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

#define MAX_FUSED_TOP_K 8

__device__ __forceinline__
bool fused_better(float lhs_score, int lhs_idx, float rhs_score, int rhs_idx)
{
    return lhs_score > rhs_score ||
           (lhs_score == rhs_score && lhs_idx < rhs_idx);
}

__device__ __forceinline__
void fused_topk_insert(
    float score,
    int idx,
    float (&scores)[MAX_FUSED_TOP_K],
    int (&indices)[MAX_FUSED_TOP_K],
    int top_k)
{
    if (top_k <= 0) return;

    int tail = top_k - 1;
    if (!fused_better(score, idx, scores[tail], indices[tail])) return;

    scores[tail] = score;
    indices[tail] = idx;

    for (int pos = tail; pos > 0; --pos) {
        if (!fused_better(scores[pos], indices[pos], scores[pos - 1], indices[pos - 1]))
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
void fused_argmax_reduce(float& score, int& walker)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_score = __shfl_down_sync(0xffffffffu, score, offset);
        int other_walker = __shfl_down_sync(0xffffffffu, walker, offset);
        if (fused_better(other_score, other_walker, score, walker)) {
            score = other_score;
            walker = other_walker;
        }
    }
}

// Softmax that keeps +inf logits from becoming NaN via inf-inf.
// +inf mass is shared uniformly among +inf entries; all-non-finite rows
// (all -inf / NaN) are uniform.
__device__ __forceinline__
float fused_softmax_prob(const float* row, int n_routes, int r, float row_max, int n_pos_inf)
{
    if (n_pos_inf > 0)
        return (isinf(row[r]) && row[r] > 0.0f) ? (1.0f / (float)n_pos_inf) : 0.0f;
    if (!isfinite(row_max))
        return 1.0f / (float)n_routes;
    return expf(row[r] - row_max);
}

__device__ __forceinline__
void fused_softmax_stats(const float* row, int n_routes, float& row_max, int& n_pos_inf)
{
    row_max = -INFINITY;
    n_pos_inf = 0;
    for (int r = 0; r < n_routes; ++r) {
        float x = row[r];
        if (isinf(x) && x > 0.0f)
            ++n_pos_inf;
        row_max = fmaxf(row_max, x);
    }
}

// ════════════════════════════════════════════════════════════════════
//  routing_softmax
//
//  Unfused baseline helper. One thread per node:
//    probs[i, r] = softmax(logits[i, *])[r]
//  When scores_are_logits == 0 the input is treated as already-normalized
//  probabilities (clamped to >= 0, not re-softmaxed).
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void routing_softmax(
    const float* __restrict__ scores,
    float* __restrict__ probs,
    int n_nodes,
    int n_routes,
    int scores_are_logits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes || n_routes <= 0) return;

    const float* row = scores + (long)tid * n_routes;
    float* out = probs + (long)tid * n_routes;

    if (!scores_are_logits) {
        for (int r = 0; r < n_routes; ++r)
            out[r] = fmaxf(row[r], 0.0f);
        return;
    }

    float row_max;
    int n_pos_inf;
    fused_softmax_stats(row, n_routes, row_max, n_pos_inf);

    if (n_pos_inf > 0 || !isfinite(row_max)) {
        for (int r = 0; r < n_routes; ++r)
            out[r] = fused_softmax_prob(row, n_routes, r, row_max, n_pos_inf);
        return;
    }

    float sum = 0.0f;
    for (int r = 0; r < n_routes; ++r)
        sum += expf(row[r] - row_max);
    float inv = 1.0f / fmaxf(sum, SHIP_EPS);

    for (int r = 0; r < n_routes; ++r)
        out[r] = expf(row[r] - row_max) * inv;
}

// ════════════════════════════════════════════════════════════════════
//  saaq_select_fused
//
//  Single-block fused SAAQ argmax. Grid-strides over n_neurons and
//  writes best_walker_out[0] without intermediate partial buffers.
//
//  Launch: <<<1, 256>>>. Occupancy is intentionally one block — this
//  path wins launch overhead, not SM occupancy (see docs/).
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(256)
void saaq_select_fused(
    const float* __restrict__ membrane,
    const float* __restrict__ adaptation,
    unsigned int* __restrict__ best_walker_out,
    int n_neurons,
    float adaptation_scale)
{
    float my_score = SAAQ_SENTINEL;
    int my_walker = INT_MAX;

    for (int tid = (int)threadIdx.x; tid < n_neurons; tid += (int)blockDim.x) {
        float score = saaq_finite_score(membrane[tid], adaptation[tid], adaptation_scale);
        if (fused_better(score, tid, my_score, my_walker)) {
            my_score = score;
            my_walker = tid;
        }
    }

    fused_argmax_reduce(my_score, my_walker);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_scores[32];
    __shared__ int s_walkers[32];

    if (lane == 0) {
        s_scores[warp_id] = my_score;
        s_walkers[warp_id] = my_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bscore = (threadIdx.x < n_warps) ? s_scores[lane] : SAAQ_SENTINEL;
        int bwalker = (threadIdx.x < n_warps) ? s_walkers[lane] : INT_MAX;
        fused_argmax_reduce(bscore, bwalker);
        if (threadIdx.x == 0)
            best_walker_out[0] = (n_neurons > 0) ? (unsigned int)bwalker : 0u;
    }
}

// ════════════════════════════════════════════════════════════════════
//  routing_saaq_fused_pass1
//
//  One thread per routing node. In a single pass over the score row:
//    1. stable softmax (or clamp if scores_are_logits == 0)
//    2. Shannon entropy H = -sum p log2(p)
//    3. top-k route indices (by probability)
//    4. SAAQ score from membrane/adaptation
//
//  Does **not** write the full probability matrix. That re-read/write
//  is the VRAM bottleneck of the unfused pipeline.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
void routing_saaq_fused_pass1(
    const float* __restrict__ scores,
    const float* __restrict__ membrane,
    const float* __restrict__ adaptation,
    int* __restrict__ top_k_indices,
    float* __restrict__ entropy_partial_sum,
    float* __restrict__ entropy_partial_max,
    float* __restrict__ saaq_partial_scores,
    unsigned int* __restrict__ saaq_partial_walkers,
    int n_nodes,
    int n_routes,
    int top_k,
    int scores_are_logits,
    float adaptation_scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_k = top_k;
    if (actual_k > MAX_FUSED_TOP_K) actual_k = MAX_FUSED_TOP_K;
    if (actual_k > n_routes) actual_k = n_routes;
    if (actual_k < 0) actual_k = 0;

    float entropy = 0.0f;
    float saaq_score = SAAQ_SENTINEL;
    int saaq_walker = INT_MAX;

    // SAAQ is independent of routing width: zero routes still argmax membrane.
    if (tid < n_nodes) {
        saaq_score = saaq_finite_score(membrane[tid], adaptation[tid], adaptation_scale);
        saaq_walker = tid;
    }

    if (tid < n_nodes && n_routes > 0) {
        const float* row = scores + (long)tid * n_routes;
        float local_scores[MAX_FUSED_TOP_K];
        int local_indices[MAX_FUSED_TOP_K];
#pragma unroll
        for (int i = 0; i < MAX_FUSED_TOP_K; ++i) {
            local_scores[i] = SAAQ_SENTINEL;
            local_indices[i] = INT_MAX;
        }

        if (scores_are_logits) {
            float row_max;
            int n_pos_inf;
            fused_softmax_stats(row, n_routes, row_max, n_pos_inf);
            float inv = 1.0f;
            if (n_pos_inf == 0 && isfinite(row_max)) {
                float sum = 0.0f;
                for (int r = 0; r < n_routes; ++r)
                    sum += expf(row[r] - row_max);
                inv = 1.0f / fmaxf(sum, SHIP_EPS);
            }

            for (int r = 0; r < n_routes; ++r) {
                float p = fused_softmax_prob(row, n_routes, r, row_max, n_pos_inf);
                if (n_pos_inf == 0 && isfinite(row_max))
                    p *= inv;
                p = fmaxf(p, 0.0f);
                if (p > SHIP_EPS)
                    entropy = fmaf(-p, log2f(p), entropy);
                fused_topk_insert(p, r, local_scores, local_indices, actual_k);
            }
        } else {
            for (int r = 0; r < n_routes; ++r) {
                float p = fmaxf(row[r], 0.0f);
                if (p > SHIP_EPS)
                    entropy = fmaf(-p, log2f(p), entropy);
                fused_topk_insert(p, r, local_scores, local_indices, actual_k);
            }
        }

        if (top_k_indices && actual_k > 0) {
            int* out_idx = top_k_indices + (long)tid * top_k;
            for (int t = 0; t < actual_k; ++t)
                out_idx[t] = (local_scores[t] > SAAQ_SENTINEL) ? local_indices[t] : -1;
            for (int t = actual_k; t < top_k; ++t)
                out_idx[t] = -1;
        }
    } else if (tid < n_nodes && top_k_indices && top_k > 0) {
        int* out_idx = top_k_indices + (long)tid * top_k;
        for (int t = 0; t < top_k; ++t)
            out_idx[t] = -1;
    }

    float sum_v = warp_reduce_sum(entropy);
    float max_v = warp_reduce_max(entropy);
    fused_argmax_reduce(saaq_score, saaq_walker);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];
    __shared__ float s_saaq[32];
    __shared__ int s_walk[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
        s_saaq[warp_id] = saaq_score;
        s_walk[warp_id] = saaq_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bsum = (threadIdx.x < n_warps) ? s_sum[lane] : 0.0f;
        float bmax = (threadIdx.x < n_warps) ? s_max[lane] : 0.0f;
        float bsaaq = (threadIdx.x < n_warps) ? s_saaq[lane] : SAAQ_SENTINEL;
        int bwalk = (threadIdx.x < n_warps) ? s_walk[lane] : INT_MAX;
        bsum = warp_reduce_sum(bsum);
        bmax = warp_reduce_max(bmax);
        fused_argmax_reduce(bsaaq, bwalk);

        if (threadIdx.x == 0) {
            entropy_partial_sum[blockIdx.x] = bsum;
            entropy_partial_max[blockIdx.x] = bmax;
            saaq_partial_scores[blockIdx.x] = bsaaq;
            saaq_partial_walkers[blockIdx.x] = (unsigned int)bwalk;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  fused_telemetry_reduce_pass2
//
//  Final on-device reduction of fused pass1 partials:
//    entropy_sum_out[0]  = sum(partial entropy)
//    entropy_max_out[0]  = max(partial entropy)
//    best_walker_out[0]  = argmax of SAAQ (score, walker) pairs
//
//  Telemetry never leaves the device until this 12-byte write.
//  Launch: <<<1, 256>>>.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(256)
void fused_telemetry_reduce_pass2(
    const float* __restrict__ entropy_partial_sum,
    const float* __restrict__ entropy_partial_max,
    const float* __restrict__ saaq_partial_scores,
    const unsigned int* __restrict__ saaq_partial_walkers,
    float* __restrict__ entropy_sum_out,
    float* __restrict__ entropy_max_out,
    unsigned int* __restrict__ best_walker_out,
    int n_partials)
{
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_max = 0.0f;
    float local_saaq = SAAQ_SENTINEL;
    int local_walker = INT_MAX;

    for (int i = tid; i < n_partials; i += blockDim.x) {
        local_sum += entropy_partial_sum[i];
        local_max = fmaxf(local_max, entropy_partial_max[i]);
        float sc = saaq_partial_scores[i];
        int w = (int)saaq_partial_walkers[i];
        if (fused_better(sc, w, local_saaq, local_walker)) {
            local_saaq = sc;
            local_walker = w;
        }
    }

    float sum_v = warp_reduce_sum(local_sum);
    float max_v = warp_reduce_max(local_max);
    fused_argmax_reduce(local_saaq, local_walker);

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];
    __shared__ float s_saaq[32];
    __shared__ int s_walk[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
        s_saaq[warp_id] = local_saaq;
        s_walk[warp_id] = local_walker;
    }
    __syncthreads();

    if (warp_id == 0) {
        float bsum = (tid < n_warps) ? s_sum[lane] : 0.0f;
        float bmax = (tid < n_warps) ? s_max[lane] : 0.0f;
        float bsaaq = (tid < n_warps) ? s_saaq[lane] : SAAQ_SENTINEL;
        int bwalk = (tid < n_warps) ? s_walk[lane] : INT_MAX;
        bsum = warp_reduce_sum(bsum);
        bmax = warp_reduce_max(bmax);
        fused_argmax_reduce(bsaaq, bwalk);
        if (tid == 0) {
            entropy_sum_out[0] = bsum;
            entropy_max_out[0] = bmax;
            best_walker_out[0] = (n_partials > 0) ? (unsigned int)bwalk : 0u;
        }
    }
}
