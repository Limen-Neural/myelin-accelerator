// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  routing_selection.cu — routing reductions and adjusted-score selection
//
//  Kernels exported (name-exact for PTX symbol lookup in kernel.rs):
//    routing_softmax             — scores → row-normalized probabilities
//    routing_top_k               — per-row top-k indices
//    adjusted_argmax_pass1       — per-block (score, index) winners
//    indexed_argmax_reduce_pass2 — global (score, index) winner
//    adjusted_argmax_single_block — one-block argmax without partials
//    fused_routing_pass1         — softmax + entropy + top-k + argmax per block
//    fused_routing_pass2         — global entropy sum/max and argmax
//
//  Numerical contract (mirrored by src/routing.rs, documented in
//  docs/ROUTING.md):
//    rank value  logits: NaN → -inf. probabilities: non-finite/negative → 0.
//    ordering    higher rank value first; equal values keep the lower index.
//    selection   score = state - scale * adaptation, single-rounded FMA.
//                NaN and -inf scores are ineligible; when nothing is eligible
//                the index is NO_SELECTION and the score is -inf.
//
//  Target: sm_120 (RTX 5080 Blackwell) · CUDA 13.x
// ════════════════════════════════════════════════════════════════════

#include "common.cuh"

#define ROUTING_BLOCK_SIZE 256
#define MAX_ROUTING_TOP_K 8
#define NO_SELECTION 0xffffffffu
#define NO_INDEX INT_MAX

// log2(e), converts a natural-log expression to bits.
#define LOG2_E 1.44269504088896340736f

enum RoutingScoreKind {
    ROUTING_SCORES_ARE_PROBABILITIES = 0,
    ROUTING_SCORES_ARE_LOGITS = 1,
};

// ── Ordering ───────────────────────────────────────────────────────

__device__ __forceinline__
bool ranks_before(float lhs_value, int lhs_index, float rhs_value, int rhs_index)
{
    return lhs_value > rhs_value ||
           (lhs_value == rhs_value && lhs_index < rhs_index);
}

__device__ __forceinline__
float sanitized_logit(float x)
{
    return isnan(x) ? -INFINITY : x;
}

__device__ __forceinline__
float sanitized_score(float x, int score_kind)
{
    return score_kind == ROUTING_SCORES_ARE_LOGITS
        ? sanitized_logit(x)
        : sanitized_probability(x);
}

// Single-rounded so the host reference (`f32::mul_add`) ranks the same values.
__device__ __forceinline__
float adjusted_score(float state, float adaptation, float scale)
{
    float score = fmaf(-scale, adaptation, state);
    return isnan(score) ? -INFINITY : score;
}

__device__ __forceinline__
int clamped_top_k(int top_k, int n_routes)
{
    int k = top_k;
    if (k > MAX_ROUTING_TOP_K) k = MAX_ROUTING_TOP_K;
    if (k > n_routes) k = n_routes;
    return k < 0 ? 0 : k;
}

// ── Top-k insertion, bounded by MAX_ROUTING_TOP_K ──────────────────

__device__ __forceinline__
void top_k_init(float (&values)[MAX_ROUTING_TOP_K], int (&indices)[MAX_ROUTING_TOP_K])
{
#pragma unroll
    for (int i = 0; i < MAX_ROUTING_TOP_K; ++i) {
        values[i] = -INFINITY;
        indices[i] = NO_INDEX;
    }
}

__device__ __forceinline__
void top_k_insert(
    float value,
    int index,
    float (&values)[MAX_ROUTING_TOP_K],
    int (&indices)[MAX_ROUTING_TOP_K],
    int k)
{
    if (k <= 0) return;

    int tail = k - 1;
    if (!ranks_before(value, index, values[tail], indices[tail])) return;

    values[tail] = value;
    indices[tail] = index;

    for (int pos = tail; pos > 0; --pos) {
        if (!ranks_before(values[pos], indices[pos], values[pos - 1], indices[pos - 1]))
            break;
        float swap_value = values[pos - 1];
        int swap_index = indices[pos - 1];
        values[pos - 1] = values[pos];
        indices[pos - 1] = indices[pos];
        values[pos] = swap_value;
        indices[pos] = swap_index;
    }
}

// ── Row summaries ──────────────────────────────────────────────────

struct LogitRow {
    float max_logit;    // largest sanitized logit
    float exp_sum;      // Σ exp(x - max_logit); valid when finite_row
    int n_pos_inf;      // number of +inf logits
    int n_routes;
    bool finite_row;    // no +inf logits and at least one logit above -inf
};

__device__ __forceinline__
float logit_row_probability(const LogitRow& row, float logit)
{
    if (row.n_pos_inf > 0)
        return logit == INFINITY ? 1.0f / (float)row.n_pos_inf : 0.0f;
    if (!row.finite_row)
        return 1.0f / (float)row.n_routes;
    return expf(logit - row.max_logit) / row.exp_sum;
}

// Accumulates exp_sum and returns entropy in bits:
//   H = log2(S) - (1/S) Σ exp(d) d log2(e),   d = x - max_logit
// Row special cases have a closed form and need no second read of the row.
__device__ __forceinline__
float logit_row_finish(const float* row_scores, LogitRow& row)
{
    if (row.n_pos_inf > 0)
        return log2f((float)row.n_pos_inf);
    if (!row.finite_row)
        return log2f((float)row.n_routes);

    float exp_sum = 0.0f;
    float weighted = 0.0f;
    for (int r = 0; r < row.n_routes; ++r) {
        float d = sanitized_logit(row_scores[r]) - row.max_logit;
        float e = expf(d);
        if (e > 0.0f) {
            exp_sum += e;
            weighted = fmaf(e, d, weighted);
        }
    }
    row.exp_sum = exp_sum;
    return fmaxf(log2f(exp_sum) - (weighted / exp_sum) * LOG2_E, 0.0f);
}

// ── Warp / block reductions ────────────────────────────────────────

__device__ __forceinline__
void argmax_warp_reduce(float& score, int& index)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_score = __shfl_down_sync(0xffffffffu, score, offset);
        int other_index = __shfl_down_sync(0xffffffffu, index, offset);
        if (ranks_before(other_score, other_index, score, index)) {
            score = other_score;
            index = other_index;
        }
    }
}

__device__ __forceinline__
unsigned int selected_index_or_none(float score, int index)
{
    return score > -INFINITY ? (unsigned int)index : NO_SELECTION;
}

// ════════════════════════════════════════════════════════════════════
//  routing_softmax
//
//  Composable path. One thread per node. Reads each logit row three times
//  (max, normalizer, output); probability rows are sanitized and copied.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void routing_softmax(
    const float* __restrict__ scores,
    float* __restrict__ probabilities,
    int n_nodes,
    int n_routes,
    int score_kind)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes || n_routes <= 0) return;

    const float* row_scores = scores + (long)tid * n_routes;
    float* out = probabilities + (long)tid * n_routes;

    if (score_kind != ROUTING_SCORES_ARE_LOGITS) {
        for (int r = 0; r < n_routes; ++r)
            out[r] = sanitized_probability(row_scores[r]);
        return;
    }

    LogitRow row;
    row.max_logit = -INFINITY;
    row.n_pos_inf = 0;
    row.n_routes = n_routes;
    row.exp_sum = 1.0f;
    for (int r = 0; r < n_routes; ++r) {
        float x = sanitized_logit(row_scores[r]);
        if (x == INFINITY) ++row.n_pos_inf;
        row.max_logit = fmaxf(row.max_logit, x);
    }
    row.finite_row = row.n_pos_inf == 0 && row.max_logit > -INFINITY;

    if (row.finite_row) {
        float exp_sum = 0.0f;
        for (int r = 0; r < n_routes; ++r)
            exp_sum += expf(sanitized_logit(row_scores[r]) - row.max_logit);
        row.exp_sum = exp_sum;
    }

    for (int r = 0; r < n_routes; ++r)
        out[r] = logit_row_probability(row, sanitized_logit(row_scores[r]));
}

// ════════════════════════════════════════════════════════════════════
//  routing_top_k
//
//  Composable path. Ranks the input scores directly: softmax preserves
//  logit order, and ranking before exp() cannot create underflow ties.
//  Slots past min(top_k, n_routes) are written as -1.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void routing_top_k(
    const float* __restrict__ scores,
    int* __restrict__ top_k_indices,
    int n_nodes,
    int n_routes,
    int top_k,
    int score_kind)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes || top_k <= 0) return;

    int k = clamped_top_k(top_k, n_routes);
    float values[MAX_ROUTING_TOP_K];
    int indices[MAX_ROUTING_TOP_K];
    top_k_init(values, indices);

    const float* row_scores = scores + (long)tid * n_routes;
    for (int r = 0; r < n_routes; ++r)
        top_k_insert(sanitized_score(row_scores[r], score_kind), r, values, indices, k);

    int* out = top_k_indices + (long)tid * top_k;
    for (int t = 0; t < top_k; ++t)
        out[t] = t < k ? indices[t] : -1;
}

// ════════════════════════════════════════════════════════════════════
//  adjusted_argmax_pass1
//
//  score[i] = state[i] - scale * adaptation[i]
//  Writes one (score, index) winner per block.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void adjusted_argmax_pass1(
    const float* __restrict__ state,
    const float* __restrict__ adaptation,
    float* __restrict__ partial_scores,
    unsigned int* __restrict__ partial_indices,
    int n,
    float scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float score = -INFINITY;
    int index = NO_INDEX;

    if (tid < n) {
        score = adjusted_score(state[tid], adaptation[tid], scale);
        index = tid;
    }

    argmax_warp_reduce(score, index);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_scores[32];
    __shared__ int s_indices[32];

    if (lane == 0) {
        s_scores[warp_id] = score;
        s_indices[warp_id] = index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_score = (threadIdx.x < n_warps) ? s_scores[lane] : -INFINITY;
        int block_index = (threadIdx.x < n_warps) ? s_indices[lane] : NO_INDEX;
        argmax_warp_reduce(block_score, block_index);

        if (threadIdx.x == 0) {
            partial_scores[blockIdx.x] = block_score;
            partial_indices[blockIdx.x] = (unsigned int)block_index;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  indexed_argmax_reduce_pass2
//
//  Reduces per-block winners to one (score, index). Launch <<<1, 256>>>;
//  threads stride over n_partials, so any partial count reduces correctly.
//  n_partials == 0 writes NO_SELECTION / -inf.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void indexed_argmax_reduce_pass2(
    const float* __restrict__ partial_scores,
    const unsigned int* __restrict__ partial_indices,
    unsigned int* __restrict__ selected_index,
    float* __restrict__ selected_score,
    int n_partials)
{
    int tid = threadIdx.x;
    float score = -INFINITY;
    int index = NO_INDEX;

    for (int i = tid; i < n_partials; i += (int)blockDim.x) {
        float candidate = partial_scores[i];
        int candidate_index = (int)partial_indices[i];
        if (ranks_before(candidate, candidate_index, score, index)) {
            score = candidate;
            index = candidate_index;
        }
    }

    argmax_warp_reduce(score, index);

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_scores[32];
    __shared__ int s_indices[32];

    if (lane == 0) {
        s_scores[warp_id] = score;
        s_indices[warp_id] = index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_score = (tid < n_warps) ? s_scores[lane] : -INFINITY;
        int block_index = (tid < n_warps) ? s_indices[lane] : NO_INDEX;
        argmax_warp_reduce(block_score, block_index);

        if (tid == 0) {
            selected_index[0] = selected_index_or_none(block_score, block_index);
            selected_score[0] = block_score;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  adjusted_argmax_single_block
//
//  One-launch argmax without partial buffers. Launch <<<1, 256>>>; threads
//  stride over n. Occupies one block regardless of n; see docs/ROUTING.md for
//  the measured size where the two-pass path becomes faster.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void adjusted_argmax_single_block(
    const float* __restrict__ state,
    const float* __restrict__ adaptation,
    unsigned int* __restrict__ selected_index,
    float* __restrict__ selected_score,
    int n,
    float scale)
{
    int tid = threadIdx.x;
    float score = -INFINITY;
    int index = NO_INDEX;

    // 64-bit stride: n can be within one block of INT_MAX.
    for (long i = tid; i < n; i += blockDim.x) {
        float candidate = adjusted_score(state[i], adaptation[i], scale);
        if (ranks_before(candidate, (int)i, score, index)) {
            score = candidate;
            index = (int)i;
        }
    }

    argmax_warp_reduce(score, index);

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_scores[32];
    __shared__ int s_indices[32];

    if (lane == 0) {
        s_scores[warp_id] = score;
        s_indices[warp_id] = index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_score = (tid < n_warps) ? s_scores[lane] : -INFINITY;
        int block_index = (tid < n_warps) ? s_indices[lane] : NO_INDEX;
        argmax_warp_reduce(block_score, block_index);

        if (tid == 0) {
            selected_index[0] = selected_index_or_none(block_score, block_index);
            selected_score[0] = block_score;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  fused_routing_pass1
//
//  One thread per node. Top-k ranks during the first read of a score row;
//  logit rows are read once more for the normalizer and entropy. The
//  probability matrix is never written. `top_k_probabilities` may be null.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void fused_routing_pass1(
    const float* __restrict__ scores,
    const float* __restrict__ state,
    const float* __restrict__ adaptation,
    int* __restrict__ top_k_indices,
    float* __restrict__ top_k_probabilities,
    float* __restrict__ partial_entropy_sum,
    float* __restrict__ partial_entropy_max,
    float* __restrict__ partial_scores,
    unsigned int* __restrict__ partial_indices,
    int n_nodes,
    int n_routes,
    int top_k,
    int score_kind,
    float scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float entropy = 0.0f;
    float score = -INFINITY;
    int index = NO_INDEX;

    if (tid < n_nodes) {
        score = adjusted_score(state[tid], adaptation[tid], scale);
        index = tid;

        const float* row_scores = scores + (long)tid * n_routes;
        int k = clamped_top_k(top_k, n_routes);
        float values[MAX_ROUTING_TOP_K];
        int indices[MAX_ROUTING_TOP_K];
        top_k_init(values, indices);

        if (n_routes > 0 && score_kind == ROUTING_SCORES_ARE_LOGITS) {
            LogitRow row;
            row.max_logit = -INFINITY;
            row.n_pos_inf = 0;
            row.n_routes = n_routes;
            row.exp_sum = 1.0f;
            for (int r = 0; r < n_routes; ++r) {
                float x = sanitized_logit(row_scores[r]);
                if (x == INFINITY) ++row.n_pos_inf;
                row.max_logit = fmaxf(row.max_logit, x);
                top_k_insert(x, r, values, indices, k);
            }
            row.finite_row = row.n_pos_inf == 0 && row.max_logit > -INFINITY;
            entropy = logit_row_finish(row_scores, row);

            if (top_k_probabilities) {
                float* out = top_k_probabilities + (long)tid * top_k;
                for (int t = 0; t < top_k; ++t)
                    out[t] = t < k ? logit_row_probability(row, values[t]) : 0.0f;
            }
        } else {
            for (int r = 0; r < n_routes; ++r) {
                float p = sanitized_probability(row_scores[r]);
                if (p > SHIP_EPS)
                    entropy = fmaf(-p, log2f(p), entropy);
                top_k_insert(p, r, values, indices, k);
            }

            if (top_k_probabilities) {
                float* out = top_k_probabilities + (long)tid * top_k;
                for (int t = 0; t < top_k; ++t)
                    out[t] = t < k ? values[t] : 0.0f;
            }
        }

        int* out_indices = top_k_indices + (long)tid * top_k;
        for (int t = 0; t < top_k; ++t)
            out_indices[t] = t < k ? indices[t] : -1;
    }

    float sum_v = warp_reduce_sum(entropy);
    float max_v = warp_reduce_max(entropy);
    argmax_warp_reduce(score, index);

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];
    __shared__ float s_scores[32];
    __shared__ int s_indices[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
        s_scores[warp_id] = score;
        s_indices[warp_id] = index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (threadIdx.x < n_warps) ? s_sum[lane] : 0.0f;
        float block_max = (threadIdx.x < n_warps) ? s_max[lane] : 0.0f;
        float block_score = (threadIdx.x < n_warps) ? s_scores[lane] : -INFINITY;
        int block_index = (threadIdx.x < n_warps) ? s_indices[lane] : NO_INDEX;
        block_sum = warp_reduce_sum(block_sum);
        block_max = warp_reduce_max(block_max);
        argmax_warp_reduce(block_score, block_index);

        if (threadIdx.x == 0) {
            partial_entropy_sum[blockIdx.x] = block_sum;
            partial_entropy_max[blockIdx.x] = block_max;
            partial_scores[blockIdx.x] = block_score;
            partial_indices[blockIdx.x] = (unsigned int)block_index;
        }
    }
}

// ════════════════════════════════════════════════════════════════════
//  fused_routing_pass2
//
//  Final reduction of fused_routing_pass1 partials into four scalars.
//  Launch <<<1, 256>>>. n_partials == 0 writes 0, 0, NO_SELECTION, -inf.
// ════════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(ROUTING_BLOCK_SIZE)
void fused_routing_pass2(
    const float* __restrict__ partial_entropy_sum,
    const float* __restrict__ partial_entropy_max,
    const float* __restrict__ partial_scores,
    const unsigned int* __restrict__ partial_indices,
    float* __restrict__ entropy_sum,
    float* __restrict__ entropy_max,
    unsigned int* __restrict__ selected_index,
    float* __restrict__ selected_score,
    int n_partials)
{
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_max = 0.0f;
    float score = -INFINITY;
    int index = NO_INDEX;

    for (int i = tid; i < n_partials; i += (int)blockDim.x) {
        local_sum += partial_entropy_sum[i];
        local_max = fmaxf(local_max, partial_entropy_max[i]);
        float candidate = partial_scores[i];
        int candidate_index = (int)partial_indices[i];
        if (ranks_before(candidate, candidate_index, score, index)) {
            score = candidate;
            index = candidate_index;
        }
    }

    float sum_v = warp_reduce_sum(local_sum);
    float max_v = warp_reduce_max(local_max);
    argmax_warp_reduce(score, index);

    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float s_sum[32];
    __shared__ float s_max[32];
    __shared__ float s_scores[32];
    __shared__ int s_indices[32];

    if (lane == 0) {
        s_sum[warp_id] = sum_v;
        s_max[warp_id] = max_v;
        s_scores[warp_id] = score;
        s_indices[warp_id] = index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (tid < n_warps) ? s_sum[lane] : 0.0f;
        float block_max = (tid < n_warps) ? s_max[lane] : 0.0f;
        float block_score = (tid < n_warps) ? s_scores[lane] : -INFINITY;
        int block_index = (tid < n_warps) ? s_indices[lane] : NO_INDEX;
        block_sum = warp_reduce_sum(block_sum);
        block_max = warp_reduce_max(block_max);
        argmax_warp_reduce(block_score, block_index);

        if (tid == 0) {
            entropy_sum[0] = block_sum;
            entropy_max[0] = block_max;
            selected_index[0] = selected_index_or_none(block_score, block_index);
            selected_score[0] = block_score;
        }
    }
}
