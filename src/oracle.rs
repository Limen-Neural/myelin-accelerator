// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Independent scalar CPU oracles for the public CUDA kernel paths.
//!
//! These implementations favour numerical transparency over speed. They do
//! **not** reuse device algorithms (warps, shared-memory tiles, two-pass
//! reductions, or FMA accumulation order). Differential tests compare
//! device output against these oracles; mismatches report the first index
//! plus the input seed and shape so a failing case can be replayed.
//!
//! NaN is never treated as equal to anything, including another NaN.
//! Length mismatches are reported explicitly (no silent `zip` truncation).
//!
//! ## Selected kernels
//!
//! | Kernel | Oracle | Tolerance |
//! |--------|--------|-----------|
//! | `poisson_encode` | [`poisson_encode_oracle`] | exact `u32` |
//! | `cosine_similarity_batched` | [`cosine_similarity_batched_oracle`] | abs `1e-5`, rel `1e-5` |
//! | `satsolver_extract` | [`satsolver_extract_oracle`] | exact `u8` |
//! | `satsolver_aux_reduce_best` | [`satsolver_aux_reduce_best_oracle`] | exact flags / `i32` |
//! | `ternary_gemv` | [`ternary_gemv_oracle`] | abs `1e-4`, rel `1e-5` |
//! | `ternary_gemm` | [`ternary_gemm_oracle`] | abs `1e-4`, rel `1e-5` |
//! | `lif_step` | [`lif_step_oracle`] | exact membrane bits / `u32` state |
//! | `lif_step_weighted` | [`lif_step_weighted_oracle`] | exact membrane bits / `u32` state |
//!
//! `cosine_similarity_batched` is loaded in `KernelModule` but has no
//! `GpuAccelerator` launch wrapper yet; CPU tests still run in CI.
//! `lif_step` / `lif_step_weighted` are likewise raw registered symbols with no
//! `GpuAccelerator` wrapper in v0.2.0; their GPU differential tests launch the
//! raw symbols (`tests/snn_fixtures_gpu.rs`, see `docs/SNN_COMPATIBILITY.md`).
//! The remaining rows have public wrappers and GPU differential tests
//! (capability gated with `#[ignore]`).

#![allow(clippy::needless_range_loop)]

use std::fmt::Debug;

/// Device `SHIP_EPS` from `cu/common.cuh` (cosine denominator floor).
pub const SHIP_EPS: f32 = 1.0e-8;

/// Absolute tolerance for ternary GEMV / GEMM vs device FMA order.
pub const TERNARY_ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for ternary GEMV / GEMM vs device FMA order.
pub const TERNARY_REL_TOL: f32 = 1.0e-5;
/// Absolute tolerance for cosine similarity vs device reductions.
pub const COSINE_ABS_TOL: f32 = 1.0e-5;
/// Relative tolerance for cosine similarity vs device reductions.
pub const COSINE_REL_TOL: f32 = 1.0e-5;

const TERNARY_VALUES_PER_WORD: usize = 16;

// ── Numerical Recipes LCG (scalar; matches `lcg_next` / `lcg_float`) ────────

/// Advance a 32-bit LCG: `state * 1664525 + 1013904223` (mod 2³²).
///
/// Same multipliers as `cu/common.cuh`, written independently as wrapping
/// integer arithmetic (no CUDA headers).
#[must_use]
pub fn lcg_next(state: u32) -> u32 {
    state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// Map an already-advanced LCG state to `f32` in `[0, 1)`.
///
/// Uses the high 24 bits after a right shift by 8, scaled by `2⁻²⁴`.
#[must_use]
pub fn lcg_unit(state: u32) -> f32 {
    (state >> 8) as f32 * (1.0 / 16_777_216.0)
}

// ── Encoding ────────────────────────────────────────────────────────────────

/// Poisson spike encoding: clamp rate to `[0, 1]`, then `U[0,1) < rate`.
///
/// RNG stream per channel `i`: `s1 = lcg_next(seed XOR i)`, `s2 = lcg_next(s1)`,
/// `r = lcg_unit(s2)`. Output is `1` iff `r < threshold`.
#[must_use]
pub fn poisson_encode_oracle(stimuli: &[f32], seed: u32) -> Vec<u32> {
    let n = stimuli.len();
    let mut spikes = vec![0u32; n];
    for i in 0..n {
        let mut rng = lcg_next(seed ^ (i as u32));
        // Match CUDA `fminf`/`fmaxf`: NaN is ignored, so a NaN rate becomes 1.0.
        // Rust `clamp` leaves NaN and `r < NaN` is always false.
        #[allow(clippy::manual_clamp)]
        let threshold = stimuli[i].min(1.0).max(0.0);
        rng = lcg_next(rng);
        let r = lcg_unit(rng);
        spikes[i] = u32::from(r < threshold);
    }
    spikes
}

// ── Routing ─────────────────────────────────────────────────────────────────

/// Batched cosine similarity: `out[q,k] = dot / (|q| |k| + eps)`.
///
/// Scalar loops; `eps` is [`SHIP_EPS`]. Empty `dim` yields `0 / eps`.
#[must_use]
pub fn cosine_similarity_batched_oracle(
    queries: &[f32],
    keys: &[f32],
    n_queries: usize,
    n_keys: usize,
    dim: usize,
) -> Vec<f32> {
    assert!(
        queries.len() >= n_queries.saturating_mul(dim),
        "queries shorter than n_queries×dim"
    );
    assert!(
        keys.len() >= n_keys.saturating_mul(dim),
        "keys shorter than n_keys×dim"
    );
    let mut out = vec![0.0f32; n_queries.saturating_mul(n_keys)];
    for q in 0..n_queries {
        for k in 0..n_keys {
            let mut dot = 0.0f32;
            let mut norm_q = 0.0f32;
            let mut norm_k = 0.0f32;
            for d in 0..dim {
                let qi = queries[q * dim + d];
                let ki = keys[k * dim + d];
                dot += qi * ki;
                norm_q += qi * qi;
                norm_k += ki * ki;
            }
            let denom = norm_q.sqrt() * norm_k.sqrt() + SHIP_EPS;
            out[q * n_keys + k] = dot / denom;
        }
    }
    out
}

// ── SAT extract / reduce ────────────────────────────────────────────────────

/// Copy walker `best_walker` into an `n_vars` assignment; out-of-range
/// walker indices fall back to walker `0` (same as the device kernel).
#[must_use]
pub fn satsolver_extract_oracle(
    assignment: &[u8],
    best_walker: i32,
    n_vars: usize,
    n_walkers: usize,
) -> Vec<u8> {
    if n_vars == 0 {
        return Vec::new();
    }
    assert!(
        n_walkers > 0,
        "satsolver_extract_oracle: n_walkers must be > 0 when n_vars > 0"
    );
    let need = n_walkers.saturating_mul(n_vars);
    assert!(
        assignment.len() >= need,
        "assignment shorter than n_walkers×n_vars"
    );
    let bw = if best_walker < 0 || (best_walker as usize) >= n_walkers {
        0
    } else {
        best_walker as usize
    };
    assignment[bw * n_vars..bw * n_vars + n_vars].to_vec()
}

/// Result of [`satsolver_aux_reduce_best_oracle`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuxReduceBest {
    pub sat_flags: Vec<u8>,
    pub best_score: i32,
    pub best_walker: i32,
}

/// Recompute per-clause sat flags from assignments and reduce `scores` to
/// the minimum `(score, walker)` pair (lower walker index wins ties).
///
/// Scores are **not** recomputed from the formula — they are an input, matching
/// `satsolver_aux_update`.
#[must_use]
pub fn satsolver_aux_reduce_best_oracle(
    assignment: &[u8],
    scores: &[i32],
    clauses: &[i32],
    n_walkers: usize,
    n_vars: usize,
    n_clauses: usize,
    clause_len: usize,
) -> AuxReduceBest {
    assert!(n_walkers > 0, "n_walkers must be > 0");
    assert!(
        assignment.len() >= n_walkers.saturating_mul(n_vars),
        "assignment shorter than n_walkers×n_vars"
    );
    assert!(scores.len() >= n_walkers, "scores shorter than n_walkers");
    assert!(
        clauses.len() >= n_clauses.saturating_mul(clause_len),
        "clauses shorter than n_clauses×clause_len"
    );

    let mut sat_flags = vec![0u8; n_walkers.saturating_mul(n_clauses)];
    for w in 0..n_walkers {
        let asgn = &assignment[w * n_vars..w * n_vars + n_vars];
        for c in 0..n_clauses {
            let clause = &clauses[c * clause_len..c * clause_len + clause_len];
            sat_flags[w * n_clauses + c] = eval_clause(asgn, clause);
        }
    }

    let mut best_score = scores[0];
    let mut best_walker = 0i32;
    for w in 1..n_walkers {
        let s = scores[w];
        if s < best_score || (s == best_score && (w as i32) < best_walker) {
            best_score = s;
            best_walker = w as i32;
        }
    }

    AuxReduceBest {
        sat_flags,
        best_score,
        best_walker,
    }
}

fn eval_clause(asgn: &[u8], clause: &[i32]) -> u8 {
    if clause.is_empty() {
        return 0;
    }
    for &lit in clause {
        let encoded = lit as u32;
        let var = (encoded >> 1) as usize;
        if var >= asgn.len() {
            continue;
        }
        let neg = encoded & 1;
        let val = u32::from(asgn[var]) ^ neg;
        if val != 0 {
            return 1;
        }
    }
    0
}

// ── Packed ternary ──────────────────────────────────────────────────────────

fn trit_at(packed: &[u32], row: usize, k: usize, words_per_row: usize) -> i8 {
    let word = packed[row * words_per_row + k / TERNARY_VALUES_PER_WORD];
    let shift = (k % TERNARY_VALUES_PER_WORD) * 2;
    match (word >> shift) & 0b11 {
        0b01 => 1,
        0b10 => -1,
        _ => 0,
    }
}

fn words_per_row(k: usize) -> usize {
    k.div_ceil(TERNARY_VALUES_PER_WORD)
}

fn groups_per_row(k: usize, group_size: usize) -> usize {
    assert!(group_size > 0, "group_size must be > 0");
    k.div_ceil(group_size)
}

/// Scalar group-scaled ternary GEMV: `y[m] = Σ_k trit(W[m,k]) * s[m,g(k)] * x[k]`.
///
/// Trits are decoded one element at a time from packed `u32` words. When
/// `skip_zeros` is true, zero trits are omitted (same finite result).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ternary_gemv_oracle(
    packed: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    group_size: usize,
    skip_zeros: bool,
) -> Vec<f32> {
    let wpr = words_per_row(k);
    let gpr = groups_per_row(k, group_size);
    assert!(
        packed.len() >= m.saturating_mul(wpr),
        "packed too short for m×k"
    );
    assert!(
        scales.len() >= m.saturating_mul(gpr),
        "scales too short for m×groups"
    );
    assert!(x.len() >= k, "x too short for k");

    let mut y = vec![0.0f32; m];
    if k == 0 {
        return y;
    }
    for row in 0..m {
        let mut acc = 0.0f32;
        for kk in 0..k {
            let w = trit_at(packed, row, kk, wpr);
            if skip_zeros && w == 0 {
                continue;
            }
            let s = scales[row * gpr + kk / group_size];
            acc += (w as f32) * s * x[kk];
        }
        y[row] = acc;
    }
    y
}

/// Scalar group-scaled ternary GEMM: `C[m,n] = Σ_k trit(W[m,k]) * s[m,g(k)] * B[k,n]`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ternary_gemm_oracle(
    packed: &[u32],
    scales: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    skip_zeros: bool,
) -> Vec<f32> {
    let wpr = words_per_row(k);
    let gpr = groups_per_row(k, group_size);
    assert!(
        packed.len() >= m.saturating_mul(wpr),
        "packed too short for m×k"
    );
    assert!(
        scales.len() >= m.saturating_mul(gpr),
        "scales too short for m×groups"
    );
    assert!(b.len() >= k.saturating_mul(n), "b too short for k×n");

    let mut c = vec![0.0f32; m.saturating_mul(n)];
    if k == 0 || m == 0 || n == 0 {
        return c;
    }
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let w = trit_at(packed, row, kk, wpr);
                if skip_zeros && w == 0 {
                    continue;
                }
                let s = scales[row * gpr + kk / group_size];
                acc += (w as f32) * s * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

// ── Fixed v0.2.0 LIF dynamics ───────────────────────────────────────────────
//
// The v0.2.0 `lif_step` / `lif_step_weighted` kernels have *fixed* dynamics
// (no per-neuron or configurable parameters). The contract, per tick and per
// neuron, is:
//
// 1. Refractory branch first: if `refract > 0`, force the membrane to
//    `LIF_RESET`, decrement `refract`, ignore the input, emit no spike.
// 2. Otherwise integrate once with a single-rounded fused multiply-add:
//    `v = fma(LIF_DECAY, v, I)`.
// 3. If `v >= LIF_THRESHOLD`, spike: output `1`, `v = LIF_RESET`,
//    `refract = LIF_REFRACT_TICKS`.
//
// Consequently a neuron that spikes at tick `t` is held at reset for ticks
// `t+1` and `t+2` and integrates again from reset at `t+3`.
//
// Subnormals: these oracles use IEEE `f32::mul_add` and keep subnormal
// values. The shipped PTX is built with `--use_fast_math` and the LIF FMAs
// are `fma.rn.ftz.f32`, which flush subnormal inputs and results to zero.
// Oracle and device therefore agree bit-for-bit only while no subnormal
// membrane, current, weight, or input occurs. The fixture differential tests
// assert that precondition. Mirroring the flush is deliberately left out: it
// is a build-flag property, not part of the LIF contract.

/// v0.2.0 membrane decay multiplier per tick (fixed; not configurable).
pub const LIF_DECAY: f32 = 0.85;
/// v0.2.0 normalised firing threshold; `v >= LIF_THRESHOLD` spikes.
pub const LIF_THRESHOLD: f32 = 1.0;
/// v0.2.0 membrane value after a spike and throughout the refractory period.
pub const LIF_RESET: f32 = 0.0;
/// v0.2.0 absolute refractory period, in ticks, set on each spike.
pub const LIF_REFRACT_TICKS: u32 = 2;

/// One neuron, one tick of the fixed v0.2.0 LIF rule. Returns the spike flag.
fn lif_update(v: &mut f32, refract: &mut u32, current: f32) -> u32 {
    if *refract > 0 {
        *v = LIF_RESET;
        *refract -= 1;
        return 0;
    }
    let next = LIF_DECAY.mul_add(*v, current);
    if next >= LIF_THRESHOLD {
        *v = LIF_RESET;
        *refract = LIF_REFRACT_TICKS;
        1
    } else {
        *v = next;
        0
    }
}

/// Scalar reference for one `lif_step` tick with pre-summed current `i_ext`.
///
/// Updates `membrane` and `refract` in place and returns the per-neuron spike
/// flags (`0` / `1`). All three slices must have the same length
/// (`n_neurons`); mismatches panic with the offending dimensions.
///
/// IEEE semantics: subnormals are kept, whereas the `--use_fast_math` device
/// build flushes them to zero, so bit-exact agreement assumes subnormal-free
/// state and inputs.
#[must_use]
pub fn lif_step_oracle(membrane: &mut [f32], i_ext: &[f32], refract: &mut [u32]) -> Vec<u32> {
    let n = membrane.len();
    assert_eq!(
        i_ext.len(),
        n,
        "lif_step_oracle: i_ext has {} elements but membrane has n_neurons={n}",
        i_ext.len()
    );
    assert_eq!(
        refract.len(),
        n,
        "lif_step_oracle: refract has {} elements but membrane has n_neurons={n}",
        refract.len()
    );
    let mut spikes = vec![0u32; n];
    for i in 0..n {
        spikes[i] = lif_update(&mut membrane[i], &mut refract[i], i_ext[i]);
    }
    spikes
}

/// Scalar reference for one `lif_step_weighted` tick.
///
/// `weights` is row-major `[n_neurons × n_inputs]` (`weights[n * n_inputs + j]`
/// weighs input `j` for neuron `n`); `n_neurons` is `membrane.len()`. For a
/// non-refractory neuron the current is accumulated from `0.0` in input-index
/// order with one fused multiply-add per input, `I = fma(w[n,j], x[j], I)`,
/// then the fixed LIF rule is applied. Refractory neurons skip the dot
/// product entirely. Returns the per-neuron spike flags.
///
/// Panics with the offending dimensions if `weights`, `input_spikes`, or
/// `refract` do not match `n_neurons` / `n_inputs` exactly.
///
/// Like [`lif_step_oracle`], subnormals are kept (no device flush-to-zero),
/// so bit-exact agreement requires subnormal-free weights, inputs,
/// accumulators, and membrane.
#[must_use]
pub fn lif_step_weighted_oracle(
    membrane: &mut [f32],
    weights: &[f32],
    input_spikes: &[f32],
    refract: &mut [u32],
    n_inputs: usize,
) -> Vec<u32> {
    let n = membrane.len();
    assert_eq!(
        input_spikes.len(),
        n_inputs,
        "lif_step_weighted_oracle: input_spikes has {} elements but n_inputs={n_inputs}",
        input_spikes.len()
    );
    assert_eq!(
        refract.len(),
        n,
        "lif_step_weighted_oracle: refract has {} elements but membrane has n_neurons={n}",
        refract.len()
    );
    let expected_weights = n
        .checked_mul(n_inputs)
        .expect("lif_step_weighted_oracle: n_neurons×n_inputs overflows usize");
    assert_eq!(
        weights.len(),
        expected_weights,
        "lif_step_weighted_oracle: weights has {} elements but n_neurons×n_inputs = {n}×{n_inputs} = {expected_weights}",
        weights.len()
    );
    let mut spikes = vec![0u32; n];
    for i in 0..n {
        let current = if refract[i] > 0 {
            // Ignored by `lif_update`; the dot product is skipped like the device.
            0.0
        } else {
            let row = &weights[i * n_inputs..(i + 1) * n_inputs];
            let mut acc = 0.0f32;
            for j in 0..n_inputs {
                acc = row[j].mul_add(input_spikes[j], acc);
            }
            acc
        };
        spikes[i] = lif_update(&mut membrane[i], &mut refract[i], current);
    }
    spikes
}

// ── Comparison helpers ──────────────────────────────────────────────────────

/// Check exact equality; fail on length mismatch or the first differing index.
pub fn check_exact<T: Copy + PartialEq + Debug>(
    got: &[T],
    expected: &[T],
    seed: u64,
    shape: &str,
) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "length mismatch: expected {} got {} (seed={seed}, {shape}) \
             — output was truncated or expanded",
            expected.len(),
            got.len()
        ));
    }
    for i in 0..expected.len() {
        if got[i] != expected[i] {
            return Err(format!(
                "mismatch at index {i}: expected {:?} got {:?} (seed={seed}, {shape})",
                expected[i], got[i]
            ));
        }
    }
    Ok(())
}

/// Finite-aware `f32` compare. NaNs never match. Same-signed infinities and
/// signed zeros (`+0` / `-0`) match via IEEE equality. Otherwise
/// `|a-b| ≤ abs` **or** `|a-b| ≤ rel * max(|a|,|b|)`.
pub fn f32_close(a: f32, b: f32, abs: f32, rel: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs());
    diff <= abs || diff <= rel * scale
}

/// Check `f32` buffers with [`f32_close`].
pub fn check_f32(
    got: &[f32],
    expected: &[f32],
    abs: f32,
    rel: f32,
    seed: u64,
    shape: &str,
) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "length mismatch: expected {} got {} (seed={seed}, {shape}) \
             — output was truncated or expanded",
            expected.len(),
            got.len()
        ));
    }
    for i in 0..expected.len() {
        if !f32_close(got[i], expected[i], abs, rel) {
            return Err(format!(
                "mismatch at index {i}: expected {} got {} \
                 (seed={seed}, {shape}, abs={abs}, rel={rel})",
                expected[i], got[i]
            ));
        }
    }
    Ok(())
}

/// Panic with a replayable mismatch message.
#[track_caller]
pub fn assert_exact<T: Copy + PartialEq + Debug>(
    got: &[T],
    expected: &[T],
    seed: u64,
    shape: &str,
) {
    if let Err(msg) = check_exact(got, expected, seed, shape) {
        panic!("{msg}");
    }
}

/// Panic with a replayable floating-point mismatch message.
#[track_caller]
pub fn assert_f32(got: &[f32], expected: &[f32], abs: f32, rel: f32, seed: u64, shape: &str) {
    if let Err(msg) = check_f32(got, expected, abs, rel, seed, shape) {
        panic!("{msg}");
    }
}

// ── Seeded case generation ──────────────────────────────────────────────────

/// SplitMix64-style generator for deterministic test inputs (not the Poisson LCG).
#[derive(Clone, Debug)]
pub struct CaseRng {
    state: u64,
}

impl CaseRng {
    /// Create a generator from an explicit seed (replayable from mismatch logs).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn mix(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next 32-bit value from the SplitMix64 stream.
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        self.mix() as u32
    }

    /// Finite value in `[-1, 1)` — never NaN or Inf.
    #[must_use]
    pub fn next_unit_f32(&mut self) -> f32 {
        let u = (self.mix() >> 40) as u32;
        let x = (u as f32) / 16_777_216.0;
        x.mul_add(2.0, -1.0)
    }

    /// Finite rate in `[0, 1)`.
    #[must_use]
    pub fn next_rate_f32(&mut self) -> f32 {
        let u = (self.mix() >> 40) as u32;
        (u as f32) / 16_777_216.0
    }

    /// Next trit in `{-1, 0, +1}`.
    #[must_use]
    pub fn next_trit(&mut self) -> i8 {
        match self.mix() % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        }
    }

    /// Next bit as `0` or `1`.
    #[must_use]
    pub fn next_bit(&mut self) -> u8 {
        (self.mix() & 1) as u8
    }
}

/// Pin clamp / signed-zero edges shared by CPU and GPU Poisson suites.
pub fn pin_poisson_boundary_stimuli(stim: &mut [f32]) {
    let n = stim.len();
    if n > 0 {
        stim[0] = 0.0;
    }
    if n > 1 {
        stim[1] = 1.0;
    }
    if n > 2 {
        stim[2] = -0.0;
    }
    if n > 3 {
        stim[3] = 1.5;
    }
}

/// Fill SAT walker scores, forcing an extreme negative at `n_walkers / 2`.
pub fn fill_sat_scores(rng: &mut CaseRng, n_walkers: usize) -> Vec<i32> {
    let mut scores = vec![0i32; n_walkers];
    for (w, score) in scores.iter_mut().enumerate() {
        *score = (rng.next_u32() % 17) as i32;
        if w == n_walkers / 2 {
            *score = i32::MIN / 4;
        }
    }
    scores
}

/// Lengths that sit off warp (32) and launch-block (256) multiples.
pub const BOUNDARY_LENS: &[usize] = &[0, 1, 17, 31, 32, 33, 255, 256, 257];

/// Representative seeds printed in mismatch messages.
pub const CASE_SEEDS: &[u64] = &[1, 2, 7, 42, 99, 1_234_567_890];
