// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Routing reductions and adjusted-score selection: shared contract, CPU
//! reference, and analytical traffic model.
//!
//! The device path lives in `cu/routing_selection.cu` and is launched through
//! `GpuAccelerator`. Everything in this module is CPU-safe, so the contract is
//! checked without a GPU. `docs/ROUTING.md` documents the operation, launch
//! constraints, and measurements.
//!
//! An operation takes a row-major `n_nodes × n_routes` score matrix plus
//! per-node `state` and `adaptation`, and produces:
//!
//! - per-node top-k route indices, and optionally their probabilities;
//! - the sum and maximum of per-node Shannon entropy, in bits;
//! - the node with the highest `state - adaptation_scale * adaptation`.
//!
//! # Numerical contract
//!
//! | Case | Behavior |
//! |------|----------|
//! | NaN logit | Ranks as `-inf`; probability 0 unless the row has no larger value |
//! | `+inf` logits | Share the whole row's mass equally |
//! | Row of only `-inf`/NaN logits | Uniform probabilities |
//! | Non-finite or negative value in a probability row | Treated as 0 |
//! | Equal rank values | Lower route index first |
//! | `top_k > n_routes` | Slots past `n_routes` are `-1` (probability 0) |
//! | NaN or `-inf` adjusted score | Node is ineligible |
//! | Equal adjusted scores | Lower node index wins |
//! | No eligible node, or `n_nodes == 0` | No selection; score is `-inf` |
//! | `n_routes == 0` | Entropy 0, top-k all `-1`; selection still runs |
//! | Dimension above `i32::MAX`, `top_k` outside `1..=8`, non-finite scale | [`GpuError::InvalidInput`] |
//!
//! Probability rows are not renormalized. Top-k ranks input values, so route
//! indices do not depend on `exp` rounding or underflow.
//!
//! # Device agreement
//!
//! Indices match the device exactly unless a value is subnormal: the device
//! build (`--use_fast_math`) flushes subnormals to zero and this reference
//! keeps them, as in [`crate::oracle`]. Entropy and probabilities agree within
//! [`ENTROPY_ABS_TOL`] and [`PROBABILITY_ABS_TOL`] because the device uses its
//! fast `exp`/`log2` and a different summation order.

use crate::error::{GpuError, GpuResult};
use crate::gpu::GpuBuffer;

/// Maximum `top_k` accepted by the routing kernels (a fixed per-thread array).
pub const MAX_ROUTING_TOP_K: usize = 8;

/// Threads per block in routing launches; also the per-partial node count.
pub const ROUTING_BLOCK_SIZE: usize = 256;

/// Value written to the selected-index buffer when nothing is eligible.
pub const NO_SELECTION: u32 = u32::MAX;

/// Absolute device/reference agreement bound for entropy, per node, in bits.
pub const ENTROPY_ABS_TOL: f32 = 2.0e-3;

/// Absolute device/reference agreement bound for one probability.
pub const PROBABILITY_ABS_TOL: f32 = 1.0e-4;

/// Device `SHIP_EPS`: probability rows ignore values at or below this.
const ENTROPY_MIN_PROBABILITY: f32 = 1.0e-8;

/// Interpretation of the input score matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingScoreKind {
    /// Unnormalized logits; each row is softmaxed.
    Logits,
    /// Values already scaled by the caller; not renormalized.
    Probabilities,
}

impl RoutingScoreKind {
    /// Kernel `score_kind` argument.
    pub fn as_kernel_arg(self) -> i32 {
        match self {
            Self::Probabilities => 0,
            Self::Logits => 1,
        }
    }
}

/// Dimensions and dynamics for one routing operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingParams {
    pub n_nodes: usize,
    pub n_routes: usize,
    pub top_k: usize,
    pub score_kind: RoutingScoreKind,
    /// Weight of `adaptation` in `state - adaptation_scale * adaptation`.
    pub adaptation_scale: f32,
}

impl RoutingParams {
    /// Validate dimensions and dynamics; shared by every backend.
    pub fn validate(&self) -> GpuResult<()> {
        kernel_dimension("n_nodes", self.n_nodes)?;
        kernel_dimension("n_routes", self.n_routes)?;
        validate_top_k(self.top_k)?;
        validate_scale(self.adaptation_scale)?;
        self.score_elements()?;
        self.top_k_elements()?;
        Ok(())
    }

    /// Required length of the score matrix.
    pub fn score_elements(&self) -> GpuResult<usize> {
        matrix_elements("score matrix", self.n_nodes, self.n_routes)
    }

    /// Required length of each per-node top-k output.
    pub fn top_k_elements(&self) -> GpuResult<usize> {
        matrix_elements("top-k output", self.n_nodes, self.top_k)
    }
}

/// Convert a dimension to the kernel's `int`, or reject it.
pub(crate) fn kernel_dimension(name: &str, value: usize) -> GpuResult<i32> {
    i32::try_from(value).map_err(|_| {
        GpuError::invalid_input(format!("{name} {value} exceeds i32::MAX kernel limit"))
    })
}

pub(crate) fn validate_top_k(top_k: usize) -> GpuResult<()> {
    if (1..=MAX_ROUTING_TOP_K).contains(&top_k) {
        Ok(())
    } else {
        Err(GpuError::invalid_input(format!(
            "top_k must be in 1..={MAX_ROUTING_TOP_K}, got {top_k}"
        )))
    }
}

pub(crate) fn validate_scale(scale: f32) -> GpuResult<()> {
    if scale.is_finite() {
        Ok(())
    } else {
        Err(GpuError::invalid_input(format!(
            "adaptation_scale must be finite, got {scale}"
        )))
    }
}

pub(crate) fn matrix_elements(name: &str, rows: usize, columns: usize) -> GpuResult<usize> {
    rows.checked_mul(columns).ok_or_else(|| {
        GpuError::invalid_input(format!("{name} size {rows} x {columns} overflows usize"))
    })
}

pub(crate) fn expect_min_len(name: &str, actual: usize, minimum: usize) -> GpuResult<()> {
    if actual < minimum {
        return Err(GpuError::invalid_input(format!(
            "{name} too small: need at least {minimum} elements, got {actual}"
        )));
    }
    Ok(())
}

/// Validated node/route dimensions for a routing matrix.
#[derive(Clone, Copy)]
pub(crate) struct RoutingShape {
    nodes: i32,
    routes: i32,
    n_nodes: usize,
    n_routes: usize,
}

impl RoutingShape {
    pub(crate) fn new(n_nodes: usize, n_routes: usize) -> GpuResult<Self> {
        Ok(Self {
            nodes: kernel_dimension("n_nodes", n_nodes)?,
            routes: kernel_dimension("n_routes", n_routes)?,
            n_nodes,
            n_routes,
        })
    }

    fn elements(&self, name: &str) -> GpuResult<usize> {
        matrix_elements(name, self.n_nodes, self.n_routes)
    }
}

/// Validated kernel dimensions and score-matrix element count.
pub(crate) type RoutingMatrix = (i32, i32, usize);

fn routing_matrix(name: &str, shape: RoutingShape) -> GpuResult<RoutingMatrix> {
    Ok((shape.nodes, shape.routes, shape.elements(name)?))
}

pub(crate) fn validate_softmax_buffers(
    scores: &GpuBuffer<f32>,
    probabilities: &GpuBuffer<f32>,
    shape: RoutingShape,
) -> GpuResult<RoutingMatrix> {
    let matrix = routing_matrix("score matrix", shape)?;
    expect_min_len("scores", scores.len(), matrix.2)?;
    expect_min_len("probabilities", probabilities.len(), matrix.2)?;
    Ok(matrix)
}

pub(crate) fn validate_top_k_buffers(
    scores: &GpuBuffer<f32>,
    top_k_indices: &GpuBuffer<i32>,
    shape: RoutingShape,
    top_k: usize,
) -> GpuResult<RoutingMatrix> {
    let matrix = routing_matrix("score matrix", shape)?;
    validate_top_k(top_k)?;
    let top_k_elements = matrix_elements("top-k output", shape.n_nodes, top_k)?;
    expect_min_len("scores", scores.len(), matrix.2)?;
    expect_min_len("top_k_indices", top_k_indices.len(), top_k_elements)?;
    Ok(matrix)
}

pub(crate) fn validate_entropy_buffers(
    probabilities: &GpuBuffer<f32>,
    entropy_sum: &GpuBuffer<f32>,
    entropy_max: &GpuBuffer<f32>,
    shape: RoutingShape,
) -> GpuResult<RoutingMatrix> {
    let matrix = routing_matrix("probability matrix", shape)?;
    expect_min_len("probabilities", probabilities.len(), matrix.2)?;
    expect_min_len("entropy_sum", entropy_sum.len(), 1)?;
    expect_min_len("entropy_max", entropy_max.len(), 1)?;
    Ok(matrix)
}

/// Borrowed fused-routing inputs checked together before launch.
pub(crate) struct FusedRoutingInputs<'a> {
    pub(crate) scores: &'a GpuBuffer<f32>,
    pub(crate) state: &'a GpuBuffer<f32>,
    pub(crate) adaptation: &'a GpuBuffer<f32>,
}

/// Check every buffer of a fused routing launch against `params`.
pub(crate) fn validate_fused_buffers(
    inputs: &FusedRoutingInputs<'_>,
    outputs: &FusedRoutingOutputs,
    params: &RoutingParams,
) -> GpuResult<()> {
    params.validate()?;
    let top_k_elements = params.top_k_elements()?;
    expect_min_len("scores", inputs.scores.len(), params.score_elements()?)?;
    expect_min_len("state", inputs.state.len(), params.n_nodes)?;
    expect_min_len("adaptation", inputs.adaptation.len(), params.n_nodes)?;
    expect_min_len("top_k_indices", outputs.top_k_indices.len(), top_k_elements)?;
    if let Some(probabilities) = &outputs.top_k_probabilities {
        expect_min_len("top_k_probabilities", probabilities.len(), top_k_elements)?;
    }
    expect_min_len("entropy_sum", outputs.entropy_sum.len(), 1)?;
    expect_min_len("entropy_max", outputs.entropy_max.len(), 1)?;
    expect_min_len("selected_index", outputs.selected_index.len(), 1)?;
    expect_min_len("selected_score", outputs.selected_score.len(), 1)?;
    Ok(())
}

/// Borrowed adjusted-selection buffers checked before launch.
pub(crate) struct SelectionInputs<'a> {
    pub(crate) state: &'a GpuBuffer<f32>,
    pub(crate) adaptation: &'a GpuBuffer<f32>,
    pub(crate) selected_index: &'a GpuBuffer<u32>,
    pub(crate) selected_score: &'a GpuBuffer<f32>,
}

/// Validate adjusted-score selection buffers and return the kernel `n`.
pub(crate) fn validate_selection_buffers(
    inputs: &SelectionInputs<'_>,
    scale: f32,
) -> GpuResult<i32> {
    let state = inputs.state;
    let adaptation = inputs.adaptation;
    let selected_index = inputs.selected_index;
    let selected_score = inputs.selected_score;
    validate_scale(scale)?;
    if state.len() != adaptation.len() {
        return Err(GpuError::invalid_input(format!(
            "state length {} does not match adaptation length {}",
            state.len(),
            adaptation.len()
        )));
    }
    expect_min_len("selected_index", selected_index.len(), 1)?;
    expect_min_len("selected_score", selected_score.len(), 1)?;
    kernel_dimension("state length", state.len())
}

/// Result of adjusted-score selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjustedSelection {
    /// Winning node, or `None` when no node was eligible.
    pub index: Option<u32>,
    /// Winning score; `-inf` when `index` is `None`.
    pub score: f32,
}

impl AdjustedSelection {
    /// Build from the raw device outputs.
    pub fn from_raw(index: u32, score: f32) -> Self {
        Self {
            index: (index != NO_SELECTION).then_some(index),
            score,
        }
    }
}

/// Scalar results of a routing operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingSummary {
    /// Sum of per-node entropy, in bits.
    pub entropy_sum: f32,
    /// Largest per-node entropy, in bits; never below 0.
    pub entropy_max: f32,
    pub selection: AdjustedSelection,
}

/// Device output buffers for `GpuAccelerator::fused_routing_select`.
pub struct FusedRoutingOutputs {
    /// `n_nodes × top_k` route indices.
    pub top_k_indices: GpuBuffer<i32>,
    /// Optional `n_nodes × top_k` probabilities; omitting it skips those writes.
    pub top_k_probabilities: Option<GpuBuffer<f32>>,
    pub entropy_sum: GpuBuffer<f32>,
    pub entropy_max: GpuBuffer<f32>,
    pub selected_index: GpuBuffer<u32>,
    pub selected_score: GpuBuffer<f32>,
}

impl FusedRoutingOutputs {
    /// Allocate outputs sized for `params`.
    pub fn alloc(params: &RoutingParams, with_probabilities: bool) -> GpuResult<Self> {
        params.validate()?;
        let top_k_elements = params.top_k_elements()?;
        Ok(Self {
            top_k_indices: GpuBuffer::alloc(top_k_elements)?,
            top_k_probabilities: if with_probabilities {
                Some(GpuBuffer::alloc(top_k_elements)?)
            } else {
                None
            },
            entropy_sum: GpuBuffer::alloc(1)?,
            entropy_max: GpuBuffer::alloc(1)?,
            selected_index: GpuBuffer::alloc(1)?,
            selected_score: GpuBuffer::alloc(1)?,
        })
    }

    /// Copy the four scalar results to the host.
    ///
    /// After `fused_routing_select_async`, synchronize the accelerator first.
    pub fn read_summary(&self) -> GpuResult<RoutingSummary> {
        Ok(RoutingSummary {
            entropy_sum: first(self.entropy_sum.to_vec()?, "entropy_sum")?,
            entropy_max: first(self.entropy_max.to_vec()?, "entropy_max")?,
            selection: AdjustedSelection::from_raw(
                first(self.selected_index.to_vec()?, "selected_index")?,
                first(self.selected_score.to_vec()?, "selected_score")?,
            ),
        })
    }
}

fn first<T: Copy>(values: Vec<T>, name: &str) -> GpuResult<T> {
    values
        .first()
        .copied()
        .ok_or_else(|| GpuError::invalid_input(format!("{name} buffer is empty")))
}

// ── CPU reference ─────────────────────────────────────────────────────────────

/// CPU results for one routing operation.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingReference {
    /// `n_nodes × top_k` route indices.
    pub top_k_indices: Vec<i32>,
    /// `n_nodes × top_k` probabilities.
    pub top_k_probabilities: Vec<f32>,
    pub summary: RoutingSummary,
}

fn sanitized_logit(x: f32) -> f32 {
    if x.is_nan() { f32::NEG_INFINITY } else { x }
}

fn sanitized_probability(x: f32) -> f32 {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

fn sanitized_score(x: f32, kind: RoutingScoreKind) -> f32 {
    match kind {
        RoutingScoreKind::Logits => sanitized_logit(x),
        RoutingScoreKind::Probabilities => sanitized_probability(x),
    }
}

/// Top-k indices for one row, padded with `-1`.
pub fn top_k_row(scores: &[f32], top_k: usize, kind: RoutingScoreKind) -> Vec<i32> {
    let mut ranked: Vec<(f32, usize)> = scores
        .iter()
        .enumerate()
        .map(|(index, &value)| (sanitized_score(value, kind), index))
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut out = vec![-1; top_k];
    for (slot, &(_, index)) in out.iter_mut().zip(ranked.iter()) {
        *slot = index as i32;
    }
    out
}

/// Normalized probabilities for one row.
pub fn softmax_row(scores: &[f32], kind: RoutingScoreKind) -> Vec<f32> {
    match kind {
        RoutingScoreKind::Probabilities => {
            scores.iter().map(|&x| sanitized_probability(x)).collect()
        }
        RoutingScoreKind::Logits => {
            let shape = LogitRowShape::of(scores);
            scores
                .iter()
                .map(|&x| shape.probability(sanitized_logit(x)))
                .collect()
        }
    }
}

struct LogitRowShape {
    max_logit: f32,
    exp_sum: f32,
    weighted: f32,
    n_pos_inf: usize,
    n_routes: usize,
}

impl LogitRowShape {
    fn of(scores: &[f32]) -> Self {
        let mut shape = Self {
            max_logit: f32::NEG_INFINITY,
            exp_sum: 1.0,
            weighted: 0.0,
            n_pos_inf: 0,
            n_routes: scores.len(),
        };
        shape.scan_logits(scores);
        shape.accumulate_finite_row(scores);
        shape
    }

    fn scan_logits(&mut self, scores: &[f32]) {
        for &x in scores {
            let x = sanitized_logit(x);
            if x == f32::INFINITY {
                self.n_pos_inf += 1;
            }
            self.max_logit = self.max_logit.max(x);
        }
    }

    fn accumulate_finite_row(&mut self, scores: &[f32]) {
        if !self.is_finite_row() {
            return;
        }
        let mut exp_sum = 0.0f32;
        let mut weighted = 0.0f32;
        for &x in scores {
            let d = sanitized_logit(x) - self.max_logit;
            let e = d.exp();
            if e > 0.0 {
                exp_sum += e;
                weighted = e.mul_add(d, weighted);
            }
        }
        self.exp_sum = exp_sum;
        self.weighted = weighted;
    }

    fn is_finite_row(&self) -> bool {
        self.n_pos_inf == 0 && self.max_logit > f32::NEG_INFINITY
    }

    fn probability(&self, logit: f32) -> f32 {
        if self.n_pos_inf > 0 {
            if logit == f32::INFINITY {
                1.0 / self.n_pos_inf as f32
            } else {
                0.0
            }
        } else if !self.is_finite_row() {
            1.0 / self.n_routes as f32
        } else {
            (logit - self.max_logit).exp() / self.exp_sum
        }
    }

    fn entropy_bits(&self) -> f32 {
        if self.n_pos_inf > 0 {
            (self.n_pos_inf as f32).log2()
        } else if !self.is_finite_row() {
            (self.n_routes as f32).log2()
        } else {
            let bits =
                self.exp_sum.log2() - (self.weighted / self.exp_sum) * std::f32::consts::LOG2_E;
            bits.max(0.0)
        }
    }
}

/// Shannon entropy of one row, in bits.
///
/// Empty rows have entropy 0. A probability row ignores values at or below
/// `1e-8`, like `routing_entropy_reduce_pass1`.
pub fn entropy_row(scores: &[f32], kind: RoutingScoreKind) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    match kind {
        RoutingScoreKind::Logits => LogitRowShape::of(scores).entropy_bits(),
        RoutingScoreKind::Probabilities => {
            let mut entropy = 0.0f32;
            for &x in scores {
                let p = sanitized_probability(x);
                if p > ENTROPY_MIN_PROBABILITY {
                    entropy = (-p).mul_add(p.log2(), entropy);
                }
            }
            entropy
        }
    }
}

/// `state - scale * adaptation`, single-rounded; NaN becomes `-inf`.
pub fn adjusted_score(state: f32, adaptation: f32, scale: f32) -> f32 {
    let score = (-scale).mul_add(adaptation, state);
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// CPU reference for the adjusted-score argmax kernels.
pub fn adjusted_argmax(
    state: &[f32],
    adaptation: &[f32],
    scale: f32,
) -> GpuResult<AdjustedSelection> {
    validate_scale(scale)?;
    if state.len() != adaptation.len() {
        return Err(GpuError::invalid_input(format!(
            "state length {} does not match adaptation length {}",
            state.len(),
            adaptation.len()
        )));
    }
    kernel_dimension("state length", state.len())?;
    let mut best = AdjustedSelection {
        index: None,
        score: f32::NEG_INFINITY,
    };
    for (index, (&s, &a)) in state.iter().zip(adaptation).enumerate() {
        let score = adjusted_score(s, a, scale);
        if score > best.score {
            best = AdjustedSelection {
                index: Some(index as u32),
                score,
            };
        }
    }
    Ok(best)
}

/// CPU reference for `GpuAccelerator::fused_routing_select`, and the documented
/// fallback when no GPU is available.
pub fn routing_reference(
    scores: &[f32],
    state: &[f32],
    adaptation: &[f32],
    params: &RoutingParams,
) -> GpuResult<RoutingReference> {
    params.validate()?;
    let score_elements = params.score_elements()?;
    if scores.len() != score_elements {
        return Err(GpuError::invalid_input(format!(
            "score matrix length {} does not match {} x {}",
            scores.len(),
            params.n_nodes,
            params.n_routes
        )));
    }
    if state.len() != params.n_nodes {
        return Err(GpuError::invalid_input(format!(
            "state length {} does not match n_nodes {}",
            state.len(),
            params.n_nodes
        )));
    }
    let selection = adjusted_argmax(state, adaptation, params.adaptation_scale)?;

    let top_k_elements = params.top_k_elements()?;
    let mut top_k_indices = Vec::with_capacity(top_k_elements);
    let mut top_k_probabilities = Vec::with_capacity(top_k_elements);
    let mut entropy_sum = 0.0f32;
    let mut entropy_max = 0.0f32;
    for node in 0..params.n_nodes {
        let row = &scores[node * params.n_routes..(node + 1) * params.n_routes];
        let entropy = entropy_row(row, params.score_kind);
        entropy_sum += entropy;
        entropy_max = entropy_max.max(entropy);

        let probabilities = softmax_row(row, params.score_kind);
        for index in top_k_row(row, params.top_k, params.score_kind) {
            top_k_indices.push(index);
            top_k_probabilities.push(if index >= 0 {
                probabilities[index as usize]
            } else {
                0.0
            });
        }
    }

    Ok(RoutingReference {
        top_k_indices,
        top_k_probabilities,
        summary: RoutingSummary {
            entropy_sum,
            entropy_max,
            selection,
        },
    })
}

// ── Analytical traffic model ──────────────────────────────────────────────────

/// Analytical global-memory bytes requested by kernel code.
///
/// These are **estimates, not measurements**. They count every kernel load and
/// store of a buffer, including repeated reads of a row. Caches absorb part of
/// this before DRAM. Input upload and output download are excluded.
pub mod analytical {
    use super::ROUTING_BLOCK_SIZE;

    /// Requested bytes and launches for one execution path.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RequestedTraffic {
        pub bytes_read: u64,
        pub bytes_written: u64,
        pub launches: u32,
    }

    impl RequestedTraffic {
        pub fn bytes_total(&self) -> u64 {
            self.bytes_read.saturating_add(self.bytes_written)
        }
    }

    fn sizes(n_nodes: usize, n_routes: usize, top_k: usize) -> (u64, u64, u64, u64) {
        let nodes = n_nodes as u64;
        let matrix = nodes.saturating_mul(n_routes as u64).saturating_mul(4);
        let per_node_state = nodes.saturating_mul(8);
        let top_k_bytes = nodes.saturating_mul(top_k as u64).saturating_mul(4);
        let blocks = n_nodes.div_ceil(ROUTING_BLOCK_SIZE) as u64;
        (matrix, per_node_state, top_k_bytes, blocks)
    }

    /// `routing_softmax` (three logit-row reads), `routing_top_k`,
    /// `routing_entropy_reduce_pass1`, `latent_reduce_pass2`, and the two
    /// adjusted-argmax passes.
    pub fn composable(n_nodes: usize, n_routes: usize, top_k: usize) -> RequestedTraffic {
        if n_nodes == 0 {
            return RequestedTraffic {
                bytes_read: 0,
                bytes_written: 0,
                launches: 0,
            };
        }
        let (matrix, state, top_k_bytes, blocks) = sizes(n_nodes, n_routes, top_k);
        let partials = blocks.saturating_mul(16);
        RequestedTraffic {
            bytes_read: matrix
                .saturating_mul(5)
                .saturating_add(state)
                .saturating_add(partials),
            bytes_written: matrix
                .saturating_add(top_k_bytes)
                .saturating_add(partials)
                .saturating_add(16),
            launches: 6,
        }
    }

    /// `fused_routing_pass1` (two logit-row reads) and `fused_routing_pass2`,
    /// writing top-k indices but not top-k probabilities.
    pub fn fused(n_nodes: usize, n_routes: usize, top_k: usize) -> RequestedTraffic {
        if n_nodes == 0 {
            return RequestedTraffic {
                bytes_read: 0,
                bytes_written: 0,
                launches: 0,
            };
        }
        let (matrix, state, top_k_bytes, blocks) = sizes(n_nodes, n_routes, top_k);
        let partials = blocks.saturating_mul(16);
        RequestedTraffic {
            bytes_read: matrix
                .saturating_mul(2)
                .saturating_add(state)
                .saturating_add(partials),
            bytes_written: top_k_bytes.saturating_add(partials).saturating_add(16),
            launches: 2,
        }
    }
}

#[cfg(test)]
mod tests;
