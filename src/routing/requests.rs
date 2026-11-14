use super::{FusedRoutingOutputs, RoutingParams, RoutingScoreKind};
use crate::gpu::GpuBuffer;

/// Inputs for `routing_softmax`.
pub struct RoutingSoftmaxRequest<'a> {
    pub scores: &'a GpuBuffer<f32>,
    pub probabilities: &'a mut GpuBuffer<f32>,
    pub n_nodes: usize,
    pub n_routes: usize,
    pub score_kind: RoutingScoreKind,
}

impl<'a> RoutingSoftmaxRequest<'a> {
    pub fn new(
        scores: &'a GpuBuffer<f32>,
        probabilities: &'a mut GpuBuffer<f32>,
        n_nodes: usize,
        n_routes: usize,
        score_kind: RoutingScoreKind,
    ) -> Self {
        Self {
            scores,
            probabilities,
            n_nodes,
            n_routes,
            score_kind,
        }
    }
}

/// Inputs for `routing_top_k`.
pub struct RoutingTopKRequest<'a> {
    pub scores: &'a GpuBuffer<f32>,
    pub top_k_indices: &'a mut GpuBuffer<i32>,
    pub n_nodes: usize,
    pub n_routes: usize,
    pub top_k: usize,
    pub score_kind: RoutingScoreKind,
}

impl<'a> RoutingTopKRequest<'a> {
    pub fn new(
        scores: &'a GpuBuffer<f32>,
        top_k_indices: &'a mut GpuBuffer<i32>,
        n_nodes: usize,
        n_routes: usize,
        top_k: usize,
        score_kind: RoutingScoreKind,
    ) -> Self {
        Self {
            scores,
            top_k_indices,
            n_nodes,
            n_routes,
            top_k,
            score_kind,
        }
    }
}

/// Inputs for `routing_entropy_reduce`.
pub struct RoutingEntropyRequest<'a> {
    pub probabilities: &'a GpuBuffer<f32>,
    pub entropy_sum: &'a mut GpuBuffer<f32>,
    pub entropy_max: &'a mut GpuBuffer<f32>,
    pub n_nodes: usize,
    pub n_routes: usize,
}

impl<'a> RoutingEntropyRequest<'a> {
    pub fn new(
        probabilities: &'a GpuBuffer<f32>,
        entropy_sum: &'a mut GpuBuffer<f32>,
        entropy_max: &'a mut GpuBuffer<f32>,
        n_nodes: usize,
        n_routes: usize,
    ) -> Self {
        Self {
            probabilities,
            entropy_sum,
            entropy_max,
            n_nodes,
            n_routes,
        }
    }
}

/// Inputs for adjusted-score selection.
pub struct AdjustedSelectionRequest<'a> {
    pub state: &'a GpuBuffer<f32>,
    pub adaptation: &'a GpuBuffer<f32>,
    pub selected_index: &'a mut GpuBuffer<u32>,
    pub selected_score: &'a mut GpuBuffer<f32>,
    pub scale: f32,
}

impl<'a> AdjustedSelectionRequest<'a> {
    pub fn new(
        state: &'a GpuBuffer<f32>,
        adaptation: &'a GpuBuffer<f32>,
        selected_index: &'a mut GpuBuffer<u32>,
        selected_score: &'a mut GpuBuffer<f32>,
        scale: f32,
    ) -> Self {
        Self {
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        }
    }
}

/// Inputs for `fused_routing_select`.
pub struct FusedRoutingRequest<'a> {
    pub scores: &'a GpuBuffer<f32>,
    pub state: &'a GpuBuffer<f32>,
    pub adaptation: &'a GpuBuffer<f32>,
    pub outputs: &'a mut FusedRoutingOutputs,
    pub params: &'a RoutingParams,
}

impl<'a> FusedRoutingRequest<'a> {
    pub fn new(
        scores: &'a GpuBuffer<f32>,
        state: &'a GpuBuffer<f32>,
        adaptation: &'a GpuBuffer<f32>,
        outputs: &'a mut FusedRoutingOutputs,
        params: &'a RoutingParams,
    ) -> Self {
        Self {
            scores,
            state,
            adaptation,
            outputs,
            params,
        }
    }
}
