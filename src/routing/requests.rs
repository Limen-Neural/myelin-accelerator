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

/// Inputs for `routing_top_k`.
pub struct RoutingTopKRequest<'a> {
    pub scores: &'a GpuBuffer<f32>,
    pub top_k_indices: &'a mut GpuBuffer<i32>,
    pub n_nodes: usize,
    pub n_routes: usize,
    pub top_k: usize,
    pub score_kind: RoutingScoreKind,
}

/// Inputs for `routing_entropy_reduce`.
pub struct RoutingEntropyRequest<'a> {
    pub probabilities: &'a GpuBuffer<f32>,
    pub entropy_sum: &'a mut GpuBuffer<f32>,
    pub entropy_max: &'a mut GpuBuffer<f32>,
    pub n_nodes: usize,
    pub n_routes: usize,
}

/// Inputs for adjusted-score selection.
pub struct AdjustedSelectionRequest<'a> {
    pub state: &'a GpuBuffer<f32>,
    pub adaptation: &'a GpuBuffer<f32>,
    pub selected_index: &'a mut GpuBuffer<u32>,
    pub selected_score: &'a mut GpuBuffer<f32>,
    pub scale: f32,
}

/// Inputs for `fused_routing_select`.
pub struct FusedRoutingRequest<'a> {
    pub scores: &'a GpuBuffer<f32>,
    pub state: &'a GpuBuffer<f32>,
    pub adaptation: &'a GpuBuffer<f32>,
    pub outputs: &'a mut FusedRoutingOutputs,
    pub params: &'a RoutingParams,
}
