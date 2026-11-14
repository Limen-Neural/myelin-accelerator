// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{GpuAccelerator, GpuResult};
use crate::routing::{
    AdjustedSelectionRequest, FusedRoutingInputs, FusedRoutingRequest, RoutingEntropyRequest,
    RoutingShape, RoutingSoftmaxRequest, RoutingTopKRequest, SelectionInputs,
    validate_entropy_buffers, validate_fused_buffers, validate_selection_buffers,
    validate_softmax_buffers, validate_top_k_buffers,
};

impl GpuAccelerator {
    pub fn routing_softmax(&self, request: RoutingSoftmaxRequest<'_>) -> GpuResult<()> {
        self.routing_softmax_async(request)
    }

    pub fn routing_softmax_async(&self, request: RoutingSoftmaxRequest<'_>) -> GpuResult<()> {
        let RoutingSoftmaxRequest {
            scores,
            probabilities,
            n_nodes,
            n_routes,
            score_kind: _,
        } = request;
        let shape = RoutingShape::new(n_nodes, n_routes)?;
        validate_softmax_buffers(scores, probabilities, shape)?;
        Err(self.unavailable_error())
    }

    pub fn routing_top_k(&self, request: RoutingTopKRequest<'_>) -> GpuResult<()> {
        self.routing_top_k_async(request)
    }

    pub fn routing_top_k_async(&self, request: RoutingTopKRequest<'_>) -> GpuResult<()> {
        let RoutingTopKRequest {
            scores,
            top_k_indices,
            n_nodes,
            n_routes,
            top_k,
            score_kind: _,
        } = request;
        let shape = RoutingShape::new(n_nodes, n_routes)?;
        validate_top_k_buffers(scores, top_k_indices, shape, top_k)?;
        Err(self.unavailable_error())
    }

    pub fn routing_entropy_reduce(&self, request: RoutingEntropyRequest<'_>) -> GpuResult<()> {
        self.routing_entropy_reduce_async(request)
    }

    pub fn routing_entropy_reduce_async(
        &self,
        request: RoutingEntropyRequest<'_>,
    ) -> GpuResult<()> {
        let RoutingEntropyRequest {
            probabilities,
            entropy_sum,
            entropy_max,
            n_nodes,
            n_routes,
        } = request;
        let shape = RoutingShape::new(n_nodes, n_routes)?;
        validate_entropy_buffers(probabilities, entropy_sum, entropy_max, shape)?;
        Err(self.unavailable_error())
    }

    pub fn adjusted_argmax(&self, request: AdjustedSelectionRequest<'_>) -> GpuResult<()> {
        self.adjusted_argmax_async(request)
    }

    pub fn adjusted_argmax_async(&self, request: AdjustedSelectionRequest<'_>) -> GpuResult<()> {
        let AdjustedSelectionRequest {
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        } = request;
        let inputs = SelectionInputs::new(state, adaptation, selected_index, selected_score);
        validate_selection_buffers(&inputs, scale)?;
        Err(self.unavailable_error())
    }

    pub fn adjusted_argmax_single_block(
        &self,
        request: AdjustedSelectionRequest<'_>,
    ) -> GpuResult<()> {
        self.adjusted_argmax_single_block_async(request)
    }

    pub fn adjusted_argmax_single_block_async(
        &self,
        request: AdjustedSelectionRequest<'_>,
    ) -> GpuResult<()> {
        self.adjusted_argmax_async(request)
    }

    pub fn fused_routing_select(&self, request: FusedRoutingRequest<'_>) -> GpuResult<()> {
        self.fused_routing_select_async(request)
    }

    pub fn fused_routing_select_async(&self, request: FusedRoutingRequest<'_>) -> GpuResult<()> {
        let FusedRoutingRequest {
            scores,
            state,
            adaptation,
            outputs,
            params,
        } = request;
        let inputs = FusedRoutingInputs::new(scores, state, adaptation);
        validate_fused_buffers(&inputs, outputs, params)?;
        Err(self.unavailable_error())
    }
}
