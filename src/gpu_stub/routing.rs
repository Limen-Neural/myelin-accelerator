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
        let shape = RoutingShape::new(request.n_nodes, request.n_routes)?;
        validate_softmax_buffers(request.scores, request.probabilities, shape)?;
        Err(self.unavailable_error())
    }

    pub fn routing_top_k(&self, request: RoutingTopKRequest<'_>) -> GpuResult<()> {
        self.routing_top_k_async(request)
    }

    pub fn routing_top_k_async(&self, request: RoutingTopKRequest<'_>) -> GpuResult<()> {
        let shape = RoutingShape::new(request.n_nodes, request.n_routes)?;
        validate_top_k_buffers(request.scores, request.top_k_indices, shape, request.top_k)?;
        Err(self.unavailable_error())
    }

    pub fn routing_entropy_reduce(&self, request: RoutingEntropyRequest<'_>) -> GpuResult<()> {
        self.routing_entropy_reduce_async(request)
    }

    pub fn routing_entropy_reduce_async(
        &self,
        request: RoutingEntropyRequest<'_>,
    ) -> GpuResult<()> {
        let shape = RoutingShape::new(request.n_nodes, request.n_routes)?;
        validate_entropy_buffers(
            request.probabilities,
            request.entropy_sum,
            request.entropy_max,
            shape,
        )?;
        Err(self.unavailable_error())
    }

    pub fn adjusted_argmax(&self, request: AdjustedSelectionRequest<'_>) -> GpuResult<()> {
        self.adjusted_argmax_async(request)
    }

    pub fn adjusted_argmax_async(&self, request: AdjustedSelectionRequest<'_>) -> GpuResult<()> {
        let inputs = SelectionInputs::new(
            request.state,
            request.adaptation,
            request.selected_index,
            request.selected_score,
        );
        validate_selection_buffers(&inputs, request.scale)?;
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
        let inputs = FusedRoutingInputs::new(request.scores, request.state, request.adaptation);
        validate_fused_buffers(&inputs, request.outputs, request.params)?;
        Err(self.unavailable_error())
    }
}
