// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::gpu::accelerator::GpuAccelerator;
use crate::gpu::error::GpuResult;
use crate::gpu::memory::GpuBuffer;
use crate::routing::{
    FusedRoutingOutputs, ROUTING_BLOCK_SIZE, RoutingParams, RoutingScoreKind,
    validate_entropy_buffers, validate_fused_buffers, validate_softmax_buffers,
    validate_top_k_buffers,
};
use cust::launch;

/// Per-block partial buffers shared by the multi-pass routing launches.
pub(crate) struct RoutingScratch {
    pub(crate) entropy_sum: GpuBuffer<f32>,
    pub(crate) entropy_max: GpuBuffer<f32>,
    pub(crate) scores: GpuBuffer<f32>,
    pub(crate) indices: GpuBuffer<u32>,
}

impl GpuAccelerator {
    /// Row-normalize a score matrix into `probabilities` (composable path).
    pub fn routing_softmax(
        &self,
        scores: &GpuBuffer<f32>,
        probabilities: &mut GpuBuffer<f32>,
        n_nodes: usize,
        n_routes: usize,
        score_kind: RoutingScoreKind,
    ) -> GpuResult<()> {
        let launched =
            self.routing_softmax_async(scores, probabilities, n_nodes, n_routes, score_kind);
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Enqueue [`Self::routing_softmax`]; synchronize before using the buffers.
    pub fn routing_softmax_async(
        &self,
        scores: &GpuBuffer<f32>,
        probabilities: &mut GpuBuffer<f32>,
        n_nodes: usize,
        n_routes: usize,
        score_kind: RoutingScoreKind,
    ) -> GpuResult<()> {
        let (nodes, routes, elements) =
            validate_softmax_buffers(scores, probabilities, n_nodes, n_routes)?;
        let kernels = self.kernels()?;
        if elements == 0 {
            return Ok(());
        }

        let func = kernels.get_function("routing_softmax")?;
        let stream = self.ready_stream()?;
        let grid = Self::routing_grid(n_nodes);
        // SAFETY: the entry is `(const float*, float*, int, int, int)`. Both
        // buffers hold at least n_nodes * n_routes elements, the kernel guards
        // tid >= n_nodes, and the buffers outlive the stream work per the
        // synchronize contract.
        unsafe {
            launch!(func<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scores.as_device_ptr(),
                probabilities.as_device_ptr(),
                nodes,
                routes,
                score_kind.as_kernel_arg(),
            ))
        }
        .map_err(|e| Self::launch_error("routing_softmax", e))
    }

    /// Per-row top-k route indices (composable path), padded with `-1`.
    pub fn routing_top_k(
        &self,
        scores: &GpuBuffer<f32>,
        top_k_indices: &mut GpuBuffer<i32>,
        n_nodes: usize,
        n_routes: usize,
        top_k: usize,
        score_kind: RoutingScoreKind,
    ) -> GpuResult<()> {
        let launched =
            self.routing_top_k_async(scores, top_k_indices, n_nodes, n_routes, top_k, score_kind);
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Enqueue [`Self::routing_top_k`]; synchronize before using the buffers.
    pub fn routing_top_k_async(
        &self,
        scores: &GpuBuffer<f32>,
        top_k_indices: &mut GpuBuffer<i32>,
        n_nodes: usize,
        n_routes: usize,
        top_k: usize,
        score_kind: RoutingScoreKind,
    ) -> GpuResult<()> {
        let (nodes, routes, _) =
            validate_top_k_buffers(scores, top_k_indices, n_nodes, n_routes, top_k)?;
        let kernels = self.kernels()?;
        if n_nodes == 0 {
            return Ok(());
        }

        let func = kernels.get_function("routing_top_k")?;
        let stream = self.ready_stream()?;
        let grid = Self::routing_grid(n_nodes);
        // SAFETY: the entry is `(const float*, int*, int, int, int, int)`.
        // Buffer lengths were checked above, top_k is at most
        // MAX_ROUTING_TOP_K, and the kernel guards tid >= n_nodes. The buffers
        // outlive the stream work per the synchronize contract.
        unsafe {
            launch!(func<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scores.as_device_ptr(),
                top_k_indices.as_device_ptr(),
                nodes,
                routes,
                top_k as i32,
                score_kind.as_kernel_arg(),
            ))
        }
        .map_err(|e| Self::launch_error("routing_top_k", e))
    }

    /// Sum and maximum of per-node entropy from a probability matrix
    /// (composable path).
    pub fn routing_entropy_reduce(
        &self,
        probabilities: &GpuBuffer<f32>,
        entropy_sum: &mut GpuBuffer<f32>,
        entropy_max: &mut GpuBuffer<f32>,
        n_nodes: usize,
        n_routes: usize,
    ) -> GpuResult<()> {
        let launched = self.routing_entropy_reduce_async(
            probabilities,
            entropy_sum,
            entropy_max,
            n_nodes,
            n_routes,
        );
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Enqueue [`Self::routing_entropy_reduce`]; synchronize before using the
    /// buffers. An error after the first launch can leave that launch queued.
    pub fn routing_entropy_reduce_async(
        &self,
        probabilities: &GpuBuffer<f32>,
        entropy_sum: &mut GpuBuffer<f32>,
        entropy_max: &mut GpuBuffer<f32>,
        n_nodes: usize,
        n_routes: usize,
    ) -> GpuResult<()> {
        let (nodes, routes, _) =
            validate_entropy_buffers(probabilities, entropy_sum, entropy_max, n_nodes, n_routes)?;
        let kernels = self.kernels()?;
        if n_nodes == 0 {
            self.zero_output_prefix(entropy_sum, 1)?;
            return self.zero_output_prefix(entropy_max, 1);
        }

        let pass1 = kernels.get_function("routing_entropy_reduce_pass1")?;
        let pass2 = kernels.get_function("latent_reduce_pass2")?;
        let grid = Self::routing_grid(n_nodes);
        self.ensure_routing_scratch(grid as usize)?;
        let scratch = self.routing_scratch.borrow();
        let scratch = scratch.as_ref().ok_or_else(|| self.unavailable_error())?;
        let stream = self.ready_stream()?;
        // SAFETY: pass 1 is `(const float*, float*, float*, int, int)` and
        // writes grid partials; scratch holds at least grid elements. Pass 2 is
        // `(const float*, const float*, float*, float*, int)` and writes one
        // scalar to each output. Both launches share this stream, so pass 2
        // reads completed partials. Scratch lives on `self`.
        unsafe {
            launch!(pass1<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                probabilities.as_device_ptr(),
                scratch.entropy_sum.as_device_ptr(),
                scratch.entropy_max.as_device_ptr(),
                nodes,
                routes,
            ))
            .map_err(|e| Self::launch_error("routing_entropy_reduce_pass1", e))?;
            launch!(pass2<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scratch.entropy_sum.as_device_ptr(),
                scratch.entropy_max.as_device_ptr(),
                entropy_sum.as_device_ptr(),
                entropy_max.as_device_ptr(),
                grid as i32,
            ))
        }
        .map_err(|e| Self::launch_error("latent_reduce_pass2", e))
    }

    /// Select the node with the highest `state - scale * adaptation`, using a
    /// per-block pass and a reduction pass.
    ///
    /// Writes [`crate::routing::NO_SELECTION`] and `-inf` when no node is
    /// eligible. This is the default selection path; see `docs/ROUTING.md`.
    pub fn adjusted_argmax(
        &self,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        selected_index: &mut GpuBuffer<u32>,
        selected_score: &mut GpuBuffer<f32>,
        scale: f32,
    ) -> GpuResult<()> {
        let launched =
            self.adjusted_argmax_async(state, adaptation, selected_index, selected_score, scale);
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Enqueue [`Self::adjusted_argmax`]; synchronize before using the buffers.
    /// An error after the first launch can leave that launch queued.
    pub fn adjusted_argmax_async(
        &self,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        selected_index: &mut GpuBuffer<u32>,
        selected_score: &mut GpuBuffer<f32>,
        scale: f32,
    ) -> GpuResult<()> {
        let n = crate::routing::validate_selection_buffers(
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        )?;
        let kernels = self.kernels()?;
        let pass1 = kernels.get_function("adjusted_argmax_pass1")?;
        let pass2 = kernels.get_function("indexed_argmax_reduce_pass2")?;
        let grid = Self::routing_grid(n as usize);
        self.ensure_routing_scratch(grid.max(1) as usize)?;
        let scratch = self.routing_scratch.borrow();
        let scratch = scratch.as_ref().ok_or_else(|| self.unavailable_error())?;
        let stream = self.ready_stream()?;
        // SAFETY: pass 1 is `(const float*, const float*, float*, unsigned*,
        // int, float)`; state and adaptation hold n elements, the kernel guards
        // tid >= n, and scratch holds at least grid partials. Pass 2 is
        // `(const float*, const unsigned*, unsigned*, float*, int)` and reads
        // exactly grid partials, so grid == 0 writes the no-selection result
        // without reading scratch. Both launches share this stream.
        unsafe {
            if grid > 0 {
                launch!(pass1<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                    state.as_device_ptr(),
                    adaptation.as_device_ptr(),
                    scratch.scores.as_device_ptr(),
                    scratch.indices.as_device_ptr(),
                    n,
                    scale,
                ))
                .map_err(|e| Self::launch_error("adjusted_argmax_pass1", e))?;
            }
            launch!(pass2<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scratch.scores.as_device_ptr(),
                scratch.indices.as_device_ptr(),
                selected_index.as_device_ptr(),
                selected_score.as_device_ptr(),
                grid as i32,
            ))
        }
        .map_err(|e| Self::launch_error("indexed_argmax_reduce_pass2", e))
    }

    /// One-launch variant of [`Self::adjusted_argmax`] with the same results
    /// and no partial buffers. It occupies a single block at any `n`; prefer it
    /// only at the small sizes identified in `docs/ROUTING.md`.
    pub fn adjusted_argmax_single_block(
        &self,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        selected_index: &mut GpuBuffer<u32>,
        selected_score: &mut GpuBuffer<f32>,
        scale: f32,
    ) -> GpuResult<()> {
        let launched = self.adjusted_argmax_single_block_async(
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        );
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Enqueue [`Self::adjusted_argmax_single_block`]; synchronize before using
    /// the buffers.
    pub fn adjusted_argmax_single_block_async(
        &self,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        selected_index: &mut GpuBuffer<u32>,
        selected_score: &mut GpuBuffer<f32>,
        scale: f32,
    ) -> GpuResult<()> {
        let n = crate::routing::validate_selection_buffers(
            state,
            adaptation,
            selected_index,
            selected_score,
            scale,
        )?;
        let kernels = self.kernels()?;
        let func = kernels.get_function("adjusted_argmax_single_block")?;
        let stream = self.ready_stream()?;
        // SAFETY: the entry is `(const float*, const float*, unsigned*, float*,
        // int, float)`. Threads stride below n, which both inputs hold, and only
        // thread 0 writes one scalar to each output, which were length-checked.
        unsafe {
            launch!(func<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                state.as_device_ptr(),
                adaptation.as_device_ptr(),
                selected_index.as_device_ptr(),
                selected_score.as_device_ptr(),
                n,
                scale,
            ))
        }
        .map_err(|e| Self::launch_error("adjusted_argmax_single_block", e))
    }

    /// Run softmax, entropy, top-k, and adjusted-score selection in two
    /// launches without materializing a probability matrix.
    ///
    /// Inputs are `n_nodes × n_routes` scores and `n_nodes` state/adaptation
    /// values. Partial buffers are held by the accelerator. The contract is
    /// documented in [`crate::routing`].
    pub fn fused_routing_select(
        &self,
        scores: &GpuBuffer<f32>,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        outputs: &mut FusedRoutingOutputs,
        params: &RoutingParams,
    ) -> GpuResult<()> {
        let launched = self.fused_routing_select_async(scores, state, adaptation, outputs, params);
        let synchronized = self.synchronize();
        launched?;
        synchronized
    }

    /// Enqueue [`Self::fused_routing_select`]; synchronize before using the
    /// buffers. An error after the first launch can leave that launch queued.
    pub fn fused_routing_select_async(
        &self,
        scores: &GpuBuffer<f32>,
        state: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        outputs: &mut FusedRoutingOutputs,
        params: &RoutingParams,
    ) -> GpuResult<()> {
        validate_fused_buffers(scores, state, adaptation, outputs, params)?;
        let kernels = self.kernels()?;
        let pass1 = kernels.get_function("fused_routing_pass1")?;
        let pass2 = kernels.get_function("fused_routing_pass2")?;
        let grid = Self::routing_grid(params.n_nodes);
        self.ensure_routing_scratch(grid.max(1) as usize)?;
        let scratch = self.routing_scratch.borrow();
        let scratch = scratch.as_ref().ok_or_else(|| self.unavailable_error())?;
        let stream = self.ready_stream()?;
        let top_k_probabilities = outputs
            .top_k_probabilities
            .as_ref()
            .map_or_else(cust::memory::DevicePointer::null, |b| b.as_device_ptr());
        // SAFETY: pass 1 is `(const float*, const float*, const float*, int*,
        // float*, float*, float*, float*, unsigned*, int, int, int, int, float)`.
        // validate_fused_buffers checked every input and output length and that
        // top_k is in 1..=MAX_ROUTING_TOP_K; the kernel guards tid >= n_nodes,
        // only writes probabilities through a non-null pointer, and writes grid
        // partials into scratch. Pass 2 reads exactly grid partials, so
        // grid == 0 writes the empty result without reading scratch. Both
        // launches share this stream.
        unsafe {
            if grid > 0 {
                launch!(pass1<<<grid, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                    scores.as_device_ptr(),
                    state.as_device_ptr(),
                    adaptation.as_device_ptr(),
                    outputs.top_k_indices.as_device_ptr(),
                    top_k_probabilities,
                    scratch.entropy_sum.as_device_ptr(),
                    scratch.entropy_max.as_device_ptr(),
                    scratch.scores.as_device_ptr(),
                    scratch.indices.as_device_ptr(),
                    params.n_nodes as i32,
                    params.n_routes as i32,
                    params.top_k as i32,
                    params.score_kind.as_kernel_arg(),
                    params.adaptation_scale,
                ))
                .map_err(|e| Self::launch_error("fused_routing_pass1", e))?;
            }
            launch!(pass2<<<1u32, ROUTING_BLOCK_SIZE as u32, 0, stream>>>(
                scratch.entropy_sum.as_device_ptr(),
                scratch.entropy_max.as_device_ptr(),
                scratch.scores.as_device_ptr(),
                scratch.indices.as_device_ptr(),
                outputs.entropy_sum.as_device_ptr(),
                outputs.entropy_max.as_device_ptr(),
                outputs.selected_index.as_device_ptr(),
                outputs.selected_score.as_device_ptr(),
                grid as i32,
            ))
        }
        .map_err(|e| Self::launch_error("fused_routing_pass2", e))
    }

    /// Blocks covering `n_nodes`. Dimensions were checked against `i32::MAX`.
    fn routing_grid(n_nodes: usize) -> u32 {
        n_nodes.div_ceil(ROUTING_BLOCK_SIZE) as u32
    }
}
