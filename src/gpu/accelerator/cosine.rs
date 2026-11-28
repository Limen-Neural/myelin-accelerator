// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use cust::launch;

const COSINE_BATCHED_BLOCK_SIZE: u32 = 32;
// CUDA grid Y is capped at 65,535 blocks on supported devices.
const COSINE_BATCHED_QUERY_CHUNK: usize = 65_535;
const COSINE_TOP_K_BLOCK_SIZE: u32 = 256;

impl GpuAccelerator {
    /// Compute the full row-major `n_queries × n_keys` cosine-score matrix.
    ///
    /// Inputs are row-major `n_queries × dim` and `n_keys × dim` matrices.
    /// Buffers may be oversized; only the prefix described by the dimensions is
    /// used. Synchronizes before returning.
    pub fn cosine_similarity_batched(
        &self,
        queries: &GpuBuffer<f32>,
        keys: &GpuBuffer<f32>,
        out: &mut GpuBuffer<f32>,
        n_queries: usize,
        n_keys: usize,
        dim: usize,
    ) -> GpuResult<()> {
        self.cosine_similarity_batched_async(queries, keys, out, n_queries, n_keys, dim)?;
        self.synchronize()
    }

    /// Enqueue [`Self::cosine_similarity_batched`]; synchronize before reading
    /// or reusing the buffers. If enqueueing a chunk fails, this method
    /// attempts to synchronize previously queued chunks before returning the
    /// launch error. Discard the output on error: it may be partially written.
    pub fn cosine_similarity_batched_async(
        &self,
        queries: &GpuBuffer<f32>,
        keys: &GpuBuffer<f32>,
        out: &mut GpuBuffer<f32>,
        n_queries: usize,
        n_keys: usize,
        dim: usize,
    ) -> GpuResult<()> {
        let dims = validate_cosine_inputs(
            "cosine_similarity_batched",
            queries.len(),
            keys.len(),
            n_queries,
            n_keys,
            dim,
        )?;
        Self::expect_len(
            "similarity output",
            out.len(),
            matrix_elements("similarity output", n_queries, n_keys)?,
        )?;
        if n_queries == 0 || n_keys == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self.kernels()?.get_function("cosine_similarity_batched")?;
        let stream = self.ready_stream()?;

        for query_start in (0..n_queries).step_by(COSINE_BATCHED_QUERY_CHUNK) {
            let chunk_len = (n_queries - query_start).min(COSINE_BATCHED_QUERY_CHUNK);
            // SAFETY: the validated row-major matrices cover each chunk.
            // Offsets are within the query/output allocations and each chunk
            // writes a disjoint output range. The key matrix is unchanged.
            // Launch uses one 32-thread block per pair and a legal grid Y.
            // Buffers remain alive until synchronization.
            let launch_result = unsafe {
                let chunk_queries = queries.as_device_ptr().add(query_start * dim);
                let chunk_out = out.as_device_ptr().add(query_start * n_keys);
                launch!(func<<<(dims.n_keys_u32, chunk_len as u32, 1), COSINE_BATCHED_BLOCK_SIZE, 0, stream>>>(
                    chunk_queries,
                    keys.as_device_ptr(),
                    chunk_out,
                    chunk_len as i32,
                    dims.n_keys_i32,
                    dims.dim_i32,
                ))
            };
            if let Err(error) = launch_result {
                // A later launch can fail after an earlier one was queued.
                // Drain that work before returning when the stream is usable,
                // and preserve the original launch diagnostic.
                let _ = stream.synchronize();
                return Err(Self::launch_error("cosine_similarity_batched", error));
            }
        }
        Ok(())
    }

    /// Compute cosine scores and the per-query top-k key indices.
    ///
    /// `similarities` is the full row-major `n_queries × n_keys` matrix and
    /// `top_k_indices` is row-major `n_queries × top_k`. Selection is descending
    /// score with lower-key tie breaks; at most 32 keys are selected and unused
    /// slots are `-1`. A `top_k` of zero is a no-op that requires no output
    /// storage. Synchronizes before returning.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn cosine_similarity_top_k(
        &self,
        queries: &GpuBuffer<f32>,
        keys: &GpuBuffer<f32>,
        similarities: &mut GpuBuffer<f32>,
        top_k_indices: &mut GpuBuffer<i32>,
        n_queries: usize,
        n_keys: usize,
        dim: usize,
        top_k: usize,
    ) -> GpuResult<()> {
        let launch_result = self.cosine_similarity_top_k_async(
            queries,
            keys,
            similarities,
            top_k_indices,
            n_queries,
            n_keys,
            dim,
            top_k,
        );
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::cosine_similarity_top_k`]; synchronize before reading
    /// or reusing the buffers.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn cosine_similarity_top_k_async(
        &self,
        queries: &GpuBuffer<f32>,
        keys: &GpuBuffer<f32>,
        similarities: &mut GpuBuffer<f32>,
        top_k_indices: &mut GpuBuffer<i32>,
        n_queries: usize,
        n_keys: usize,
        dim: usize,
        top_k: usize,
    ) -> GpuResult<()> {
        let dims = validate_cosine_inputs(
            "cosine_similarity_top_k",
            queries.len(),
            keys.len(),
            n_queries,
            n_keys,
            dim,
        )?;
        let top_k_i32 = kernel_dimension("cosine_similarity_top_k", "top_k", top_k)?;
        if n_queries == 0 || top_k == 0 {
            self.require_ready()?;
            return Ok(());
        }
        Self::expect_len(
            "similarity output",
            similarities.len(),
            matrix_elements("similarity output", n_queries, n_keys)?,
        )?;
        let index_elements = matrix_elements("top_k_indices", n_queries, top_k)?;
        Self::expect_len("top_k_indices", top_k_indices.len(), index_elements)?;

        let func = self.kernels()?.get_function("cosine_similarity_top_k")?;
        let stream = self.ready_stream()?;

        // SAFETY: the CUDA entry is `(const float* queries, const float* keys,
        // float* similarities, int* top_k_indices, int n_queries, int n_keys,
        // int dim, int top_k)`. Types and ordering match. Matrix and index
        // buffer lengths cover the validated row-major outputs. Grid.x is
        // `n_queries` and the 256-thread block is within the kernel's documented
        // 32–1024 full-warp requirement. Dynamic shared memory is zero because
        // the kernel uses static shared memory only. Buffers remain alive until
        // synchronization.
        unsafe {
            launch!(func<<<dims.n_queries_u32, COSINE_TOP_K_BLOCK_SIZE, 0, stream>>>(
                queries.as_device_ptr(),
                keys.as_device_ptr(),
                similarities.as_device_ptr(),
                top_k_indices.as_device_ptr(),
                dims.n_queries_i32,
                dims.n_keys_i32,
                dims.dim_i32,
                top_k_i32,
            ))
            .map_err(|e| Self::launch_error("cosine_similarity_top_k", e))?;
        }
        Ok(())
    }
}

struct CosineDims {
    n_queries_i32: i32,
    n_keys_i32: i32,
    dim_i32: i32,
    n_queries_u32: u32,
    n_keys_u32: u32,
}

fn validate_cosine_inputs(
    operation: &str,
    queries_len: usize,
    keys_len: usize,
    n_queries: usize,
    n_keys: usize,
    dim: usize,
) -> GpuResult<CosineDims> {
    if dim == 0 {
        return Err(GpuError::invalid_input(format!(
            "{operation}: dim must be greater than zero"
        )));
    }
    let n_queries_i32 = kernel_dimension(operation, "n_queries", n_queries)?;
    let n_keys_i32 = kernel_dimension(operation, "n_keys", n_keys)?;
    let dim_i32 = kernel_dimension(operation, "dim", dim)?;
    GpuAccelerator::expect_len(
        "queries",
        queries_len,
        matrix_elements("queries", n_queries, dim)?,
    )?;
    GpuAccelerator::expect_len("keys", keys_len, matrix_elements("keys", n_keys, dim)?)?;
    Ok(CosineDims {
        n_queries_i32,
        n_keys_i32,
        dim_i32,
        n_queries_u32: n_queries as u32,
        n_keys_u32: n_keys as u32,
    })
}

fn kernel_dimension(operation: &str, name: &str, value: usize) -> GpuResult<i32> {
    i32::try_from(value).map_err(|_| {
        GpuError::invalid_input(format!(
            "{operation}: {name} {value} exceeds i32::MAX kernel limit"
        ))
    })
}

fn matrix_elements(label: &str, rows: usize, columns: usize) -> GpuResult<usize> {
    rows.checked_mul(columns).ok_or_else(|| {
        GpuError::invalid_input(format!("{label}: dimensions overflow host element count"))
    })
}
