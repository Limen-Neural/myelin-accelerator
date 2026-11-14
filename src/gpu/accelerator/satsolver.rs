// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::capability::sanitize_diagnostic;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use cust::launch;
const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;

impl GpuAccelerator {
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_extract(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        self.satsolver_extract_async(assignment, best_walker, output, n_vars, n_walkers)?;
        self.synchronize()
    }

    /// Async variant of [`Self::satsolver_extract`]: enqueues the launch on the
    /// internal stream and returns without waiting.
    ///
    /// The caller must call [`Self::synchronize`] before reading `output` on the
    /// host or dropping/reusing any argument buffer: `cust` frees device memory
    /// synchronously on `Drop`, which is not ordered against a pending launch.
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_extract_async(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        if n_vars < 0 {
            return Err(GpuError::invalid_input(format!(
                "satsolver_extract: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_walkers <= 0 {
            return Err(GpuError::invalid_input(format!(
                "satsolver_extract: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars == 0 {
            return Ok(());
        }

        let n_vars = n_vars as usize;
        let n_walkers_usize = n_walkers as usize;
        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars),
        )?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len("output", output.len(), n_vars)?;

        let kernels = self.kernels()?;
        let satsolver_extract = kernels.get_function("satsolver_extract")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        let grid = Self::ceil_div_u32(n_vars as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;

        // SAFETY: launching the `satsolver_extract` kernel.
        // - ABI: the CUDA entry (cu/satsolver.cu) is
        //   `(const u8* assignment, const i32* best_walker, u8* output, i32 n_vars, i32 n_walkers)`.
        //   Arguments below match that order and type: `assignment`/`best_walker` are read-only
        //   device pointers, `output` is the write target, and the two trailing scalars are `i32`
        //   (`n_vars` is re-narrowed from the validated `usize`; `n_walkers` is the original `i32`).
        // - Bounds: `n_vars >= 1` and `n_walkers > 0` are validated above; `assignment` has
        //   `>= n_walkers*n_vars` elements, `best_walker >= 1`, and `output >= n_vars` (checked via
        //   `expect_len`). The kernel guards `var >= n_vars` and clamps an out-of-range walker index
        //   to 0, so every `assignment[bw*n_vars + var]` read and `output[var]` write is in bounds.
        // - Launch dims: `grid = ceil_div(n_vars, block)` covers all `n_vars` output threads;
        //   dynamic shared memory is 0 (the kernel declares none).
        // - Lifetime/aliasing: `assignment`, `best_walker`, `output` are borrowed for this call and
        //   `output` is uniquely `&mut`, so there is no host aliasing; the buffers are owned by the
        //   caller and outlive the borrow. Completion is forced by the `synchronize()` in the
        //   synchronous `satsolver_extract` wrapper before the buffers can be freed or reused; the
        //   async form documents that the caller must synchronize before dropping them.
        // - Context/module/stream: `self._ctx`, `self.modules`, and `self.stream` are all `Some`
        //   here (readiness is required to reach a launch) and live as long as `self`.
        unsafe {
            launch!(satsolver_extract<<<grid, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                best_walker.as_device_ptr(),
                output.as_device_ptr(),
                n_vars as i32,
                n_walkers,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "satsolver_extract launch: {e:?}"
                )))
            })?;
        }

        Ok(())
    }

    // 11 parameters mirror the satsolver_aux_update CUDA kernel ABI;
    // grouping them would require a context struct on every call site
    // without simplifying the launch. Allow the clippy lint.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_aux_reduce_best(
        &self,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        best_score: &mut GpuBuffer<i32>,
        best_walker: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
    ) -> GpuResult<()> {
        self.satsolver_aux_reduce_best_async(
            assignment,
            sat_flags,
            scores,
            best_score,
            best_walker,
            clauses,
            n_walkers,
            n_vars,
            n_clauses,
            clause_len,
        )?;
        self.synchronize()
    }

    /// Async variant of [`Self::satsolver_aux_reduce_best`]: enqueues both
    /// reduction launches on the internal stream and returns without waiting.
    ///
    /// The caller must call [`Self::synchronize`] before reading `best_score` /
    /// `best_walker` on the host or dropping/reusing any argument buffer.
    // 11 parameters mirror the satsolver_aux_update CUDA kernel ABI;
    // grouping them would require a context struct on every call site
    // without simplifying the launch. Allow the clippy lint.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments", disable:"Large Method")
    pub fn satsolver_aux_reduce_best_async(
        &self,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        best_score: &mut GpuBuffer<i32>,
        best_walker: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
    ) -> GpuResult<()> {
        if n_walkers <= 0 {
            return Err(GpuError::invalid_input(format!(
                "satsolver_aux_reduce_best: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars < 0 {
            return Err(GpuError::invalid_input(format!(
                "satsolver_aux_reduce_best: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_clauses < 0 {
            return Err(GpuError::invalid_input(format!(
                "satsolver_aux_reduce_best: n_clauses must be >= 0, got {n_clauses}"
            )));
        }
        if clause_len < 0 {
            return Err(GpuError::invalid_input(format!(
                "satsolver_aux_reduce_best: clause_len must be >= 0, got {clause_len}"
            )));
        }

        let n_walkers_usize = n_walkers as usize;
        let n_vars_usize = n_vars as usize;
        let n_clauses_usize = n_clauses as usize;
        let clause_len_usize = clause_len as usize;

        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars_usize),
        )?;
        Self::expect_len(
            "sat_flags",
            sat_flags.len(),
            n_walkers_usize.saturating_mul(n_clauses_usize),
        )?;
        Self::expect_len("scores", scores.len(), n_walkers_usize)?;
        Self::expect_len("best_score", best_score.len(), 1)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len(
            "clauses",
            clauses.len(),
            n_clauses_usize.saturating_mul(clause_len_usize),
        )?;

        let grid_x = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;
        let partial_len = grid_x as usize;
        self.ensure_aux_scratch(partial_len)?;
        let kernels = self.kernels()?;
        let satsolver_aux_update = kernels.get_function("satsolver_aux_update")?;
        let satsolver_best_reduce_pass2 = kernels.get_function("satsolver_best_reduce_pass2")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        let partial_scores = self.aux_partial_scores.borrow();
        let partial_walkers = self.aux_partial_walkers.borrow();
        let partial_scores = partial_scores.as_ref().expect("partial_scores buffer");
        let partial_walkers = partial_walkers.as_ref().expect("partial_walkers buffer");

        // SAFETY: two stream-ordered launches implementing the block-partial + final reduction.
        //
        // `satsolver_aux_update`:
        // - ABI: the CUDA entry is `(const u8* assignment, u8* sat_flags, const i32* scores,
        //   i32* best_score, i32* best_walker, const i32* clauses, i32 n_walkers, i32 n_vars,
        //   i32 n_clauses, i32 clause_len)`. This kernel emits ONE (score, walker) pair per block,
        //   so the wrapper deliberately passes the `partial_scores`/`partial_walkers` scratch
        //   buffers into the kernel's `best_score`/`best_walker` parameter slots (both `i32*`,
        //   matching order and type); the true `best_score`/`best_walker` are reduced by pass2.
        // - Bounds: `n_walkers > 0` and `n_vars,n_clauses,clause_len >= 0` are validated; each input
        //   buffer is length-checked via `expect_len` (`assignment >= n_walkers*n_vars`,
        //   `sat_flags >= n_walkers*n_clauses`, `scores >= n_walkers`, `clauses >= n_clauses*clause_len`).
        //   `partial_scores`/`partial_walkers` are (re)allocated above to hold `>= grid_x` elements,
        //   and the kernel writes exactly one pair per block (`gridDim.x == grid_x`), so every write
        //   `partial_*[blockIdx.x]` is in bounds. The kernel guards `walker >= n_walkers`.
        // - Launch dims: `grid_x = ceil_div(n_walkers, block)`, `block = 256`; shared memory is 0
        //   (the kernel uses a fixed `__shared__ int[32]`, not dynamic shared memory).
        //
        // `satsolver_best_reduce_pass2`:
        // - ABI: `(const i32* partial_scores, const i32* partial_walkers, i32* best_score,
        //   i32* best_walker, i32 n_partials)`; arguments match. `partial_len as i32` is the exact
        //   number of partials produced by the previous launch (`grid_x`, which fits u32/i32 here).
        // - Bounds: reads `partial_*[0..n_partials]` (in bounds by the alloc above) and writes the
        //   single-element `best_score`/`best_walker` (each `expect_len(.., 1)`-checked). Launched
        //   with one block as the kernel documents (`__launch_bounds__(256)`).
        //
        // Ordering/lifetime: both launches use `self.stream`, so pass2 observes pass1's writes
        // without an intervening host sync. All buffers are borrowed for the call (`sat_flags`,
        // `best_score`, `best_walker` uniquely `&mut`; no host aliasing) and the scratch buffers are
        // owned by `self`. The synchronous `satsolver_aux_reduce_best` wrapper calls `synchronize()`
        // before returning; the async form documents the caller must synchronize before freeing or
        // reusing any argument. Context/module/stream are `Some` and live as long as `self`.
        unsafe {
            launch!(satsolver_aux_update<<<grid_x, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                sat_flags.as_device_ptr(),
                scores.as_device_ptr(),
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                clauses.as_device_ptr(),
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "satsolver_aux_update launch: {e:?}"
                )))
            })?;

            launch!(satsolver_best_reduce_pass2<<<1u32, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                best_score.as_device_ptr(),
                best_walker.as_device_ptr(),
                partial_len as i32,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "satsolver_best_reduce_pass2 launch: {e:?}"
                )))
            })?;
        }

        Ok(())
    }

    fn ensure_aux_scratch(&self, partial_len: usize) -> GpuResult<()> {
        self._ctx
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?
            .require_current()?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        let mut partial_scores = self.aux_partial_scores.borrow_mut();
        let mut partial_walkers = self.aux_partial_walkers.borrow_mut();
        let need_partial_realloc = partial_scores
            .as_ref()
            .is_none_or(|b| b.len() < partial_len)
            || partial_walkers
                .as_ref()
                .is_none_or(|b| b.len() < partial_len);
        if need_partial_realloc {
            // The current-context check protects allocation and destruction
            // of replaced scratch buffers. Wait for their last queued use.
            stream.synchronize().map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "stream sync before partial realloc: {e:?}"
                )))
            })?;
            let scores = GpuBuffer::<i32>::alloc(partial_len)?;
            let walkers = GpuBuffer::<i32>::alloc(partial_len)?;
            *partial_scores = Some(scores);
            *partial_walkers = Some(walkers);
        }
        Ok(())
    }
}
