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
    /// Initialise WalkSAT walker assignments and their initial unsatisfied-clause counts.
    ///
    /// `assignment` is row-major `n_walkers × n_vars`, `scores` has one element
    /// per walker, and `clauses` is row-major `n_clauses × clause_len` using the
    /// kernel's signed literal encoding. Synchronizes before returning.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_init(
        &self,
        assignment: &mut GpuBuffer<u8>,
        scores: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
        seed: u32,
    ) -> GpuResult<()> {
        let launch_result = self.satsolver_init_async(
            assignment, scores, clauses, n_walkers, n_vars, n_clauses, clause_len, seed,
        );
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::satsolver_init`]; synchronize before reading or reusing the buffers.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_init_async(
        &self,
        assignment: &mut GpuBuffer<u8>,
        scores: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
        seed: u32,
    ) -> GpuResult<()> {
        validate_walksat_args(
            "satsolver_init",
            assignment.len(),
            scores.len(),
            clauses.len(),
            n_walkers,
            n_vars,
            n_clauses,
            clause_len,
        )?;

        let func = self.kernels()?.get_function("satsolver_init")?;
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(u8* assignment, i32* scores,
        // const i32* clauses, int n_walkers, int n_vars, int n_clauses,
        // int clause_len, unsigned int seed)`. Types and ordering match.
        // Buffer lengths cover the validated walker assignment, score, and
        // clause matrices. Grid coverage includes every walker and the kernel
        // guards `walker >= n_walkers`. Buffers remain alive until synchronization.
        unsafe {
            launch!(func<<<grid, SATSOLVER_BLOCK_SIZE, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                scores.as_device_ptr(),
                clauses.as_device_ptr(),
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
                seed,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "satsolver_init launch: {e:?}"
                )))
            })?;
        }
        Ok(())
    }

    /// Run one WalkSAT iteration for every walker.
    ///
    /// Dimensions and buffer layout match [`Self::satsolver_init`]. `seed` is
    /// the per-step seed; callers that need independent steps should combine it
    /// with their step counter. Synchronizes before returning.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_step(
        &self,
        assignment: &mut GpuBuffer<u8>,
        scores: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
        seed: u32,
    ) -> GpuResult<()> {
        let launch_result = self.satsolver_step_async(
            assignment, scores, clauses, n_walkers, n_vars, n_clauses, clause_len, seed,
        );
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::satsolver_step`]; synchronize before reading or reusing the buffers.
    #[allow(clippy::too_many_arguments)]
    // @codescene(disable:"Excess Number of Function Arguments")
    pub fn satsolver_step_async(
        &self,
        assignment: &mut GpuBuffer<u8>,
        scores: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
        seed: u32,
    ) -> GpuResult<()> {
        validate_walksat_args(
            "satsolver_step",
            assignment.len(),
            scores.len(),
            clauses.len(),
            n_walkers,
            n_vars,
            n_clauses,
            clause_len,
        )?;

        let func = self.kernels()?.get_function("satsolver_step")?;
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(u8* assignment, i32* scores,
        // const i32* clauses, int n_walkers, int n_vars, int n_clauses,
        // int clause_len, unsigned int seed)`. Types, dimensions, grid
        // coverage, buffer bounds, and lifetimes are the same as validated for
        // `satsolver_init` above.
        unsafe {
            launch!(func<<<grid, SATSOLVER_BLOCK_SIZE, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                scores.as_device_ptr(),
                clauses.as_device_ptr(),
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
                seed,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "satsolver_step launch: {e:?}"
                )))
            })?;
        }
        Ok(())
    }

    /// Mark walkers whose score is zero as solved.
    ///
    /// `scores.len()` is the walker count and `solved` must contain at least
    /// that many bytes. Synchronizes before returning.
    pub fn satsolver_check_solution(
        &self,
        scores: &GpuBuffer<i32>,
        solved: &mut GpuBuffer<u8>,
    ) -> GpuResult<()> {
        let launch_result = self.satsolver_check_solution_async(scores, solved);
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::satsolver_check_solution`]; synchronize before reading
    /// or reusing the buffers.
    pub fn satsolver_check_solution_async(
        &self,
        scores: &GpuBuffer<i32>,
        solved: &mut GpuBuffer<u8>,
    ) -> GpuResult<()> {
        let n_walkers = scores.len();
        Self::expect_len("solved", solved.len(), n_walkers)?;
        let n_walkers_i32 = i32::try_from(n_walkers).map_err(|_| {
            GpuError::invalid_input(format!(
                "satsolver_check_solution: n_walkers {n_walkers} exceeds i32::MAX kernel limit"
            ))
        })?;
        if n_walkers == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self.kernels()?.get_function("satsolver_check_solution")?;
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(const i32* scores, u8* solved,
        // int n_walkers)`. Both arrays contain at least `n_walkers` elements,
        // the grid covers every walker, and the kernel guards
        // `walker >= n_walkers`. Buffers remain alive until synchronization.
        unsafe {
            launch!(func<<<grid, SATSOLVER_BLOCK_SIZE, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                scores.as_device_ptr(),
                solved.as_device_ptr(),
                n_walkers_i32,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "satsolver_check_solution launch: {e:?}"
                )))
            })?;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_walksat_args(
    operation: &str,
    assignment_len: usize,
    scores_len: usize,
    clauses_len: usize,
    n_walkers: i32,
    n_vars: i32,
    n_clauses: i32,
    clause_len: i32,
) -> GpuResult<()> {
    if n_walkers <= 0 {
        return Err(GpuError::invalid_input(format!(
            "{operation}: n_walkers must be > 0, got {n_walkers}"
        )));
    }
    for (name, value) in [
        ("n_vars", n_vars),
        ("n_clauses", n_clauses),
        ("clause_len", clause_len),
    ] {
        if value < 0 {
            return Err(GpuError::invalid_input(format!(
                "{operation}: {name} must be >= 0, got {value}"
            )));
        }
    }

    let walkers = n_walkers as usize;
    let vars = n_vars as usize;
    let clauses = n_clauses as usize;
    let clause_width = clause_len as usize;
    let assignment_elements = walkers.checked_mul(vars).ok_or_else(|| {
        GpuError::invalid_input(format!("{operation}: assignment dimensions overflow"))
    })?;
    let clause_elements = clauses.checked_mul(clause_width).ok_or_else(|| {
        GpuError::invalid_input(format!("{operation}: clause dimensions overflow"))
    })?;
    GpuAccelerator::expect_len("assignment", assignment_len, assignment_elements)?;
    GpuAccelerator::expect_len("scores", scores_len, walkers)?;
    GpuAccelerator::expect_len("clauses", clauses_len, clause_elements)?;
    Ok(())
}
