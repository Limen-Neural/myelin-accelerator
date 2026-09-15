// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::bitpacking::TERNARY_VALUES_PER_WORD;
use crate::fused::MAX_FUSED_TOP_K;
use crate::gpu::context::GpuContext;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernel::KernelModule;
use crate::gpu::memory::GpuBuffer;
use cust::launch;
use cust::memory::DeviceCopy;
use cust::stream::{Stream, StreamFlags};
use nvtx::{range_pop, range_push};
use std::cell::RefCell;
use tracing::warn;

const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;
const REDUCE_BLOCK_SIZE: u32 = 256;

pub struct GpuAccelerator {
    _ctx: Option<GpuContext>,
    modules: Option<KernelModule>,
    stream: Option<Stream>,
    aux_partial_scores: RefCell<Option<GpuBuffer<i32>>>,
    aux_partial_walkers: RefCell<Option<GpuBuffer<i32>>>,
    aux_entropy_partial_sum: RefCell<Option<GpuBuffer<f32>>>,
    aux_entropy_partial_max: RefCell<Option<GpuBuffer<f32>>>,
    aux_saaq_partial_scores: RefCell<Option<GpuBuffer<f32>>>,
    aux_saaq_partial_walkers: RefCell<Option<GpuBuffer<u32>>>,
}

impl GpuAccelerator {
    fn from_parts(
        ctx: Option<GpuContext>,
        modules: Option<KernelModule>,
        stream: Option<Stream>,
    ) -> Self {
        Self {
            _ctx: ctx,
            modules,
            stream,
            aux_partial_scores: RefCell::new(None),
            aux_partial_walkers: RefCell::new(None),
            aux_entropy_partial_sum: RefCell::new(None),
            aux_entropy_partial_max: RefCell::new(None),
            aux_saaq_partial_scores: RefCell::new(None),
            aux_saaq_partial_walkers: RefCell::new(None),
        }
    }

    pub fn new() -> Self {
        match GpuContext::init() {
            Ok(ctx) => match (
                KernelModule::load(),
                Stream::new(StreamFlags::DEFAULT, None),
            ) {
                (Ok(modules), Ok(stream)) => {
                    Self::from_parts(Some(ctx), Some(modules), Some(stream))
                }
                (Err(e), _) => {
                    warn!("[GPU] PTX load failed (CPU fallback): {e}");
                    Self::from_parts(Some(ctx), None, None)
                }
                (_, Err(e)) => {
                    warn!("[GPU] stream creation failed (CPU fallback): {e:?}");
                    Self::from_parts(Some(ctx), None, None)
                }
            },
            Err(e) => {
                warn!("[GPU] No CUDA device (CPU fallback): {e}");
                Self::from_parts(None, None, None)
            }
        }
    }

    pub fn is_ready(&self) -> bool {
        self._ctx.is_some() && self.modules.is_some() && self.stream.is_some()
    }

    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        self.modules.as_ref().ok_or(GpuError::NoGpu)
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("stream sync: {e:?}")))
    }

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

    pub fn satsolver_extract_async(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
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
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid = Self::ceil_div_u32(n_vars as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;

        unsafe {
            launch!(satsolver_extract<<<grid, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                best_walker.as_device_ptr(),
                output.as_device_ptr(),
                n_vars as i32,
                n_walkers,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_extract launch: {e:?}")))?;
        }

        Ok(())
    }

    // 11 parameters mirror the satsolver_aux_update CUDA kernel ABI;
    // grouping them would require a context struct on every call site
    // without simplifying the launch. Allow the clippy lint.
    #[allow(clippy::too_many_arguments)]
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

    // 11 parameters mirror the satsolver_aux_update CUDA kernel ABI;
    // grouping them would require a context struct on every call site
    // without simplifying the launch. Allow the clippy lint.
    #[allow(clippy::too_many_arguments)]
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
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_clauses < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_clauses must be >= 0, got {n_clauses}"
            )));
        }
        if clause_len < 0 {
            return Err(GpuError::LaunchFailed(format!(
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

        let kernels = self.kernels()?;
        let satsolver_aux_update = kernels.get_function("satsolver_aux_update")?;
        let satsolver_best_reduce_pass2 = kernels.get_function("satsolver_best_reduce_pass2")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid_x = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;
        let partial_len = grid_x as usize;
        let mut partial_scores = self.aux_partial_scores.borrow_mut();
        let mut partial_walkers = self.aux_partial_walkers.borrow_mut();
        let need_partial_realloc = partial_scores
            .as_ref()
            .is_none_or(|b| b.len() < partial_len)
            || partial_walkers
                .as_ref()
                .is_none_or(|b| b.len() < partial_len);
        if need_partial_realloc {
            stream.synchronize().map_err(|e| {
                GpuError::LaunchFailed(format!("stream sync before partial realloc: {e:?}"))
            })?;
            let scores = GpuBuffer::<i32>::alloc(partial_len)?;
            let walkers = GpuBuffer::<i32>::alloc(partial_len)?;
            *partial_scores = Some(scores);
            *partial_walkers = Some(walkers);
        }
        let partial_scores = partial_scores.as_ref().expect("partial_scores buffer");
        let partial_walkers = partial_walkers.as_ref().expect("partial_walkers buffer");

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
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_aux_update launch: {e:?}")))?;

            launch!(satsolver_best_reduce_pass2<<<1u32, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                best_score.as_device_ptr(),
                best_walker.as_device_ptr(),
                partial_len as i32,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(format!("satsolver_best_reduce_pass2 launch: {e:?}"))
            })?;
        }

        Ok(())
    }

    pub fn poisson_encode(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        self.poisson_encode_async(stimuli, spikes, seed)?;
        self.synchronize()
    }

    pub fn poisson_encode_async(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        let n = stimuli.len();
        Self::expect_len("spikes", spikes.len(), n)?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("poisson_encode")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;

        let block = 256;
        let grid = Self::ceil_div_u32(n as u32, block);

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                stimuli.as_device_ptr(),
                spikes.as_device_ptr(),
                n as i32,
                seed,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("poisson_encode launch: {e:?}")))?;
        }

        Ok(())
    }

    /// Group-scaled ternary GEMV: `y = scale(W) @ x`.
    ///
    /// `packed_w` is row-major `M × ternary_word_count(K)` words.
    /// `scales` is `M × groups_per_row(K, group_size)` f32.
    /// When `skip_zeros` is true, zero trits are skipped on device.
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemv(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        self.ternary_gemv_async(packed_w, scales, x, y, m, k, group_size, skip_zeros)?;
        self.synchronize()
    }

    /// Async variant of [`Self::ternary_gemv`].
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemv_async(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        if m < 0 || k < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemv: m and k must be >= 0, got m={m} k={k}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemv: group_size must be > 0, got {group_size}"
            )));
        }

        let m_u = m as usize;
        let k_u = k as usize;
        Self::expect_len("y", y.len(), m_u)?;

        // Match host-ref: empty product is the zero vector (not "leave y untouched").
        if m == 0 {
            return Ok(());
        }
        if k == 0 {
            // Device memset — no host-sized staging buffer; preserve pooled tail.
            y.zero_prefix(m_u)?;
            return Ok(());
        }

        let group_u = group_size as usize;
        let words_per_row = k_u.div_ceil(TERNARY_VALUES_PER_WORD);
        let n_groups = k_u.div_ceil(group_u);

        Self::expect_len(
            "packed_w",
            packed_w.len(),
            m_u.saturating_mul(words_per_row),
        )?;
        Self::expect_len("scales", scales.len(), m_u.saturating_mul(n_groups))?;
        Self::expect_len("x", x.len(), k_u)?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("ternary_gemv")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let block = 256u32;
        let grid = Self::ceil_div_u32(m as u32, block);
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemv");
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                packed_w.as_device_ptr(),
                scales.as_device_ptr(),
                x.as_device_ptr(),
                y.as_device_ptr(),
                m,
                k,
                group_size,
                skip,
            ))
        };
        range_pop!();
        launch_result.map_err(|e| GpuError::LaunchFailed(format!("ternary_gemv launch: {e:?}")))?;

        Ok(())
    }

    /// Group-scaled ternary GEMM: `C = scale(W) @ B`.
    ///
    /// `B` is `K × N` row-major f32; `C` is `M × N` row-major f32.
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemm(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        self.ternary_gemm_async(packed_w, scales, b, c, m, k, n, group_size, skip_zeros)?;
        self.synchronize()
    }

    /// Async variant of [`Self::ternary_gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemm_async(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        if m < 0 || k < 0 || n < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemm: m, k, n must be >= 0, got m={m} k={k} n={n}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemm: group_size must be > 0, got {group_size}"
            )));
        }

        let m_u = m as usize;
        let k_u = k as usize;
        let n_u = n as usize;
        Self::expect_len("c", c.len(), m_u.saturating_mul(n_u))?;

        // Match host-ref: empty product zeros C (not "leave C untouched").
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // Device memset — no host-sized staging buffer; preserve pooled tail.
            c.zero_prefix(m_u.saturating_mul(n_u))?;
            return Ok(());
        }

        let group_u = group_size as usize;
        let words_per_row = k_u.div_ceil(TERNARY_VALUES_PER_WORD);
        let n_groups = k_u.div_ceil(group_u);

        Self::expect_len(
            "packed_w",
            packed_w.len(),
            m_u.saturating_mul(words_per_row),
        )?;
        Self::expect_len("scales", scales.len(), m_u.saturating_mul(n_groups))?;
        Self::expect_len("b", b.len(), k_u.saturating_mul(n_u))?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("ternary_gemm")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        // m,n are nonnegative i32 after validation; product always fits u64.
        let total = (m as u64) * (n as u64);
        let block = 256u32;
        // Launch uses 1-D grid of u32 block indices over flattened M*N threads.
        let grid_u64 = total.div_ceil(block as u64);
        if grid_u64 > u32::MAX as u64 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemm: grid too large for 1-D launch ({grid_u64} blocks, M*N={total})"
            )));
        }
        let grid = grid_u64 as u32;
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemm");
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                packed_w.as_device_ptr(),
                scales.as_device_ptr(),
                b.as_device_ptr(),
                c.as_device_ptr(),
                m,
                k,
                n,
                group_size,
                skip,
            ))
        };
        range_pop!();
        launch_result.map_err(|e| GpuError::LaunchFailed(format!("ternary_gemm launch: {e:?}")))?;

        Ok(())
    }

    /// Unfused two-pass SAAQ argmax (`saaq_find_best_walker` + pass 2).
    pub fn saaq_select(
        &self,
        membrane: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        best_walker: &mut GpuBuffer<u32>,
        adaptation_scale: f32,
    ) -> GpuResult<()> {
        self.saaq_select_async(membrane, adaptation, best_walker, adaptation_scale)?;
        self.synchronize()
    }

    /// Async variant of [`Self::saaq_select`].
    pub fn saaq_select_async(
        &self,
        membrane: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        best_walker: &mut GpuBuffer<u32>,
        adaptation_scale: f32,
    ) -> GpuResult<()> {
        let n = membrane.len();
        Self::expect_len("adaptation", adaptation.len(), n)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        if n == 0 {
            // Defined empty result: no valid walker; do not leave device memory stale.
            best_walker.upload(&[0])?;
            return Ok(());
        }

        let kernels = self.kernels()?;
        let pass1 = kernels.get_function("saaq_find_best_walker")?;
        let pass2 = kernels.get_function("saaq_reduce_partials_f16")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let block = REDUCE_BLOCK_SIZE;
        let grid = Self::ceil_div_u32(n as u32, block);
        let partial_len = grid as usize;

        self.ensure_aux_f32(&self.aux_saaq_partial_scores, partial_len, stream)?;
        self.ensure_aux_u32(&self.aux_saaq_partial_walkers, partial_len, stream)?;
        let partial_scores = self.aux_saaq_partial_scores.borrow();
        let partial_walkers = self.aux_saaq_partial_walkers.borrow();
        let partial_scores = partial_scores.as_ref().expect("saaq partial scores");
        let partial_walkers = partial_walkers.as_ref().expect("saaq partial walkers");

        range_push!("saaq_select");
        let launch_result = unsafe {
            launch!(pass1<<<grid, block, 0, stream>>>(
                membrane.as_device_ptr(),
                adaptation.as_device_ptr(),
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                n as i32,
                adaptation_scale,
            ))
            .and_then(|_| {
                launch!(pass2<<<1u32, REDUCE_BLOCK_SIZE, 0, stream>>>(
                    partial_scores.as_device_ptr(),
                    partial_walkers.as_device_ptr(),
                    best_walker.as_device_ptr(),
                    partial_len as i32,
                ))
            })
        };
        range_pop!();
        launch_result.map_err(|e| GpuError::LaunchFailed(format!("saaq_select launch: {e:?}")))?;
        Ok(())
    }

    /// Fused single-block SAAQ argmax (no partial buffers).
    pub fn saaq_select_fused(
        &self,
        membrane: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        best_walker: &mut GpuBuffer<u32>,
        adaptation_scale: f32,
    ) -> GpuResult<()> {
        self.saaq_select_fused_async(membrane, adaptation, best_walker, adaptation_scale)?;
        self.synchronize()
    }

    /// Async variant of [`Self::saaq_select_fused`].
    pub fn saaq_select_fused_async(
        &self,
        membrane: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        best_walker: &mut GpuBuffer<u32>,
        adaptation_scale: f32,
    ) -> GpuResult<()> {
        let n = membrane.len();
        Self::expect_len("adaptation", adaptation.len(), n)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        if n == 0 {
            best_walker.upload(&[0])?;
            return Ok(());
        }

        let kernels = self.kernels()?;
        let func = kernels.get_function("saaq_select_fused")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;

        range_push!("saaq_select_fused");
        let launch_result = unsafe {
            launch!(func<<<1u32, REDUCE_BLOCK_SIZE, 0, stream>>>(
                membrane.as_device_ptr(),
                adaptation.as_device_ptr(),
                best_walker.as_device_ptr(),
                n as i32,
                adaptation_scale,
            ))
        };
        range_pop!();
        launch_result
            .map_err(|e| GpuError::LaunchFailed(format!("saaq_select_fused launch: {e:?}")))?;
        Ok(())
    }

    /// Unfused entropy reduce: `routing_entropy_reduce_pass1` + `latent_reduce_pass2`.
    pub fn routing_entropy_reduce(
        &self,
        routing_probs: &GpuBuffer<f32>,
        entropy_sum: &mut GpuBuffer<f32>,
        entropy_max: &mut GpuBuffer<f32>,
        n_nodes: i32,
        n_routes: i32,
    ) -> GpuResult<()> {
        self.routing_entropy_reduce_async(
            routing_probs,
            entropy_sum,
            entropy_max,
            n_nodes,
            n_routes,
        )?;
        self.synchronize()
    }

    /// Async variant of [`Self::routing_entropy_reduce`].
    pub fn routing_entropy_reduce_async(
        &self,
        routing_probs: &GpuBuffer<f32>,
        entropy_sum: &mut GpuBuffer<f32>,
        entropy_max: &mut GpuBuffer<f32>,
        n_nodes: i32,
        n_routes: i32,
    ) -> GpuResult<()> {
        if n_nodes < 0 || n_routes < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "routing_entropy_reduce: n_nodes and n_routes must be >= 0, got {n_nodes} {n_routes}"
            )));
        }
        let n_nodes_u = n_nodes as usize;
        let n_routes_u = n_routes as usize;
        Self::expect_len(
            "routing_probs",
            routing_probs.len(),
            n_nodes_u.saturating_mul(n_routes_u),
        )?;
        Self::expect_len("entropy_sum", entropy_sum.len(), 1)?;
        Self::expect_len("entropy_max", entropy_max.len(), 1)?;
        if n_nodes == 0 {
            return Ok(());
        }

        let kernels = self.kernels()?;
        let pass1 = kernels.get_function("routing_entropy_reduce_pass1")?;
        let pass2 = kernels.get_function("latent_reduce_pass2")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let block = REDUCE_BLOCK_SIZE;
        let grid = Self::ceil_div_u32(n_nodes as u32, block);
        let partial_len = grid as usize;

        self.ensure_aux_f32(&self.aux_entropy_partial_sum, partial_len, stream)?;
        self.ensure_aux_f32(&self.aux_entropy_partial_max, partial_len, stream)?;
        let partial_sum = self.aux_entropy_partial_sum.borrow();
        let partial_max = self.aux_entropy_partial_max.borrow();
        let partial_sum = partial_sum.as_ref().expect("entropy partial sum");
        let partial_max = partial_max.as_ref().expect("entropy partial max");

        range_push!("routing_entropy_reduce");
        let launch_result = unsafe {
            launch!(pass1<<<grid, block, 0, stream>>>(
                routing_probs.as_device_ptr(),
                partial_sum.as_device_ptr(),
                partial_max.as_device_ptr(),
                n_nodes,
                n_routes,
            ))
            .and_then(|_| {
                launch!(pass2<<<1u32, block, 0, stream>>>(
                    partial_sum.as_device_ptr(),
                    partial_max.as_device_ptr(),
                    entropy_sum.as_device_ptr(),
                    entropy_max.as_device_ptr(),
                    partial_len as i32,
                ))
            })
        };
        range_pop!();
        launch_result
            .map_err(|e| GpuError::LaunchFailed(format!("routing_entropy_reduce launch: {e:?}")))?;
        Ok(())
    }

    /// Unfused softmax: logits (or already-normalized scores) → probability matrix.
    pub fn routing_softmax(
        &self,
        scores: &GpuBuffer<f32>,
        probs: &mut GpuBuffer<f32>,
        n_nodes: i32,
        n_routes: i32,
        scores_are_logits: bool,
    ) -> GpuResult<()> {
        self.routing_softmax_async(scores, probs, n_nodes, n_routes, scores_are_logits)?;
        self.synchronize()
    }

    /// Async variant of [`Self::routing_softmax`].
    pub fn routing_softmax_async(
        &self,
        scores: &GpuBuffer<f32>,
        probs: &mut GpuBuffer<f32>,
        n_nodes: i32,
        n_routes: i32,
        scores_are_logits: bool,
    ) -> GpuResult<()> {
        if n_nodes < 0 || n_routes < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "routing_softmax: n_nodes and n_routes must be >= 0, got {n_nodes} {n_routes}"
            )));
        }
        let n_nodes_u = n_nodes as usize;
        let n_routes_u = n_routes as usize;
        let elems = n_nodes_u.saturating_mul(n_routes_u);
        Self::expect_len("scores", scores.len(), elems)?;
        Self::expect_len("probs", probs.len(), elems)?;
        if n_nodes == 0 || n_routes == 0 {
            return Ok(());
        }

        let kernels = self.kernels()?;
        let func = kernels.get_function("routing_softmax")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let block = REDUCE_BLOCK_SIZE;
        let grid = Self::ceil_div_u32(n_nodes as u32, block);
        let logits = if scores_are_logits { 1i32 } else { 0i32 };

        range_push!("routing_softmax");
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                scores.as_device_ptr(),
                probs.as_device_ptr(),
                n_nodes,
                n_routes,
                logits,
            ))
        };
        range_pop!();
        launch_result
            .map_err(|e| GpuError::LaunchFailed(format!("routing_softmax launch: {e:?}")))?;
        Ok(())
    }

    /// Fused routing + entropy + SAAQ selection. Telemetry stays on-device
    /// until the 12-byte pass2 write (`entropy_sum`, `entropy_max`, `best_walker`).
    #[allow(clippy::too_many_arguments)]
    pub fn routing_saaq_fused(
        &self,
        scores: &GpuBuffer<f32>,
        membrane: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        top_k_indices: &mut GpuBuffer<i32>,
        entropy_sum: &mut GpuBuffer<f32>,
        entropy_max: &mut GpuBuffer<f32>,
        best_walker: &mut GpuBuffer<u32>,
        n_nodes: i32,
        n_routes: i32,
        top_k: i32,
        adaptation_scale: f32,
        scores_are_logits: bool,
    ) -> GpuResult<()> {
        self.routing_saaq_fused_async(
            scores,
            membrane,
            adaptation,
            top_k_indices,
            entropy_sum,
            entropy_max,
            best_walker,
            n_nodes,
            n_routes,
            top_k,
            adaptation_scale,
            scores_are_logits,
        )?;
        self.synchronize()
    }

    /// Async variant of [`Self::routing_saaq_fused`].
    #[allow(clippy::too_many_arguments)]
    pub fn routing_saaq_fused_async(
        &self,
        scores: &GpuBuffer<f32>,
        membrane: &GpuBuffer<f32>,
        adaptation: &GpuBuffer<f32>,
        top_k_indices: &mut GpuBuffer<i32>,
        entropy_sum: &mut GpuBuffer<f32>,
        entropy_max: &mut GpuBuffer<f32>,
        best_walker: &mut GpuBuffer<u32>,
        n_nodes: i32,
        n_routes: i32,
        top_k: i32,
        adaptation_scale: f32,
        scores_are_logits: bool,
    ) -> GpuResult<()> {
        if n_nodes < 0 || n_routes < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "routing_saaq_fused: n_nodes and n_routes must be >= 0, got {n_nodes} {n_routes}"
            )));
        }
        if top_k <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "routing_saaq_fused: top_k must be > 0, got {top_k}"
            )));
        }
        if top_k as usize > MAX_FUSED_TOP_K {
            return Err(GpuError::LaunchFailed(format!(
                "routing_saaq_fused: top_k {top_k} exceeds MAX_FUSED_TOP_K ({MAX_FUSED_TOP_K})"
            )));
        }
        let n_nodes_u = n_nodes as usize;
        let n_routes_u = n_routes as usize;
        let top_k_u = top_k as usize;
        Self::expect_len("scores", scores.len(), n_nodes_u.saturating_mul(n_routes_u))?;
        Self::expect_len("membrane", membrane.len(), n_nodes_u)?;
        Self::expect_len("adaptation", adaptation.len(), n_nodes_u)?;
        Self::expect_len(
            "top_k_indices",
            top_k_indices.len(),
            n_nodes_u.saturating_mul(top_k_u),
        )?;
        Self::expect_len("entropy_sum", entropy_sum.len(), 1)?;
        Self::expect_len("entropy_max", entropy_max.len(), 1)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        if n_nodes == 0 {
            entropy_sum.upload(&[0.0])?;
            entropy_max.upload(&[0.0])?;
            best_walker.upload(&[0])?;
            return Ok(());
        }

        let kernels = self.kernels()?;
        let pass1 = kernels.get_function("routing_saaq_fused_pass1")?;
        let pass2 = kernels.get_function("fused_telemetry_reduce_pass2")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let block = REDUCE_BLOCK_SIZE;
        let grid = Self::ceil_div_u32(n_nodes as u32, block);
        let partial_len = grid as usize;

        self.ensure_aux_f32(&self.aux_entropy_partial_sum, partial_len, stream)?;
        self.ensure_aux_f32(&self.aux_entropy_partial_max, partial_len, stream)?;
        self.ensure_aux_f32(&self.aux_saaq_partial_scores, partial_len, stream)?;
        self.ensure_aux_u32(&self.aux_saaq_partial_walkers, partial_len, stream)?;

        let entropy_sum_p = self.aux_entropy_partial_sum.borrow();
        let entropy_max_p = self.aux_entropy_partial_max.borrow();
        let saaq_scores_p = self.aux_saaq_partial_scores.borrow();
        let saaq_walkers_p = self.aux_saaq_partial_walkers.borrow();
        let entropy_sum_p = entropy_sum_p.as_ref().expect("entropy partial sum");
        let entropy_max_p = entropy_max_p.as_ref().expect("entropy partial max");
        let saaq_scores_p = saaq_scores_p.as_ref().expect("saaq partial scores");
        let saaq_walkers_p = saaq_walkers_p.as_ref().expect("saaq partial walkers");

        let logits = if scores_are_logits { 1i32 } else { 0i32 };

        range_push!("routing_saaq_fused");
        let launch_result = unsafe {
            launch!(pass1<<<grid, block, 0, stream>>>(
                scores.as_device_ptr(),
                membrane.as_device_ptr(),
                adaptation.as_device_ptr(),
                top_k_indices.as_device_ptr(),
                entropy_sum_p.as_device_ptr(),
                entropy_max_p.as_device_ptr(),
                saaq_scores_p.as_device_ptr(),
                saaq_walkers_p.as_device_ptr(),
                n_nodes,
                n_routes,
                top_k,
                logits,
                adaptation_scale,
            ))
            .and_then(|_| {
                launch!(pass2<<<1u32, block, 0, stream>>>(
                    entropy_sum_p.as_device_ptr(),
                    entropy_max_p.as_device_ptr(),
                    saaq_scores_p.as_device_ptr(),
                    saaq_walkers_p.as_device_ptr(),
                    entropy_sum.as_device_ptr(),
                    entropy_max.as_device_ptr(),
                    best_walker.as_device_ptr(),
                    partial_len as i32,
                ))
            })
        };
        range_pop!();
        launch_result
            .map_err(|e| GpuError::LaunchFailed(format!("routing_saaq_fused launch: {e:?}")))?;
        Ok(())
    }

    fn ensure_aux_f32(
        &self,
        cell: &RefCell<Option<GpuBuffer<f32>>>,
        len: usize,
        stream: &Stream,
    ) -> GpuResult<()> {
        Self::ensure_aux_len(cell, len, stream)
    }

    fn ensure_aux_u32(
        &self,
        cell: &RefCell<Option<GpuBuffer<u32>>>,
        len: usize,
        stream: &Stream,
    ) -> GpuResult<()> {
        Self::ensure_aux_len(cell, len, stream)
    }

    fn ensure_aux_len<T: DeviceCopy + Default + Clone>(
        cell: &RefCell<Option<GpuBuffer<T>>>,
        len: usize,
        stream: &Stream,
    ) -> GpuResult<()> {
        let mut slot = cell.borrow_mut();
        let need = slot.as_ref().is_none_or(|b| b.len() < len);
        if need {
            stream.synchronize().map_err(|e| {
                GpuError::LaunchFailed(format!("stream sync before aux realloc: {e:?}"))
            })?;
            *slot = Some(GpuBuffer::<T>::alloc(len)?);
        }
        Ok(())
    }

    fn expect_len(name: &str, actual: usize, minimum: usize) -> GpuResult<()> {
        if actual < minimum {
            return Err(GpuError::MemoryError(format!(
                "{name} too small: need at least {minimum} elements, got {actual}"
            )));
        }
        Ok(())
    }

    fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
        value.div_ceil(divisor)
    }
}

impl Drop for GpuAccelerator {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}
