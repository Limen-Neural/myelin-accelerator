// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::bitpacking::TERNARY_VALUES_PER_WORD;
use crate::capability::{
    Backend, CapabilityFacts, CapabilityReport, ComputeCapability, ExecutionPolicy, FallbackReason,
    FallbackRecord, KernelAvailability, apply_failure_to_facts, evaluate_capabilities,
    sanitize_diagnostic,
};
use crate::gpu::context::{CurrentContextGuard, GpuContext};
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernel::KernelModule;
use crate::gpu::memory::GpuBuffer;
use cust::launch;
use cust::stream::{Stream, StreamFlags};
use nvtx::{range_pop, range_push};
use std::cell::RefCell;
use tracing::warn;

const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;

pub struct GpuAccelerator {
    _ctx: Option<GpuContext>,
    modules: Option<KernelModule>,
    stream: Option<Stream>,
    aux_partial_scores: RefCell<Option<GpuBuffer<i32>>>,
    aux_partial_walkers: RefCell<Option<GpuBuffer<i32>>>,
    capabilities: CapabilityReport,
}

struct InitFailure {
    facts: CapabilityFacts,
    reason: FallbackReason,
    detail: String,
}

impl GpuAccelerator {
    /// Construct with [`ExecutionPolicy::PreferGpu`] (caller-approved CPU fallback).
    pub fn new() -> Self {
        match Self::with_policy(ExecutionPolicy::PreferGpu) {
            Ok(acc) => acc,
            Err(_) => unreachable!("PreferGpu construction is infallible"),
        }
    }

    /// Fail closed: never return a CPU-backend accelerator.
    pub fn require_gpu() -> GpuResult<Self> {
        Self::with_policy(ExecutionPolicy::RequireGpu)
    }

    /// Construct under an explicit execution policy.
    pub fn with_policy(policy: ExecutionPolicy) -> GpuResult<Self> {
        match Self::try_init_gpu() {
            Ok(acc) => Ok(acc),
            Err(failure) => {
                let detail = sanitize_diagnostic(&failure.detail);
                let capabilities =
                    capability_report_for_failure(failure.facts, failure.reason, detail.as_str());
                match policy {
                    ExecutionPolicy::PreferGpu => {
                        warn!(
                            reason = failure.reason.code(),
                            backend = Backend::Cpu.code(),
                            detail = detail.as_str(),
                            "GPU unavailable; using CPU fallback"
                        );
                        Ok(Self::cpu_fallback(capabilities))
                    }
                    ExecutionPolicy::RequireGpu => {
                        Err(GpuError::unavailable(failure.reason, detail))
                    }
                }
            }
        }
    }

    fn try_init_gpu() -> Result<Self, InitFailure> {
        let current_context = CurrentContextGuard::capture();
        let mut facts = crate::host_facts();

        let ctx = match GpuContext::init() {
            Ok(ctx) => ctx,
            Err(e) => {
                let reason = classify_context_failure(&facts, &e);
                return Err(InitFailure {
                    facts,
                    reason,
                    detail: e.to_string(),
                });
            }
        };

        if facts.compute_capability.is_none() {
            facts.compute_capability = ctx.compute_capability;
        }
        if let Some(cc) = facts.compute_capability {
            if !cc.meets_minimum() {
                return Err(InitFailure {
                    facts,
                    reason: FallbackReason::UnsupportedHardware,
                    detail: format!(
                        "compute capability {cc} is below required {}",
                        ComputeCapability::REQUIRED
                    ),
                });
            }
        } else {
            return Err(InitFailure {
                facts,
                reason: FallbackReason::DriverRuntimeFailure,
                detail: "compute capability query failed".to_string(),
            });
        }

        let modules = match KernelModule::load_with_availability() {
            Ok(modules) => modules,
            Err(failure) => {
                facts.kernels = failure.availability;
                let reason = failure
                    .error
                    .fallback_reason()
                    .unwrap_or(FallbackReason::DriverRuntimeFailure);
                return Err(InitFailure {
                    facts,
                    reason,
                    detail: failure.error.to_string(),
                });
            }
        };

        facts.kernels = KernelAvailability::all_available();
        let stream = match Stream::new(StreamFlags::DEFAULT, None) {
            Ok(stream) => stream,
            Err(e) => {
                return Err(InitFailure {
                    facts,
                    reason: FallbackReason::DriverRuntimeFailure,
                    detail: format!("{e:?}"),
                });
            }
        };

        let capabilities = capability_report_for_success(facts);
        let accelerator = Self {
            _ctx: Some(ctx),
            modules: Some(modules),
            stream: Some(stream),
            aux_partial_scores: RefCell::new(None),
            aux_partial_walkers: RefCell::new(None),
            capabilities,
        };
        current_context.disarm();
        Ok(accelerator)
    }

    fn cpu_fallback(capabilities: CapabilityReport) -> Self {
        Self {
            _ctx: None,
            modules: None,
            stream: None,
            aux_partial_scores: RefCell::new(None),
            aux_partial_walkers: RefCell::new(None),
            capabilities,
        }
    }

    pub fn is_ready(&self) -> bool {
        self._ctx.is_some()
            && self.modules.is_some()
            && self.stream.is_some()
            && self.capabilities.gpu_usable()
    }

    /// Snapshot of the capability probe used to select this backend.
    pub fn capabilities(&self) -> &CapabilityReport {
        &self.capabilities
    }

    /// Implementation selected for this instance.
    pub fn selected_backend(&self) -> Backend {
        self.capabilities.selected_backend
    }

    /// Fallback record when this instance is not running on CUDA.
    pub fn fallback(&self) -> Option<&FallbackRecord> {
        self.capabilities.fallback.as_ref()
    }

    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        self.modules
            .as_ref()
            .ok_or_else(|| self.unavailable_error())
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        stream.synchronize().map_err(|e| {
            GpuError::LaunchFailed(sanitize_diagnostic(&format!("stream sync: {e:?}")))
        })
    }

    fn unavailable_error(&self) -> GpuError {
        match &self.capabilities.fallback {
            Some(fb) => GpuError::unavailable(fb.reason, fb.detail.clone()),
            None => GpuError::NoGpu,
        }
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

    /// Async variant of [`Self::satsolver_extract`]: enqueues the launch on the
    /// internal stream and returns without waiting.
    ///
    /// The caller must call [`Self::synchronize`] before reading `output` on the
    /// host or dropping/reusing any argument buffer: `cust` frees device memory
    /// synchronously on `Drop`, which is not ordered against a pending launch.
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

        let kernels = self.kernels()?;
        let satsolver_aux_update = kernels.get_function("satsolver_aux_update")?;
        let satsolver_best_reduce_pass2 = kernels.get_function("satsolver_best_reduce_pass2")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
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
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "stream sync before partial realloc: {e:?}"
                )))
            })?;
            let scores = GpuBuffer::<i32>::alloc(partial_len)?;
            let walkers = GpuBuffer::<i32>::alloc(partial_len)?;
            *partial_scores = Some(scores);
            *partial_walkers = Some(walkers);
        }
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

    pub fn poisson_encode(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        self.poisson_encode_async(stimuli, spikes, seed)?;
        self.synchronize()
    }

    /// Async variant of [`Self::poisson_encode`]: enqueues the launch on the
    /// internal stream and returns without waiting.
    ///
    /// The caller must call [`Self::synchronize`] before reading `spikes` on the
    /// host or dropping/reusing either buffer.
    pub fn poisson_encode_async(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        let n = stimuli.len();
        Self::expect_len("spikes", spikes.len(), n)?;

        // The CUDA kernel takes the element count as `int`. Reject `n > i32::MAX`
        // before the cast: `n as i32` would otherwise wrap negative and the
        // kernel's `tid >= n` guard would make every thread skip, silently
        // producing no output. `n as u32` for the grid is safe once `n` fits i32.
        let n_i32 = i32::try_from(n).map_err(|_| {
            GpuError::invalid_input(format!(
                "poisson_encode: element count {n} exceeds i32::MAX kernel limit"
            ))
        })?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("poisson_encode")?;
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;

        let block = 256;
        let grid = Self::ceil_div_u32(n as u32, block);

        // SAFETY: launching the `poisson_encode` kernel.
        // - ABI: the CUDA entry (cu/spiking_network.cu) is
        //   `(const float* stimuli, unsigned int* spikes, int n, unsigned int seed)`. Arguments match:
        //   `stimuli` is a read-only f32 device pointer, `spikes` is the u32 write target, `n_i32`
        //   is the element count, and `seed` is the u32 RNG seed.
        // - Bounds: `spikes.len() == stimuli.len() == n` is enforced above via `expect_len`, and
        //   `n` is checked to fit `i32` via `i32::try_from` (so `n_i32 >= 0` and the grid cast
        //   `n as u32` cannot wrap). The kernel guards `tid >= n`, so every `stimuli[tid]` read and
        //   `spikes[tid]` write is in bounds.
        // - Launch dims: `grid = ceil_div(n, 256)` covers all `n` threads; 0 dynamic shared memory
        //   (the kernel declares none).
        // - Lifetime/aliasing: `stimuli` is shared `&`, `spikes` is unique `&mut` (no host aliasing);
        //   both are caller-owned and outlive the borrow. The synchronous `poisson_encode` wrapper
        //   synchronizes before returning; the async form requires the caller to synchronize before
        //   freeing/reusing the buffers. Context/module/stream are `Some` for as long as `self`.
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                stimuli.as_device_ptr(),
                spikes.as_device_ptr(),
                n_i32,
                seed,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(sanitize_diagnostic(&format!(
                    "poisson_encode launch: {e:?}"
                )))
            })?;
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

    /// Async variant of [`Self::ternary_gemv`]. All output writes, including
    /// empty-K zeroing, are enqueued on the accelerator stream. Call
    /// [`Self::synchronize`] before reading or dropping any argument buffer.
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
            return Err(GpuError::invalid_input(format!(
                "ternary_gemv: m and k must be >= 0, got m={m} k={k}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::invalid_input(format!(
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
            let stream = self
                .stream
                .as_ref()
                .ok_or_else(|| self.unavailable_error())?;
            // SAFETY: the caller must keep y alive until synchronize(); all
            // accelerator launches and this memset use the same stream.
            unsafe { y.zero_prefix_on(m_u, stream) }?;
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
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        let block = 256u32;
        let grid = Self::ceil_div_u32(m as u32, block);
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemv");
        // SAFETY: launching the `ternary_gemv` kernel.
        // - ABI: the CUDA entry (cu/ternary_gemm.cu) is `(const u32* packed_w, const f32* scales,
        //   const f32* x, f32* y, i32 M, i32 K, i32 group_size, i32 skip_zeros)`. Arguments match in
        //   order and type: three read-only device pointers, the `y` write target, then the four
        //   `i32` scalars (`skip` is `0`/`1`).
        // - Bounds: `m,k >= 0` and `group_size > 0` are validated. The `m == 0` and `k == 0` cases
        //   returned earlier, so here `m,k >= 1`. Buffer lengths are checked via `expect_len`:
        //   `y >= m`, `packed_w >= m*ceil(k/TERNARY_VALUES_PER_WORD)`, `scales >= m*ceil(k/group)`,
        //   `x >= k`. These are exactly the strides the kernel indexes (one output row per thread,
        //   `words_per_row` words and `n_groups` scales per row), so all accesses are in bounds.
        // - Launch dims: `grid = ceil_div(m, 256)` covers all `m` output rows; 0 dynamic shared memory.
        // - Lifetime/aliasing: `packed_w`/`scales`/`x` are shared `&`, `y` is unique `&mut` (no host
        //   aliasing); all are caller-owned and outlive the borrow. The synchronous `ternary_gemv`
        //   wrapper synchronizes before returning; the async form requires the caller to synchronize
        //   before freeing/reusing the buffers. Context/module/stream are `Some` for as long as `self`.
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
        launch_result.map_err(|e| {
            GpuError::LaunchFailed(sanitize_diagnostic(&format!("ternary_gemv launch: {e:?}")))
        })?;

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

    /// Async variant of [`Self::ternary_gemm`]. All output writes, including
    /// empty-K zeroing, are enqueued on the accelerator stream. Call
    /// [`Self::synchronize`] before reading or dropping any argument buffer.
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
            return Err(GpuError::invalid_input(format!(
                "ternary_gemm: m, k, n must be >= 0, got m={m} k={k} n={n}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::invalid_input(format!(
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
            let stream = self
                .stream
                .as_ref()
                .ok_or_else(|| self.unavailable_error())?;
            // SAFETY: the caller must keep c alive until synchronize(); all
            // accelerator launches and this memset use the same stream.
            unsafe { c.zero_prefix_on(m_u.saturating_mul(n_u), stream) }?;
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
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        // m,n are nonnegative i32 after validation; product always fits u64.
        let total = (m as u64) * (n as u64);
        let block = 256u32;
        // Launch uses 1-D grid of u32 block indices over flattened M*N threads.
        let grid_u64 = total.div_ceil(block as u64);
        if grid_u64 > u32::MAX as u64 {
            return Err(GpuError::invalid_input(format!(
                "ternary_gemm: grid too large for 1-D launch ({grid_u64} blocks, M*N={total})"
            )));
        }
        let grid = grid_u64 as u32;
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemm");
        // SAFETY: launching the `ternary_gemm` kernel.
        // - ABI: the CUDA entry (cu/ternary_gemm.cu) is `(const u32* packed_w, const f32* scales,
        //   const f32* b, f32* c, i32 M, i32 K, i32 N, i32 group_size, i32 skip_zeros)`. Arguments
        //   match in order and type: three read-only device pointers, the `c` write target, then the
        //   five `i32` scalars (`skip` is `0`/`1`).
        // - Bounds: `m,k,n >= 0` and `group_size > 0` are validated. The `m == 0 || n == 0` and
        //   `k == 0` cases returned earlier, so here `m,k,n >= 1`. Buffer lengths are checked via
        //   `expect_len`: `c >= m*n`, `packed_w >= m*ceil(k/TERNARY_VALUES_PER_WORD)`,
        //   `scales >= m*ceil(k/group)`, `b >= k*n`. The kernel maps one flattened `M*N` output
        //   element per thread and indexes those same strides, so all accesses are in bounds.
        // - Launch dims: `grid = ceil_div(M*N, 256)` computed in u64 and rejected above if it would
        //   exceed `u32::MAX`, so the 1-D grid covers every output element without overflow;
        //   0 dynamic shared memory.
        // - Lifetime/aliasing: `packed_w`/`scales`/`b` are shared `&`, `c` is unique `&mut` (no host
        //   aliasing); all are caller-owned and outlive the borrow. The synchronous `ternary_gemm`
        //   wrapper synchronizes before returning; the async form requires the caller to synchronize
        //   before freeing/reusing the buffers. Context/module/stream are `Some` for as long as `self`.
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
        launch_result.map_err(|e| {
            GpuError::LaunchFailed(sanitize_diagnostic(&format!("ternary_gemm launch: {e:?}")))
        })?;

        Ok(())
    }

    fn expect_len(name: &str, actual: usize, minimum: usize) -> GpuResult<()> {
        if actual < minimum {
            return Err(GpuError::invalid_input(format!(
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
        if self.stream.is_some()
            && let Err(error) = self.synchronize()
        {
            warn!(%error, "accelerator stream synchronization failed during drop");
        }
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

fn classify_init_failure(facts: &CapabilityFacts) -> FallbackReason {
    if !facts.runtime_available {
        FallbackReason::DriverRuntimeFailure
    } else if !facts.device_available {
        FallbackReason::DeviceUnavailable
    } else if facts
        .compute_capability
        .is_some_and(|cc| !cc.meets_minimum())
    {
        FallbackReason::UnsupportedHardware
    } else {
        FallbackReason::DriverRuntimeFailure
    }
}

fn classify_context_failure(facts: &CapabilityFacts, error: &GpuError) -> FallbackReason {
    error
        .fallback_reason()
        .unwrap_or_else(|| classify_init_failure(facts))
}

fn capability_report_for_success(mut facts: CapabilityFacts) -> CapabilityReport {
    facts.runtime_available = true;
    facts.device_available = true;
    facts.kernels = KernelAvailability::all_available();
    evaluate_capabilities(&facts)
}

fn capability_report_for_failure(
    mut facts: CapabilityFacts,
    reason: FallbackReason,
    detail: &str,
) -> CapabilityReport {
    apply_failure_to_facts(&mut facts, reason);
    let mut report = evaluate_capabilities(&facts);
    report.selected_backend = Backend::Cpu;
    report.fallback = Some(FallbackRecord::cpu(reason, detail));
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_report_preserves_sanitized_initialization_detail() {
        let facts = CapabilityFacts {
            cuda_built: true,
            runtime_available: true,
            device_available: true,
            compute_capability: Some(ComputeCapability::REQUIRED),
            kernels: KernelAvailability::compiled_unverified(),
        };

        let report = capability_report_for_failure(
            facts,
            FallbackReason::KernelSpecializationUnavailable,
            "PTX load failed module=/home/alice/private.ptx InvalidPtx",
        );
        let fallback = report.fallback.expect("failed initialization falls back");

        assert_eq!(
            fallback.reason,
            FallbackReason::KernelSpecializationUnavailable
        );
        assert_eq!(fallback.detail, "PTX load failed module=<path> InvalidPtx");
    }

    #[test]
    fn typed_context_failure_reason_takes_precedence_over_stale_facts() {
        let facts = CapabilityFacts {
            cuda_built: true,
            runtime_available: true,
            device_available: true,
            compute_capability: Some(ComputeCapability::REQUIRED),
            kernels: KernelAvailability::compiled_unverified(),
        };
        let error =
            GpuError::unavailable(FallbackReason::DeviceUnavailable, "get_device(0): NoDevice");

        assert_eq!(
            classify_context_failure(&facts, &error),
            FallbackReason::DeviceUnavailable
        );
    }

    #[test]
    fn successful_initialization_overrides_stale_negative_facts() {
        let facts = CapabilityFacts {
            cuda_built: true,
            runtime_available: false,
            device_available: false,
            compute_capability: Some(ComputeCapability::REQUIRED),
            kernels: KernelAvailability::compiled_unverified(),
        };

        let report = capability_report_for_success(facts);

        assert!(report.runtime_available);
        assert!(report.device_available);
        assert!(report.gpu_usable());
        assert_eq!(report.selected_backend, Backend::Cuda);
    }
}
