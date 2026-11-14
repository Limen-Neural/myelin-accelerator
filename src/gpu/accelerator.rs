// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::capability::{
    Backend, CapabilityFacts, CapabilityReport, ComputeCapability, ExecutionPolicy, FallbackReason,
    FallbackRecord, KernelAvailability, apply_failure_to_facts, evaluate_capabilities,
    sanitize_diagnostic,
};
use crate::gpu::context::{CurrentContextGuard, GpuContext};
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::kernel::KernelModule;
use crate::gpu::memory::GpuBuffer;
use crate::gpu::routing_ops::RoutingScratch;
use crate::gpu::temporal::TemporalState;
use crate::launch_hook::{LaunchFailure, LaunchType, report_launch_failure};
use cust::stream::{Stream, StreamFlags};
use std::cell::RefCell;
use tracing::warn;

pub(crate) mod poisson;
pub(crate) mod routing;
pub(crate) mod satsolver;
pub(crate) mod stdp;
pub(crate) mod ternary;

/// Owns CUDA resources whose explicit teardown is implemented in `Drop`.
///
/// `KernelModule` and `GpuBuffer` do not retain a context. Every CUDA-owning
/// field added here must be handled in `Drop`, while our context is current,
/// before restoring the caller binding and releasing `_ctx`. Keep `_ctx` last
/// as a defensive lifetime backstop; field order alone is not the teardown.
/// If teardown cannot capture or activate the context, the process aborts:
/// unwinding or returning would destroy resources under an unknown context.
pub struct GpuAccelerator {
    modules: Option<KernelModule>,
    stream: Option<Stream>,
    pub(crate) temporal_state: Option<TemporalState>,
    aux_partial_scores: RefCell<Option<GpuBuffer<i32>>>,
    aux_partial_walkers: RefCell<Option<GpuBuffer<i32>>>,
    pub(super) routing_scratch: RefCell<Option<RoutingScratch>>,
    capabilities: CapabilityReport,
    _ctx: Option<GpuContext>,
}

struct InitFailure {
    facts: CapabilityFacts,
    reason: FallbackReason,
    detail: String,
}

#[derive(Clone, Copy)]
pub(crate) struct LaunchShape {
    pub(crate) grid: u32,
    pub(crate) block: u32,
    pub(crate) shared_mem: u32,
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
        match Self::try_init_gpu(policy) {
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

    fn try_init_gpu(policy: ExecutionPolicy) -> Result<Self, InitFailure> {
        let current_context = CurrentContextGuard::capture();
        let mut facts = crate::host_facts();

        let ctx = Self::init_cuda_context(&mut facts)?;
        let modules = Self::load_kernel_modules(&mut facts, policy)?;
        let stream = Self::create_default_stream(&facts)?;

        Ok(Self::finish_gpu_init(
            ctx,
            modules,
            stream,
            facts,
            current_context,
        ))
    }

    fn init_cuda_context(facts: &mut CapabilityFacts) -> Result<GpuContext, InitFailure> {
        let ctx = match GpuContext::init() {
            Ok(ctx) => ctx,
            Err(e) => {
                let reason = classify_context_failure(facts, &e);
                return Err(InitFailure {
                    facts: facts.clone(),
                    reason,
                    detail: e.to_string(),
                });
            }
        };

        if facts.compute_capability.is_none() {
            facts.compute_capability = ctx.compute_capability;
        }
        match facts.compute_capability {
            Some(cc) if cc.meets_minimum() => Ok(ctx),
            Some(cc) => Err(InitFailure {
                facts: facts.clone(),
                reason: FallbackReason::UnsupportedHardware,
                detail: format!(
                    "compute capability {cc} is below required {}",
                    ComputeCapability::REQUIRED
                ),
            }),
            None => Err(InitFailure {
                facts: facts.clone(),
                reason: FallbackReason::DriverRuntimeFailure,
                detail: "compute capability query failed".to_string(),
            }),
        }
    }

    fn load_kernel_modules(
        facts: &mut CapabilityFacts,
        policy: ExecutionPolicy,
    ) -> Result<Option<KernelModule>, InitFailure> {
        match KernelModule::load_with_availability() {
            Ok(modules) => {
                facts.kernels = KernelAvailability::all_available();
                Ok(Some(modules))
            }
            Err(failure) => {
                facts.kernels = failure.availability;
                let reason = failure
                    .error
                    .fallback_reason()
                    .unwrap_or(FallbackReason::DriverRuntimeFailure);
                if policy == ExecutionPolicy::RequireGpu {
                    return Err(InitFailure {
                        facts: facts.clone(),
                        reason,
                        detail: failure.error.to_string(),
                    });
                }
                warn!(
                    reason = reason.code(),
                    detail = sanitize_diagnostic(&failure.error.to_string()),
                    "fatbin/PTX load failed; continuing with shim-only temporal path when possible"
                );
                Ok(None)
            }
        }
    }

    fn create_default_stream(facts: &CapabilityFacts) -> Result<Stream, InitFailure> {
        Stream::new(StreamFlags::DEFAULT, None).map_err(|e| InitFailure {
            facts: facts.clone(),
            reason: FallbackReason::DriverRuntimeFailure,
            detail: format!("{e:?}"),
        })
    }

    fn finish_gpu_init(
        ctx: GpuContext,
        modules: Option<KernelModule>,
        stream: Stream,
        facts: CapabilityFacts,
        current_context: CurrentContextGuard,
    ) -> Self {
        let capabilities = capability_report_for_success(facts, modules.is_some());
        let accelerator = Self {
            _ctx: Some(ctx),
            modules,
            stream: Some(stream),
            temporal_state: None,
            aux_partial_scores: RefCell::new(None),
            aux_partial_walkers: RefCell::new(None),
            routing_scratch: RefCell::new(None),
            capabilities,
        };
        current_context.disarm();
        accelerator
    }

    fn cpu_fallback(capabilities: CapabilityReport) -> Self {
        Self {
            _ctx: None,
            modules: None,
            stream: None,
            temporal_state: None,
            aux_partial_scores: RefCell::new(None),
            aux_partial_walkers: RefCell::new(None),
            routing_scratch: RefCell::new(None),
            capabilities,
        }
    }

    /// `true` when a CUDA context, stream, and kernel modules are ready for GPU use.
    pub fn is_ready(&self) -> bool {
        self._ctx.is_some()
            && self.stream.is_some()
            && self.modules.is_some()
            && self.capabilities.gpu_usable()
    }

    /// `true` when [`Self::is_ready`] and fatbin/PTX modules loaded successfully.
    pub fn kernels_ready(&self) -> bool {
        self.is_ready()
    }

    pub(crate) fn require_ready(&self) -> GpuResult<()> {
        if self.is_ready() {
            Ok(())
        } else {
            Err(self.unavailable_error())
        }
    }

    pub(crate) fn has_context(&self) -> bool {
        self._ctx.is_some()
    }

    pub(crate) fn require_context_current(&self) -> GpuResult<()> {
        if !self.has_context() {
            return Err(self.unavailable_error());
        }
        self._ctx
            .as_ref()
            .expect("has_context checked above")
            .require_current()
    }

    pub(crate) fn ptx_launch_error(
        kernel_name: &str,
        shape: LaunchShape,
        neuron_count: Option<usize>,
        error: impl std::fmt::Debug,
    ) -> GpuError {
        Self::reported_launch_error(
            kernel_name,
            LaunchType::PtxFatbin,
            shape,
            neuron_count,
            error,
        )
    }

    fn reported_launch_error(
        kernel_name: &str,
        launch_type: LaunchType,
        shape: LaunchShape,
        neuron_count: Option<usize>,
        error: impl std::fmt::Debug,
    ) -> GpuError {
        let gpu_error = GpuError::LaunchFailed(format!("{kernel_name} launch: {error:?}"));
        report_launch_failure(LaunchFailure {
            kernel_name: kernel_name.to_string(),
            launch_type,
            grid: (shape.grid, 1, 1),
            block: (shape.block, 1, 1),
            shared_mem: shape.shared_mem,
            neuron_count,
            error: gpu_error.to_string(),
            jit_error_log: None,
            jit_info_log: None,
        });
        gpu_error
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

    pub(crate) fn has_modules(&self) -> bool {
        self.modules.is_some()
    }

    pub(crate) fn cuda_stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
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

    pub(crate) fn unavailable_error(&self) -> GpuError {
        match &self.capabilities.fallback {
            Some(fb) => GpuError::unavailable(fb.reason, fb.detail.clone()),
            None => GpuError::NoGpu,
        }
    }

    /// Return the current stream, or the unavailable error on CPU-only builds.
    pub(crate) fn ready_stream(&self) -> GpuResult<&Stream> {
        self.stream.as_ref().ok_or_else(|| self.unavailable_error())
    }

    pub(crate) fn launch_error(kernel: &str, error: impl std::fmt::Debug) -> GpuError {
        GpuError::LaunchFailed(sanitize_diagnostic(&format!("{kernel} launch: {error:?}")))
    }

    pub(crate) fn zero_output_prefix(
        &self,
        output: &mut GpuBuffer<f32>,
        count: usize,
    ) -> GpuResult<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| self.unavailable_error())?;
        // SAFETY: output and stream remain alive until the synchronization
        // below completes; all accelerator work uses this same stream.
        unsafe { output.zero_prefix_on(count, stream) }?;
        self.synchronize()
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
        let Some(ctx) = self._ctx.as_ref() else {
            // CPU fallback owns no CUDA resources and makes no CUDA calls.
            return;
        };
        let current_context = match CurrentContextGuard::enter(ctx) {
            Ok(guard) => guard,
            Err(error) => {
                // Returning OR panicking would run field destructors under
                // an unknown context. There is no fallible Drop API and no
                // safe recovery here without leaking resources. Fail-stop;
                // even a failed diagnostic write must not cause unwinding.
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "fatal: cannot establish accelerator context for teardown: {error:?}"
                );
                std::process::abort();
            }
        };
        // Move every resource out before any fallible work or user-provided
        // tracing callback. Reverse local declaration order also preserves
        // teardown during unwinding: these locals drop before the guard.
        let modules = self.modules.take();
        let stream = self.stream.take();
        let temporal_state = self.temporal_state.take();
        let partial_walkers = self.aux_partial_walkers.get_mut().take();
        let partial_scores = self.aux_partial_scores.get_mut().take();
        let routing_scratch = self.routing_scratch.get_mut().take();
        if let Some(stream) = &stream
            && let Err(error) = stream.synchronize()
        {
            warn!(
                ?error,
                "accelerator stream synchronization failed during drop"
            );
        }
        // Explicit ownership invariant: synchronize -> scratch -> stream ->
        // modules -> caller context restoration -> final context release.
        drop(routing_scratch);
        drop(partial_scores);
        drop(partial_walkers);
        drop(temporal_state);
        drop(stream);
        drop(modules);
        drop(current_context);
        drop(self._ctx.take());
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

fn capability_report_for_success(
    mut facts: CapabilityFacts,
    kernels_loaded: bool,
) -> CapabilityReport {
    facts.runtime_available = true;
    facts.device_available = true;
    if kernels_loaded {
        facts.kernels = KernelAvailability::all_available();
    }
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

        let report = capability_report_for_success(facts, true);

        assert!(report.runtime_available);
        assert!(report.device_available);
        assert!(report.gpu_usable());
        assert_eq!(report.selected_backend, Backend::Cuda);
    }

    /// An empty Poisson request on a CPU-fallback accelerator stays
    /// fail-closed on both entry points (GH #48): the async no-op requires
    /// readiness, and the sync wrapper propagates the same `Unavailable`
    /// instead of succeeding-then-failing at `synchronize`.
    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn empty_poisson_on_cpu_fallback_fails_closed_on_both_paths() {
        let _gpu = GpuAccelerator::require_gpu().expect("GPU required for buffer setup");
        let facts = CapabilityFacts {
            cuda_built: true,
            runtime_available: false,
            device_available: false,
            compute_capability: None,
            kernels: KernelAvailability::compiled_unverified(),
        };
        let report = capability_report_for_failure(
            facts,
            FallbackReason::DriverRuntimeFailure,
            "no device for fallback probe",
        );
        let acc = GpuAccelerator::cpu_fallback(report);
        assert!(!acc.is_ready());

        let stim = GpuBuffer::<f32>::alloc(0).unwrap();
        let mut spikes = GpuBuffer::<u32>::alloc(0).unwrap();
        match acc.poisson_encode_async(&stim, &mut spikes, 7).unwrap_err() {
            GpuError::Unavailable { .. } => {}
            other => panic!("expected Unavailable, got {other}"),
        }
        match acc.poisson_encode(&stim, &mut spikes, 7).unwrap_err() {
            GpuError::Unavailable { .. } => {}
            other => panic!("expected Unavailable, got {other}"),
        }
    }
}
