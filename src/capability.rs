// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured CUDA capability probes and fallback reason codes.
//!
//! Call [`probe_capabilities`] before launching work. Selection follows one
//! documented policy ([`ExecutionPolicy`]):
//!
//! * [`ExecutionPolicy::PreferGpu`] — caller-approved CPU fallback, recorded
//!   as a [`FallbackRecord`].
//! * [`ExecutionPolicy::RequireGpu`] — fail closed; never execute on CPU.
//!
//! Decision table (first matching row wins):
//!
//! | `cuda_built` | runtime | device | CC ≥ 12.0 | kernel runtime | selected | reason |
//! |--------------|---------|--------|-----------|----------------|----------|--------|
//! | false | * | * | * | * | `Cpu` | `cuda_feature_not_built` |
//! | true | false | * | * | * | `Cpu` | `driver_runtime_failure` |
//! | true | true | false | * | * | `Cpu` | `device_unavailable` |
//! | true | true | true | missing | * | `Cpu` | `driver_runtime_failure` |
//! | true | true | true | false | * | `Cpu` | `unsupported_hardware` |
//! | true | true | true | true | any `false` | `Cpu` | `kernel_specialization_unavailable` |
//! | true | true | true | true | unknown/any `false` | `Cpu` | `kernel_specialization_unavailable` |
//! | true | true | true | true | all `true` | `Cuda` | — |
//!
//! `invalid_input` is a request-level reason (bad launch arguments), not a
//! host-probe outcome.

use std::fmt;

/// Blackwell-class floor for first-party PTX (`sm_120`).
pub const REQUIRED_COMPUTE_CAPABILITY: ComputeCapability = ComputeCapability {
    major: 12,
    minor: 0,
};

/// Selected implementation for a probe or accelerator instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// CUDA device kernels (requires the `cuda` feature and a usable GPU).
    Cuda,
    /// Host/CPU path (stub buffers, no device launches).
    Cpu,
}

impl Backend {
    /// Stable telemetry token.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// How public entry points choose a backend when GPU is unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPolicy {
    /// Use GPU when capable; otherwise CPU with an explicit [`FallbackRecord`].
    PreferGpu,
    /// Require GPU. Construction and launch fail; CPU is never selected.
    RequireGpu,
}

impl ExecutionPolicy {
    /// Stable telemetry token.
    pub const fn code(self) -> &'static str {
        match self {
            Self::PreferGpu => "prefer_gpu",
            Self::RequireGpu => "require_gpu",
        }
    }
}

impl fmt::Display for ExecutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Stable reason code explaining why GPU was not selected (or why a request failed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FallbackReason {
    /// Built without the `cuda` Cargo feature (PTX/kernels not compiled in).
    CudaFeatureNotBuilt,
    /// CUDA driver/runtime init or stream setup failed.
    DriverRuntimeFailure,
    /// No accessible CUDA device.
    DeviceUnavailable,
    /// Device compute capability is below [`REQUIRED_COMPUTE_CAPABILITY`].
    UnsupportedHardware,
    /// Required PTX family or kernel symbol is missing after JIT.
    KernelSpecializationUnavailable,
    /// Caller-supplied launch arguments are invalid.
    InvalidInput,
}

impl FallbackReason {
    /// Stable snake_case code for tests and telemetry.
    pub const fn code(self) -> &'static str {
        match self {
            Self::CudaFeatureNotBuilt => "cuda_feature_not_built",
            Self::DriverRuntimeFailure => "driver_runtime_failure",
            Self::DeviceUnavailable => "device_unavailable",
            Self::UnsupportedHardware => "unsupported_hardware",
            Self::KernelSpecializationUnavailable => "kernel_specialization_unavailable",
            Self::InvalidInput => "invalid_input",
        }
    }
}

impl fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// CUDA compute capability (`major.minor`), e.g. Blackwell `12.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    /// Same as [`REQUIRED_COMPUTE_CAPABILITY`].
    pub const REQUIRED: Self = REQUIRED_COMPUTE_CAPABILITY;

    /// `true` when this capability can run `sm_120` first-party PTX.
    ///
    /// The published floor is [`REQUIRED_COMPUTE_CAPABILITY`] (`12.0`). Any
    /// `major >= 12` is accepted; `minor` is reserved for a future raise.
    pub const fn meets_minimum(self) -> bool {
        self.major >= Self::REQUIRED.major
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Compile-time and optional runtime availability of the four PTX families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelAvailability {
    /// PTX for the first-party kernels was compiled into this binary.
    pub compiled: bool,
    /// Spiking-network family after JIT, if attempted.
    pub spiking_network: Option<bool>,
    /// Vector-similarity family after JIT, if attempted.
    pub vector_similarity: Option<bool>,
    /// SAT-solver family after JIT, if attempted.
    pub satsolver: Option<bool>,
    /// Ternary GEMV/GEMM family after JIT, if attempted.
    pub ternary_gemm: Option<bool>,
}

impl KernelAvailability {
    /// Nothing compiled in (CPU-only / stub build).
    pub const fn not_compiled() -> Self {
        Self {
            compiled: false,
            spiking_network: None,
            vector_similarity: None,
            satsolver: None,
            ternary_gemm: None,
        }
    }

    /// PTX is embedded; driver JIT has not been attempted.
    pub const fn compiled_unverified() -> Self {
        Self {
            compiled: true,
            spiking_network: None,
            vector_similarity: None,
            satsolver: None,
            ternary_gemm: None,
        }
    }

    /// All four families JIT-loaded successfully.
    pub const fn all_available() -> Self {
        Self {
            compiled: true,
            spiking_network: Some(true),
            vector_similarity: Some(true),
            satsolver: Some(true),
            ternary_gemm: Some(true),
        }
    }

    /// PTX is present but runtime specialization failed.
    pub const fn all_unavailable() -> Self {
        Self {
            compiled: true,
            spiking_network: Some(false),
            vector_similarity: Some(false),
            satsolver: Some(false),
            ternary_gemm: Some(false),
        }
    }

    /// `true` when a runtime probe reported at least one missing family.
    pub const fn any_runtime_unavailable(self) -> bool {
        matches!(self.spiking_network, Some(false))
            || matches!(self.vector_similarity, Some(false))
            || matches!(self.satsolver, Some(false))
            || matches!(self.ternary_gemm, Some(false))
    }

    /// `true` only after all required PTX families JIT-loaded successfully.
    pub const fn all_runtime_available(self) -> bool {
        matches!(self.spiking_network, Some(true))
            && matches!(self.vector_similarity, Some(true))
            && matches!(self.satsolver, Some(true))
            && matches!(self.ternary_gemm, Some(true))
    }
}

impl Default for KernelAvailability {
    fn default() -> Self {
        Self::not_compiled()
    }
}

/// Observable inputs to [`evaluate_capabilities`].
///
/// Production code uses [`probe_capabilities`]; tests inject mocks here.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapabilityFacts {
    pub cuda_built: bool,
    pub runtime_available: bool,
    pub device_available: bool,
    pub compute_capability: Option<ComputeCapability>,
    pub kernels: KernelAvailability,
}

impl CapabilityFacts {
    /// Honest CPU-only / not-built snapshot.
    pub const fn not_built() -> Self {
        Self {
            cuda_built: false,
            runtime_available: false,
            device_available: false,
            compute_capability: None,
            kernels: KernelAvailability::not_compiled(),
        }
    }
}

/// Recorded CPU (or reduced-kernel) selection with a stable reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackRecord {
    pub reason: FallbackReason,
    pub selected_backend: Backend,
    /// Path-free, secret-free diagnostic suitable for tests and telemetry.
    pub detail: String,
}

impl FallbackRecord {
    /// CPU fallback with a sanitized detail string.
    pub fn cpu(reason: FallbackReason, detail: impl Into<String>) -> Self {
        Self {
            reason,
            selected_backend: Backend::Cpu,
            detail: sanitize_diagnostic(&detail.into()),
        }
    }
}

/// Typed capability snapshot for callers and telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityReport {
    pub cuda_built: bool,
    pub runtime_available: bool,
    pub device_available: bool,
    pub compute_capability: Option<ComputeCapability>,
    pub kernels: KernelAvailability,
    pub selected_backend: Backend,
    pub fallback: Option<FallbackRecord>,
}

impl CapabilityReport {
    /// `true` when CUDA is the selected backend and no fallback was recorded.
    pub fn gpu_usable(&self) -> bool {
        self.selected_backend == Backend::Cuda && self.fallback.is_none()
    }

    /// Apply an execution policy to this probe.
    ///
    /// [`ExecutionPolicy::PreferGpu`] returns the selected backend (possibly
    /// CPU). [`ExecutionPolicy::RequireGpu`] returns `Err` instead of CPU.
    pub fn select_backend(&self, policy: ExecutionPolicy) -> Result<Backend, FallbackRecord> {
        match policy {
            ExecutionPolicy::PreferGpu => Ok(self.selected_backend),
            ExecutionPolicy::RequireGpu => {
                if self.gpu_usable() {
                    Ok(Backend::Cuda)
                } else {
                    Err(self.fallback.clone().unwrap_or_else(|| {
                        FallbackRecord::cpu(
                            FallbackReason::DeviceUnavailable,
                            "GPU was required but is not usable",
                        )
                    }))
                }
            }
        }
    }
}

/// Probe this process, including driver/device checks and required PTX JIT.
///
/// A CUDA backend is reported usable only after constructing an accelerator
/// has verified every required kernel family.
#[must_use]
pub fn probe_capabilities() -> CapabilityReport {
    crate::GpuAccelerator::new().capabilities().clone()
}

/// Pure decision function over [`CapabilityFacts`] (mocked tests welcome).
#[must_use]
pub fn evaluate_capabilities(facts: &CapabilityFacts) -> CapabilityReport {
    let (selected_backend, fallback) = select_from_facts(facts);
    CapabilityReport {
        cuda_built: facts.cuda_built,
        runtime_available: facts.runtime_available,
        device_available: facts.device_available,
        compute_capability: facts.compute_capability,
        kernels: facts.kernels,
        selected_backend,
        fallback,
    }
}

fn select_from_facts(facts: &CapabilityFacts) -> (Backend, Option<FallbackRecord>) {
    if !facts.cuda_built {
        return cpu(
            FallbackReason::CudaFeatureNotBuilt,
            "crate built without the cuda feature",
        );
    }
    if !facts.runtime_available {
        return cpu(
            FallbackReason::DriverRuntimeFailure,
            "CUDA driver or runtime init failed",
        );
    }
    if !facts.device_available {
        return cpu(
            FallbackReason::DeviceUnavailable,
            "no CUDA device is accessible",
        );
    }
    match facts.compute_capability {
        None => {
            return cpu(
                FallbackReason::DriverRuntimeFailure,
                "compute capability query failed",
            );
        }
        Some(cc) if !cc.meets_minimum() => {
            return cpu(
                FallbackReason::UnsupportedHardware,
                format!(
                    "compute capability {cc} is below required {}",
                    ComputeCapability::REQUIRED
                ),
            );
        }
        Some(_) => {}
    }
    if !facts.kernels.compiled {
        return cpu(
            FallbackReason::CudaFeatureNotBuilt,
            "required PTX families were not compiled into this binary",
        );
    }
    if !facts.kernels.all_runtime_available() {
        return cpu(
            FallbackReason::KernelSpecializationUnavailable,
            "required kernel specialization is unavailable",
        );
    }
    (Backend::Cuda, None)
}

fn cpu(reason: FallbackReason, detail: impl Into<String>) -> (Backend, Option<FallbackRecord>) {
    (Backend::Cpu, Some(FallbackRecord::cpu(reason, detail)))
}

/// Strip absolute user paths and obvious secret assignments from diagnostics.
#[must_use]
pub fn sanitize_diagnostic(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }
    input
        .split_whitespace()
        .map(sanitize_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn sanitize_token(tok: &str) -> String {
    let (prefix, core, suffix) = split_wrapping_punct(tok);
    let lower = core.to_ascii_lowercase();
    if is_user_path(&lower) {
        return format!("{prefix}<path>{suffix}");
    }
    if let Some((key, value)) = lower.split_once('=') {
        let orig_key = core.split_once('=').map(|(k, _)| k).unwrap_or(core);
        if is_secret_key(key) {
            return format!("{prefix}{orig_key}=<redacted>{suffix}");
        }
        if is_user_path(value) {
            return format!("{prefix}{orig_key}=<path>{suffix}");
        }
    }
    tok.to_string()
}

fn split_wrapping_punct(tok: &str) -> (&str, &str, &str) {
    let prefix_len: usize = tok
        .chars()
        .take_while(|c| matches!(c, '(' | '[' | '{' | '"' | '\''))
        .map(char::len_utf8)
        .sum();
    let suffix_len: usize = tok
        .chars()
        .rev()
        .take_while(|c| matches!(c, ')' | ']' | '}' | '"' | '\'' | ',' | ';' | '.' | ':'))
        .map(char::len_utf8)
        .sum();
    if prefix_len + suffix_len >= tok.len() {
        return ("", tok, "");
    }
    let core_end = tok.len() - suffix_len;
    (
        &tok[..prefix_len],
        &tok[prefix_len..core_end],
        &tok[core_end..],
    )
}

fn is_user_path(lower: &str) -> bool {
    lower.starts_with("/home/")
        || lower.starts_with("/users/")
        || lower.starts_with("/root/")
        || lower.starts_with("/tmp/")
        || lower.starts_with("/var/folders/")
        || lower.contains("/.ssh/")
        || lower.contains("/.aws/")
        || (lower.len() >= 3
            && lower.as_bytes()[1] == b':'
            && (lower.as_bytes()[2] == b'\\' || lower.as_bytes()[2] == b'/')
            && (lower.contains("\\users\\") || lower.contains("/users/")))
}

fn is_secret_key(key: &str) -> bool {
    matches!(
        key,
        "token"
            | "password"
            | "secret"
            | "authorization"
            | "api_key"
            | "credential"
            | "access_key"
            | "secret_key"
    ) || key.ends_with("token")
        || key.ends_with("password")
        || key.ends_with("secret")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case {
        name: &'static str,
        facts: CapabilityFacts,
        backend: Backend,
        reason: Option<FallbackReason>,
    }

    fn facts(
        cuda_built: bool,
        runtime: bool,
        device: bool,
        cc: Option<ComputeCapability>,
        kernels: KernelAvailability,
    ) -> CapabilityFacts {
        CapabilityFacts {
            cuda_built,
            runtime_available: runtime,
            device_available: device,
            compute_capability: cc,
            kernels,
        }
    }

    #[test]
    fn decision_table_covers_build_runtime_device_kernel_outcomes() {
        let sm89 = ComputeCapability { major: 8, minor: 9 };
        let sm120 = ComputeCapability {
            major: 12,
            minor: 0,
        };
        let sm121 = ComputeCapability {
            major: 12,
            minor: 1,
        };

        let cases = [
            Case {
                name: "cpu-only build",
                facts: CapabilityFacts::not_built(),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::CudaFeatureNotBuilt),
            },
            Case {
                name: "driver/runtime failure",
                facts: facts(
                    true,
                    false,
                    false,
                    None,
                    KernelAvailability::compiled_unverified(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::DriverRuntimeFailure),
            },
            Case {
                name: "device unavailable",
                facts: facts(
                    true,
                    true,
                    false,
                    None,
                    KernelAvailability::compiled_unverified(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::DeviceUnavailable),
            },
            Case {
                name: "CC query failed",
                facts: facts(
                    true,
                    true,
                    true,
                    None,
                    KernelAvailability::compiled_unverified(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::DriverRuntimeFailure),
            },
            Case {
                name: "unsupported hardware",
                facts: facts(
                    true,
                    true,
                    true,
                    Some(sm89),
                    KernelAvailability::compiled_unverified(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::UnsupportedHardware),
            },
            Case {
                name: "kernel specialization missing",
                facts: facts(
                    true,
                    true,
                    true,
                    Some(sm120),
                    KernelAvailability::all_unavailable(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::KernelSpecializationUnavailable),
            },
            Case {
                name: "compiled but JIT not attempted (probe)",
                facts: facts(
                    true,
                    true,
                    true,
                    Some(sm120),
                    KernelAvailability::compiled_unverified(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::KernelSpecializationUnavailable),
            },
            Case {
                name: "all kernels available",
                facts: facts(
                    true,
                    true,
                    true,
                    Some(sm121),
                    KernelAvailability::all_available(),
                ),
                backend: Backend::Cuda,
                reason: None,
            },
            Case {
                name: "PTX not compiled despite cuda_built flag",
                facts: facts(
                    true,
                    true,
                    true,
                    Some(sm120),
                    KernelAvailability::not_compiled(),
                ),
                backend: Backend::Cpu,
                reason: Some(FallbackReason::CudaFeatureNotBuilt),
            },
        ];

        for case in cases {
            let report = evaluate_capabilities(&case.facts);
            assert_eq!(
                report.selected_backend, case.backend,
                "{}: backend",
                case.name
            );
            assert_eq!(
                report.fallback.as_ref().map(|f| f.reason),
                case.reason,
                "{}: reason",
                case.name
            );
            if let Some(fb) = &report.fallback {
                assert_eq!(fb.selected_backend, Backend::Cpu, "{}", case.name);
                assert_eq!(fb.reason.code(), fb.reason.to_string());
                assert!(!fb.detail.contains("/home/"), "{}", case.name);
            } else {
                assert!(report.gpu_usable(), "{}", case.name);
            }
        }
    }

    #[test]
    fn require_gpu_policy_fails_closed_on_cpu_selection() {
        let report = evaluate_capabilities(&CapabilityFacts::not_built());
        let err = report
            .select_backend(ExecutionPolicy::RequireGpu)
            .unwrap_err();
        assert_eq!(err.reason, FallbackReason::CudaFeatureNotBuilt);
        assert_eq!(err.selected_backend, Backend::Cpu);
        assert_eq!(
            report.select_backend(ExecutionPolicy::PreferGpu).unwrap(),
            Backend::Cpu
        );
    }

    #[test]
    fn require_gpu_policy_accepts_cuda_selection() {
        let report = evaluate_capabilities(&facts(
            true,
            true,
            true,
            Some(ComputeCapability::REQUIRED),
            KernelAvailability::all_available(),
        ));
        assert_eq!(
            report.select_backend(ExecutionPolicy::RequireGpu).unwrap(),
            Backend::Cuda
        );
    }

    #[test]
    fn fallback_reason_codes_are_stable() {
        let expected = [
            (
                FallbackReason::CudaFeatureNotBuilt,
                "cuda_feature_not_built",
            ),
            (
                FallbackReason::DriverRuntimeFailure,
                "driver_runtime_failure",
            ),
            (FallbackReason::DeviceUnavailable, "device_unavailable"),
            (FallbackReason::UnsupportedHardware, "unsupported_hardware"),
            (
                FallbackReason::KernelSpecializationUnavailable,
                "kernel_specialization_unavailable",
            ),
            (FallbackReason::InvalidInput, "invalid_input"),
        ];
        for (reason, code) in expected {
            assert_eq!(reason.code(), code);
            assert_eq!(reason.to_string(), code);
        }
    }

    #[test]
    fn compute_capability_floor_is_sm_120() {
        assert!(!ComputeCapability { major: 8, minor: 9 }.meets_minimum());
        assert!(
            !ComputeCapability {
                major: 11,
                minor: 0
            }
            .meets_minimum()
        );
        assert!(
            ComputeCapability {
                major: 12,
                minor: 0
            }
            .meets_minimum()
        );
        assert!(
            ComputeCapability {
                major: 12,
                minor: 1
            }
            .meets_minimum()
        );
        assert!(
            ComputeCapability {
                major: 13,
                minor: 0
            }
            .meets_minimum()
        );
    }

    #[test]
    fn sanitize_diagnostic_redacts_user_paths_and_secrets() {
        let raw = "failed /home/alice/.ssh/id_rsa token=supersecret C:\\Users\\bob\\key.pem ok";
        let clean = sanitize_diagnostic(raw);
        assert!(!clean.contains("/home/alice"));
        assert!(!clean.contains("supersecret"));
        assert!(!clean.contains("bob"));
        assert!(clean.contains("<path>"));
        assert!(clean.contains("token=<redacted>"));
        assert!(clean.contains("ok"));
    }

    #[test]
    fn sanitize_diagnostic_keeps_stable_reason_tokens() {
        let s = sanitize_diagnostic("cuda_feature_not_built sm_120 driver_runtime_failure");
        assert_eq!(s, "cuda_feature_not_built sm_120 driver_runtime_failure");
    }

    #[test]
    fn sanitize_diagnostic_redacts_paths_after_assignment_keys() {
        let clean = sanitize_diagnostic(
            "path=/home/alice/private.ptx file=/Users/bob/key cache=C:\\Users\\eve\\cache.bin",
        );
        assert_eq!(clean, "path=<path> file=<path> cache=<path>");
    }
}
