// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public capability-probe contract.
//!
//! CPU-only builds always report `cuda_feature_not_built`. CUDA builds are
//! capability-gated: real-device assertions run only when the probe says GPU
//! is usable.

use myelin_accelerator::{
    Backend, CapabilityFacts, ComputeCapability, ExecutionPolicy, FallbackReason, GpuAccelerator,
    GpuError, KernelAvailability, evaluate_capabilities, probe_capabilities, sanitize_diagnostic,
};

#[test]
fn probe_returns_typed_report_not_log_text() {
    let report = probe_capabilities();
    let _ = report.cuda_built;
    let _ = report.runtime_available;
    let _ = report.device_available;
    let _ = report.compute_capability;
    let _ = report.kernels;
    let _ = report.selected_backend;
    match report.fallback {
        None => assert_eq!(report.selected_backend, Backend::Cuda),
        Some(fb) => {
            assert_eq!(fb.selected_backend, Backend::Cpu);
            assert!(!fb.reason.code().is_empty());
            assert!(!fb.detail.contains("/home/"));
            assert!(!fb.detail.contains("/Users/"));
        }
    }
}

#[test]
fn require_gpu_never_returns_cpu_backend() {
    match GpuAccelerator::require_gpu() {
        Ok(acc) => {
            assert_eq!(acc.selected_backend(), Backend::Cuda);
            assert!(acc.is_ready());
            assert!(acc.fallback().is_none());
            assert!(acc.capabilities().gpu_usable());
        }
        Err(err) => match err {
            GpuError::Unavailable { reason, detail } => {
                assert_ne!(reason, FallbackReason::InvalidInput);
                assert!(!detail.contains("/home/"));
            }
            other => panic!("require_gpu must fail as Unavailable, got {other}"),
        },
    }
}

#[test]
fn prefer_gpu_records_fallback_when_cpu_is_selected() {
    let acc = GpuAccelerator::with_policy(ExecutionPolicy::PreferGpu).unwrap();
    if acc.selected_backend() == Backend::Cpu {
        let fb = acc
            .fallback()
            .expect("CPU selection must record a fallback");
        assert_eq!(fb.selected_backend, Backend::Cpu);
        assert!(!acc.is_ready());
        let launch_err = acc.synchronize().unwrap_err();
        assert_eq!(launch_err.fallback_reason(), Some(fb.reason));
    } else {
        assert!(acc.is_ready());
        assert!(acc.fallback().is_none());
    }
}

#[test]
fn mocked_capability_facts_drive_the_decision_table() {
    let cpu = evaluate_capabilities(&CapabilityFacts::not_built());
    assert_eq!(cpu.selected_backend, Backend::Cpu);
    assert_eq!(
        cpu.fallback.as_ref().map(|f| f.reason),
        Some(FallbackReason::CudaFeatureNotBuilt)
    );

    let gpu = evaluate_capabilities(&CapabilityFacts {
        cuda_built: true,
        runtime_available: true,
        device_available: true,
        compute_capability: Some(ComputeCapability {
            major: 12,
            minor: 0,
        }),
        kernels: KernelAvailability::all_available(),
    });
    assert!(gpu.gpu_usable());
    assert_eq!(gpu.selected_backend, Backend::Cuda);
}

#[test]
fn diagnostics_never_include_user_paths() {
    let clean = sanitize_diagnostic("init failed /home/eve/secret token=abc123");
    assert!(!clean.contains("/home/eve"));
    assert!(!clean.contains("abc123"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_only_build_is_not_built() {
    let report = probe_capabilities();
    assert!(!report.cuda_built);
    assert!(!report.kernels.compiled);
    assert_eq!(
        report.fallback.as_ref().map(|f| f.reason),
        Some(FallbackReason::CudaFeatureNotBuilt)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_build_reports_compiled_and_gates_device_assertions() {
    let report = probe_capabilities();
    assert!(report.cuda_built);
    assert!(report.kernels.compiled);
    if report.gpu_usable() {
        let acc = match GpuAccelerator::require_gpu() {
            Ok(acc) => acc,
            Err(err) => panic!("probe said GPU is usable: {err}"),
        };
        assert_eq!(acc.selected_backend(), Backend::Cuda);
        let cc = acc
            .capabilities()
            .compute_capability
            .expect("usable GPU reports compute capability");
        assert!(cc.meets_minimum());
    } else {
        assert_eq!(report.selected_backend, Backend::Cpu);
        let fb = report
            .fallback
            .as_ref()
            .expect("unusable GPU records a reason");
        assert!(matches!(
            fb.reason,
            FallbackReason::DriverRuntimeFailure
                | FallbackReason::DeviceUnavailable
                | FallbackReason::UnsupportedHardware
                | FallbackReason::KernelSpecializationUnavailable
                | FallbackReason::StreamCreationFailure
        ));
        assert!(GpuAccelerator::require_gpu().is_err());
    }
}
