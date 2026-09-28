// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consumer-installed launch-failure reporter.
//!
//! This crate does **not** depend on Sentry (or any other telemetry SDK).
//! A consumer such as `corinth-canal` installs a hook that forwards
//! [`LaunchFailure`] into its own reporter.

use std::sync::{Arc, RwLock};

/// How the failed kernel was launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchType {
    /// Fatbin/SASS load or `cust::launch!` PTX path.
    PtxFatbin,
    /// Blackwell-critical F16 GIF / SAAQ path via the C ABI shim.
    CAbiShim,
}

impl LaunchType {
    /// Stable tag for consumer telemetry (`ptx_fatbin` / `c_abi_shim`).
    pub fn as_str(self) -> &'static str {
        match self {
            LaunchType::PtxFatbin => "ptx_fatbin",
            LaunchType::CAbiShim => "c_abi_shim",
        }
    }
}

/// Structured context for a CUDA module-load or kernel-launch failure.
#[derive(Debug, Clone)]
pub struct LaunchFailure {
    /// Kernel or module name (e.g. `gif_step_weighted`, `spiking_network`).
    pub kernel_name: String,
    /// Launch mechanism.
    pub launch_type: LaunchType,
    /// Grid dimensions `(x, y, z)`. All zeros for module-load failures.
    pub grid: (u32, u32, u32),
    /// Block dimensions `(x, y, z)`. All zeros for module-load failures.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub shared_mem: u32,
    /// Neuron count when the failure is on a temporal kernel.
    pub neuron_count: Option<usize>,
    /// Display form of [`crate::GpuError`].
    pub error: String,
    /// `CU_JIT_ERROR_LOG_BUFFER` when the driver returned JIT diagnostics.
    pub jit_error_log: Option<String>,
    /// `CU_JIT_INFO_LOG_BUFFER` when the driver returned JIT diagnostics.
    pub jit_info_log: Option<String>,
}

/// Callback installed by a consumer to observe launch / module-load failures.
pub type LaunchFailureHook = Arc<dyn Fn(&LaunchFailure) + Send + Sync>;

static HOOK: RwLock<Option<LaunchFailureHook>> = RwLock::new(None);

fn hook_lock() -> std::sync::RwLockWriteGuard<'static, Option<LaunchFailureHook>> {
    HOOK.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn hook_read() -> std::sync::RwLockReadGuard<'static, Option<LaunchFailureHook>> {
    HOOK.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Install (or replace) the process-wide launch-failure hook.
///
/// The hook is invoked after a [`crate::GpuError`] is constructed and before
/// it is returned to the caller. It must not panic.
pub fn set_launch_failure_hook(hook: impl Fn(&LaunchFailure) + Send + Sync + 'static) {
    *hook_lock() = Some(Arc::new(hook));
}

/// Remove any installed launch-failure hook.
pub fn clear_launch_failure_hook() {
    *hook_lock() = None;
}

/// Called from CUDA launch wrappers. No-op unless a consumer installed a hook.
///
/// The `Arc` is cloned and the `RwLock` is dropped before the callback runs so
/// a hook may call [`set_launch_failure_hook`] / [`clear_launch_failure_hook`]
/// without deadlocking.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn report_launch_failure(failure: LaunchFailure) {
    let hook = {
        let guard = hook_read();
        guard.clone()
    };
    if let Some(hook) = hook {
        hook(&failure);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn launch_type_tags() {
        assert_eq!(LaunchType::PtxFatbin.as_str(), "ptx_fatbin");
        assert_eq!(LaunchType::CAbiShim.as_str(), "c_abi_shim");
    }

    #[test]
    fn hook_is_invoked_and_can_be_cleared() {
        let seen = Arc::new(Mutex::new(None::<String>));
        let seen_cb = Arc::clone(&seen);
        set_launch_failure_hook(move |failure| {
            *seen_cb.lock().expect("test mutex") = Some(failure.kernel_name.clone());
        });
        report_launch_failure(LaunchFailure {
            kernel_name: "test_kernel".into(),
            launch_type: LaunchType::PtxFatbin,
            grid: (8, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            neuron_count: Some(2048),
            error: "Kernel launch failed: boom".into(),
            jit_error_log: None,
            jit_info_log: None,
        });
        assert_eq!(
            seen.lock().expect("test mutex").as_deref(),
            Some("test_kernel")
        );
        clear_launch_failure_hook();
        report_launch_failure(LaunchFailure {
            kernel_name: "after_clear".into(),
            launch_type: LaunchType::CAbiShim,
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            neuron_count: None,
            error: "nope".into(),
            jit_error_log: None,
            jit_info_log: None,
        });
        assert_eq!(
            seen.lock().expect("test mutex").as_deref(),
            Some("test_kernel")
        );
        let seen_reenter = Arc::clone(&seen);
        set_launch_failure_hook(move |failure| {
            *seen_reenter.lock().expect("test mutex") =
                Some(format!("reenter:{}", failure.kernel_name));
            clear_launch_failure_hook();
        });
        report_launch_failure(LaunchFailure {
            kernel_name: "nested".into(),
            launch_type: LaunchType::CAbiShim,
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            neuron_count: None,
            error: "nested".into(),
            jit_error_log: None,
            jit_info_log: None,
        });
        assert_eq!(
            seen.lock().expect("test mutex").as_deref(),
            Some("reenter:nested")
        );
    }
}
