// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reproducible benchmark manifests and opt-in regression budgets.
//!
//! The example harness (`examples/benchmark.rs`) emits a versioned manifest
//! beside its JSON/CSV output. Schema and statistics tests in this module
//! run on the CPU-safe CI path and do not require a GPU.
//!
//! Hardware budget *enforcement* is opt-in (`--enforce-budget` or
//! `MYELIN_BENCH_ENFORCE_BUDGET=1`). Ordinary CI never sets that flag, so a
//! run on different hardware cannot fail the default test suite.

mod compare;
mod manifest;
mod redact;
mod stats;

pub use compare::{
    ComparisonCase, ComparisonRejection, ComparisonRejectionReason, ComparisonReport,
    RegressionBudget, RegressionClass, SampleSource, classify, compare_one, comparison_report,
    enforce_budget_requested, parse_enforce_flag,
};
pub use manifest::{
    BenchmarkManifest, CudaDeviceUuid, DeviceIdentity, GitProvenance, MANIFEST_SCHEMA_VERSION,
    ManifestCase, PowerClockControls, RunTiming, ToolchainInfo, capture_git, capture_toolchain,
    enabled_features, paths_refer_to_same_file, probe_power_clock, write_atomic_bytes,
    write_canonical_json, write_canonical_manifest,
};
pub use redact::{RedactionContext, canonicalize_json_value, redact_and_canonicalize};
pub use stats::{SampleStats, sample_stats};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn enforce_flag_defaults_off() {
        assert!(!parse_enforce_flag(None));
        assert!(!parse_enforce_flag(Some("")));
        assert!(!parse_enforce_flag(Some("0")));
        assert!(!parse_enforce_flag(Some("false")));
        assert!(!parse_enforce_flag(Some("off")));
        assert!(parse_enforce_flag(Some("1")));
        assert!(parse_enforce_flag(Some("true")));
        assert!(parse_enforce_flag(Some("YES")));
        assert!(parse_enforce_flag(Some("on")));
    }

    #[test]
    fn enabled_features_is_sorted() {
        let features = enabled_features();
        let mut sorted = features.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(features, sorted);
    }

    #[test]
    fn canonicalize_sorts_keys_and_redacts() {
        let ctx = RedactionContext {
            home: Some("/home/alice".into()),
            user: Some("alice".into()),
        };
        let value = json!({
            "z": "/home/alice/.cargo/bin/nvcc",
            "a": "hello",
            "nested": { "b": 1, "a": "alice-only-path-skip" },
            "api_token": "super-secret",
        });
        let out = canonicalize_json_value(value, &ctx);
        let obj = out.as_object().expect("object");
        let keys: Vec<_> = obj.keys().cloned().collect();
        assert_eq!(keys, vec!["a", "api_token", "nested", "z"]);
        assert_eq!(obj["z"], json!("$HOME/.cargo/bin/nvcc"));
        assert_eq!(obj["api_token"], json!("$REDACTED"));
        assert_eq!(
            obj["nested"],
            json!({ "a": "alice-only-path-skip", "b": 1 })
        );
    }
}
