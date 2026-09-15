// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-safe schema and regression-classification tests for benchmark manifests.
//!
//! These tests do not require a GPU and must not consult hardware budgets.

use myelin_accelerator::bench::{
    BenchmarkManifest, MANIFEST_SCHEMA_VERSION, RedactionContext, RegressionBudget,
    RegressionClass, SampleSource, canonicalize_json_value, classify, compare_one,
    comparison_report, parse_enforce_flag,
};
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bench")
        .join(name)
}

#[derive(Debug, Deserialize)]
struct ClassificationFixture {
    budget: RegressionBudget,
    cases: Vec<ClassificationCase>,
    report: Value,
}

#[derive(Debug, Deserialize)]
struct ClassificationCase {
    name: String,
    baseline_samples_us: Vec<f64>,
    current_samples_us: Vec<f64>,
    expected_class: RegressionClass,
}

#[test]
fn sanitized_manifest_is_parseable_with_null_device_fields() {
    let raw = std::fs::read_to_string(fixture("manifest.sanitized.json")).unwrap();
    let manifest: BenchmarkManifest = serde_json::from_str(&raw).expect("parse sanitized manifest");
    assert_eq!(manifest.schema_version, MANIFEST_SCHEMA_VERSION);
    assert!(manifest.device.name.is_none());
    assert!(manifest.device.uuid.is_none());
    assert!(manifest.device.compute_capability.is_none());
    assert!(manifest.power_clock.persistence_mode.is_none());
    assert!(manifest.power_clock.graphics_clock_mhz.is_none());
    assert!(!manifest.cases.is_empty());

    let value: Value = serde_json::from_str(&raw).unwrap();
    let dumped = serde_json::to_string(&value).unwrap();
    assert!(
        !dumped.contains("/home/"),
        "sanitized fixture must not contain /home/"
    );
    assert!(
        !dumped.to_ascii_lowercase().contains("alice"),
        "sanitized fixture must not contain a local username"
    );
}

#[test]
fn fixture_classification_is_deterministic() {
    let raw = std::fs::read_to_string(fixture("comparison.fixture.json")).unwrap();
    let fixture: ClassificationFixture =
        serde_json::from_str(&raw).expect("parse comparison fixture");
    let budget = fixture.budget;
    let mut rows = Vec::new();
    for case in &fixture.cases {
        let row = compare_one(
            &case.name,
            SampleSource::Samples(&case.baseline_samples_us),
            SampleSource::Samples(&case.current_samples_us),
            &budget,
        );
        assert_eq!(
            row.class, case.expected_class,
            "class mismatch for {}",
            case.name
        );
        // Re-running with the same samples must not flip the class.
        assert_eq!(
            classify(
                SampleSource::Samples(&case.baseline_samples_us),
                SampleSource::Samples(&case.current_samples_us),
                &budget
            ),
            case.expected_class
        );
        rows.push(row);
    }

    let report = comparison_report(rows, budget, false);
    assert!(!report.enforced);
    assert!(!parse_enforce_flag(None));

    let expected_classes: Vec<_> = fixture.cases.iter().map(|c| c.expected_class).collect();
    let report_classes: Vec<_> = report.cases.iter().map(|c| c.class).collect();
    assert_eq!(report_classes, expected_classes);

    let snapshot = fixture.report;
    let names: Vec<_> = snapshot["cases"]
        .as_array()
        .expect("report.cases")
        .iter()
        .map(|c| c["class"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        names,
        expected_classes
            .iter()
            .map(|c| serde_json::to_value(c)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string())
            .collect::<Vec<_>>()
    );
}

#[test]
fn default_ci_cannot_fail_on_hardware_enforcement() {
    assert!(!parse_enforce_flag(None));
    assert!(!parse_enforce_flag(Some("0")));
}

#[test]
fn canonical_json_never_leaks_home_or_tokens() {
    let ctx = RedactionContext {
        home: Some("/home/alice".into()),
        user: Some("alice".into()),
    };
    let value = serde_json::json!({
        "nvcc": "/home/alice/cuda/bin/nvcc",
        "note": "token=ghp_abcdefghijklmnopqrstuvwxyz012345",
        "z": 1,
        "a": 2,
    });
    let out = canonicalize_json_value(value, &ctx);
    let text = serde_json::to_string(&out).unwrap();
    assert!(!text.contains("alice"));
    assert!(!text.contains("/home/"));
    assert!(!text.contains("ghp_"));
    let keys: Vec<_> = out.as_object().unwrap().keys().cloned().collect();
    assert_eq!(keys, vec!["a", "note", "nvcc", "z"]);
}
