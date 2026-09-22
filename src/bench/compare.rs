// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in baseline comparison: fail only when relative *and* absolute budgets trip.

use crate::bench::stats::{SampleStats, sample_stats};
use serde::{Deserialize, Serialize};

/// How a case is classified against a baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionClass {
    Pass,
    Fail,
    Noisy,
    InsufficientSamples,
}

/// Budgets that must *both* be exceeded for [`RegressionClass::Fail`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionBudget {
    /// Relative median increase (0.10 = 10%).
    pub relative: f64,
    /// Minimum absolute median increase in microseconds.
    pub min_absolute_us: f64,
    /// Both sides must have at least this many samples.
    pub min_samples: usize,
    /// Relative MAD (`mad/median`) above this → [`RegressionClass::Noisy`].
    pub noisy_relative_dispersion: f64,
}

impl Default for RegressionBudget {
    fn default() -> Self {
        Self {
            relative: 0.10,
            min_absolute_us: 2.0,
            min_samples: 8,
            noisy_relative_dispersion: 0.25,
        }
    }
}

/// Samples or pre-aggregated stats used as one side of a comparison.
#[derive(Clone, Debug)]
pub enum SampleSource<'a> {
    Samples(&'a [f64]),
    Stats(SampleStats),
    /// Pre-aggregated statistics from a format that did not record dispersion.
    StatsWithoutDispersion(SampleStats),
}

impl SampleSource<'_> {
    fn stats(&self) -> (SampleStats, Option<f64>) {
        match self {
            SampleSource::Samples(samples) => {
                let stats = sample_stats(samples);
                let dispersion = stats.relative_dispersion;
                (stats, Some(dispersion))
            }
            SampleSource::Stats(stats) => {
                let dispersion = stats.relative_dispersion;
                (stats.clone(), Some(dispersion))
            }
            SampleSource::StatsWithoutDispersion(stats) => (stats.clone(), None),
        }
    }
}

/// One named comparison row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonCase {
    pub name: String,
    pub class: RegressionClass,
    pub baseline_n: usize,
    pub current_n: usize,
    pub baseline_median_us: f64,
    pub current_median_us: f64,
    /// Relative change from the baseline, or `None` when the baseline median is zero.
    pub relative_delta: Option<f64>,
    pub absolute_delta_us: f64,
    pub baseline_relative_dispersion: Option<f64>,
    pub current_relative_dispersion: Option<f64>,
}

/// Why a baseline/current row could not be compared safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonRejectionReason {
    BaselineReadFailure,
    BaselineParseFailure,
    MissingCurrentCase,
    BuildProfileMismatch,
    HardwareIdentityMismatch,
    PowerClockMismatch,
    HostIdentityMismatch,
    FeatureSetMismatch,
    WarmupMismatch,
    WorkloadMetadataMismatch,
    DuplicateBaselineName,
    DuplicateCurrentName,
    NoComparableCases,
}

/// A comparison input rejected before statistical classification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonRejection {
    pub reason: ComparisonRejectionReason,
    pub case_name: Option<String>,
}

/// Versioned comparison report written beside benchmark output when a baseline is supplied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub schema_version: u32,
    pub budget: RegressionBudget,
    pub enforced: bool,
    /// Whether the requested enforcement gate accepted the complete comparison.
    pub gate_passed: bool,
    /// Inputs that could not be represented as classified comparison rows.
    pub rejections: Vec<ComparisonRejection>,
    pub cases: Vec<ComparisonCase>,
}

impl ComparisonReport {
    /// True when any classified case cannot pass an enforced gate.
    pub fn has_failure(&self) -> bool {
        self.cases.iter().any(|c| {
            matches!(
                c.class,
                RegressionClass::Fail | RegressionClass::InsufficientSamples
            )
        })
    }
}

/// Classify `current` against `baseline` using `budget`.
///
/// Order: insufficient samples → noisy → fail (both budgets, regression) → pass.
pub fn classify(
    baseline: SampleSource<'_>,
    current: SampleSource<'_>,
    budget: &RegressionBudget,
) -> RegressionClass {
    compare_one("case", baseline, current, budget).class
}

/// Wrap classified rows into a versioned comparison report.
pub fn comparison_report(
    cases: Vec<ComparisonCase>,
    budget: RegressionBudget,
    enforced: bool,
    mut rejections: Vec<ComparisonRejection>,
) -> ComparisonReport {
    if cases.is_empty()
        && !rejections
            .iter()
            .any(|rejection| rejection.reason == ComparisonRejectionReason::NoComparableCases)
    {
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::NoComparableCases,
            case_name: None,
        });
    }
    let has_failure = cases.iter().any(|case| {
        matches!(
            case.class,
            RegressionClass::Fail | RegressionClass::InsufficientSamples
        )
    });
    let gate_passed = !enforced || (rejections.is_empty() && !has_failure);
    ComparisonReport {
        schema_version: 1,
        budget,
        enforced,
        gate_passed,
        rejections,
        cases,
    }
}

pub fn compare_one(
    name: &str,
    baseline: SampleSource<'_>,
    current: SampleSource<'_>,
    budget: &RegressionBudget,
) -> ComparisonCase {
    let (b, baseline_relative_dispersion) = baseline.stats();
    let (c, current_relative_dispersion) = current.stats();
    let absolute_delta_us = c.median - b.median;
    let relative_delta = if b.median.abs() > 1e-12 {
        Some(absolute_delta_us / b.median.abs())
    } else {
        None
    };
    let relative_budget_exceeded = relative_delta
        .map(|delta| delta > budget.relative)
        .unwrap_or(absolute_delta_us > 0.0);

    let class = if b.n < budget.min_samples
        || c.n < budget.min_samples
        || baseline_relative_dispersion.is_none()
        || current_relative_dispersion.is_none()
    {
        RegressionClass::InsufficientSamples
    } else if baseline_relative_dispersion
        .is_some_and(|value| finite_or_inf(value) > budget.noisy_relative_dispersion)
        || current_relative_dispersion
            .is_some_and(|value| finite_or_inf(value) > budget.noisy_relative_dispersion)
    {
        RegressionClass::Noisy
    } else if relative_budget_exceeded && absolute_delta_us > budget.min_absolute_us {
        RegressionClass::Fail
    } else {
        RegressionClass::Pass
    };

    ComparisonCase {
        name: name.to_string(),
        class,
        baseline_n: b.n,
        current_n: c.n,
        baseline_median_us: b.median,
        current_median_us: c.median,
        relative_delta,
        absolute_delta_us,
        baseline_relative_dispersion,
        current_relative_dispersion,
    }
}

fn finite_or_inf(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::INFINITY }
}

/// Parse `MYELIN_BENCH_ENFORCE_BUDGET` / CLI values. Unset and common falsy tokens are off.
pub fn parse_enforce_flag(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
    }
}

/// Read the opt-in hardware-enforcement environment flag.
pub fn enforce_budget_requested() -> bool {
    parse_enforce_flag(std::env::var("MYELIN_BENCH_ENFORCE_BUDGET").ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n_copies(v: f64, n: usize) -> Vec<f64> {
        vec![v; n]
    }

    fn budget() -> RegressionBudget {
        RegressionBudget::default()
    }

    #[test]
    fn pass_when_change_is_small() {
        let base = n_copies(100.0, 8);
        let cur = n_copies(101.0, 8);
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::Pass
        );
    }

    #[test]
    fn fail_when_relative_and_absolute_both_exceeded() {
        let base = n_copies(100.0, 8);
        let cur = n_copies(130.0, 8);
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::Fail
        );
    }

    #[test]
    fn pass_when_only_relative_exceeded() {
        let mut b = budget();
        b.min_absolute_us = 25.0;
        let base = n_copies(100.0, 8);
        let cur = n_copies(120.0, 8); // +20% and +20us
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &b
            ),
            RegressionClass::Pass
        );
    }

    #[test]
    fn pass_when_only_absolute_exceeded() {
        let base = n_copies(100.0, 8);
        let cur = n_copies(104.0, 8); // +4% and +4us; relative budget is 10%
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::Pass
        );
    }

    #[test]
    fn improvements_are_pass() {
        let base = n_copies(100.0, 8);
        let cur = n_copies(50.0, 8);
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::Pass
        );
    }

    #[test]
    fn noisy_when_dispersion_is_high() {
        let base = n_copies(100.0, 8);
        let cur = [10.0, 20.0, 50.0, 80.0, 100.0, 250.0, 300.0, 400.0];
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::Noisy
        );
    }

    #[test]
    fn insufficient_samples() {
        let base = n_copies(100.0, 3);
        let cur = n_copies(200.0, 3);
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::InsufficientSamples
        );
    }

    #[test]
    fn noisy_takes_precedence_over_fail() {
        let base = [10.0, 20.0, 50.0, 80.0, 100.0, 250.0, 300.0, 400.0];
        let cur = n_copies(500.0, 8);
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            RegressionClass::Noisy
        );
    }

    #[test]
    fn classification_is_order_independent() {
        let base = [100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 100.0, 100.0];
        let mut shuffled = base;
        shuffled.reverse();
        let cur = n_copies(100.0, 8);
        assert_eq!(
            classify(
                SampleSource::Samples(&base),
                SampleSource::Samples(&cur),
                &budget()
            ),
            classify(
                SampleSource::Samples(&shuffled),
                SampleSource::Samples(&cur),
                &budget()
            )
        );
    }

    #[test]
    fn zero_baseline_relative_delta_roundtrips_as_undefined() {
        let baseline = n_copies(0.0, 8);
        let current = n_copies(10.0, 8);
        let row = compare_one(
            "zero-baseline",
            SampleSource::Samples(&baseline),
            SampleSource::Samples(&current),
            &budget(),
        );

        assert_eq!(row.class, RegressionClass::Fail);
        assert_eq!(row.relative_delta, None);
        let json = serde_json::to_string(&row).expect("serialize comparison row");
        assert!(json.contains("\"relative_delta\":null"));
        let parsed: ComparisonCase =
            serde_json::from_str(&json).expect("deserialize comparison row");
        assert_eq!(parsed, row);
    }

    #[test]
    fn zero_to_zero_relative_delta_is_undefined_but_passes() {
        let baseline = n_copies(0.0, 8);
        let current = n_copies(0.0, 8);
        let row = compare_one(
            "zero-to-zero",
            SampleSource::Samples(&baseline),
            SampleSource::Samples(&current),
            &budget(),
        );

        assert_eq!(row.class, RegressionClass::Pass);
        assert_eq!(row.relative_delta, None);
        let json = serde_json::to_string(&row).expect("serialize comparison row");
        let parsed: ComparisonCase =
            serde_json::from_str(&json).expect("deserialize comparison row");
        assert_eq!(parsed, row);
    }

    #[test]
    fn missing_baseline_dispersion_is_insufficient_not_a_failure() {
        let baseline = SampleStats {
            n: 8,
            mean: 100.0,
            median: 100.0,
            mad: 0.0,
            relative_dispersion: 0.0,
            min: 90.0,
            max: 110.0,
            p50: 100.0,
            p95: 110.0,
            p99: 110.0,
        };
        let current = n_copies(130.0, 8);

        let row = compare_one(
            "legacy",
            SampleSource::StatsWithoutDispersion(baseline),
            SampleSource::Samples(&current),
            &budget(),
        );

        assert_eq!(row.class, RegressionClass::InsufficientSamples);
        assert_eq!(row.baseline_relative_dispersion, None);
    }

    #[test]
    fn insufficient_samples_are_an_enforcement_failure() {
        let baseline = n_copies(100.0, 8);
        let current = n_copies(130.0, 1);
        let row = compare_one(
            "under-sampled",
            SampleSource::Samples(&baseline),
            SampleSource::Samples(&current),
            &budget(),
        );
        let report = comparison_report(vec![row], budget(), true, Vec::new());

        assert!(report.has_failure());
        assert!(!report.gate_passed);
    }

    #[test]
    fn enforced_empty_report_is_rejected_by_constructor() {
        let report = comparison_report(Vec::new(), budget(), true, Vec::new());

        assert!(!report.gate_passed);
        assert_eq!(
            report.rejections,
            vec![ComparisonRejection {
                reason: ComparisonRejectionReason::NoComparableCases,
                case_name: None,
            }]
        );
    }
}
