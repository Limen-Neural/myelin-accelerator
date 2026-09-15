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
}

impl SampleSource<'_> {
    fn stats(&self) -> SampleStats {
        match self {
            SampleSource::Samples(s) => sample_stats(s),
            SampleSource::Stats(s) => s.clone(),
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
    pub relative_delta: f64,
    pub absolute_delta_us: f64,
    pub baseline_relative_dispersion: f64,
    pub current_relative_dispersion: f64,
}

/// Versioned comparison report written beside benchmark output when a baseline is supplied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub schema_version: u32,
    pub budget: RegressionBudget,
    pub enforced: bool,
    pub cases: Vec<ComparisonCase>,
}

impl ComparisonReport {
    /// True when any case is a budget failure.
    pub fn has_failure(&self) -> bool {
        self.cases.iter().any(|c| c.class == RegressionClass::Fail)
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
) -> ComparisonReport {
    ComparisonReport {
        schema_version: 1,
        budget,
        enforced,
        cases,
    }
}

pub fn compare_one(
    name: &str,
    baseline: SampleSource<'_>,
    current: SampleSource<'_>,
    budget: &RegressionBudget,
) -> ComparisonCase {
    let b = baseline.stats();
    let c = current.stats();
    let absolute_delta_us = c.median - b.median;
    let relative_delta = if b.median.abs() > 1e-12 {
        absolute_delta_us / b.median.abs()
    } else if absolute_delta_us.abs() > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let class = if b.n < budget.min_samples || c.n < budget.min_samples {
        RegressionClass::InsufficientSamples
    } else if finite_or_inf(b.relative_dispersion) > budget.noisy_relative_dispersion
        || finite_or_inf(c.relative_dispersion) > budget.noisy_relative_dispersion
    {
        RegressionClass::Noisy
    } else if relative_delta > budget.relative && absolute_delta_us > budget.min_absolute_us {
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
        baseline_relative_dispersion: b.relative_dispersion,
        current_relative_dispersion: c.relative_dispersion,
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
}
