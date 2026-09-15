// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Median / dispersion statistics for benchmark samples.

/// Aggregates computed from a sample of latencies (microseconds).
#[derive(Clone, Debug, PartialEq)]
pub struct SampleStats {
    /// Number of samples.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median (even-n: average of the two central values).
    pub median: f64,
    /// Median absolute deviation from the median.
    pub mad: f64,
    /// `mad / median` when `|median|` is meaningful; otherwise 0.
    pub relative_dispersion: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Compute [`SampleStats`] from latency samples in microseconds.
///
/// Empty input yields zeros. Samples are sorted internally; caller order does
/// not affect the result.
pub fn sample_stats(samples: &[f64]) -> SampleStats {
    if samples.is_empty() {
        return SampleStats {
            n: 0,
            mean: 0.0,
            median: 0.0,
            mad: 0.0,
            relative_dispersion: 0.0,
            min: 0.0,
            max: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
        };
    }

    let n = samples.len();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let total: f64 = sorted.iter().sum();
    let mean = total / n as f64;
    let median = median_of_sorted(&sorted);
    let mut absdevs: Vec<f64> = sorted.iter().map(|x| (x - median).abs()).collect();
    absdevs.sort_by(|a, b| a.total_cmp(b));
    let mad = median_of_sorted(&absdevs);
    let relative_dispersion = if median.abs() > 1e-12 {
        mad / median.abs()
    } else if mad > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    SampleStats {
        n,
        mean,
        median,
        mad,
        relative_dispersion,
        min: sorted[0],
        max: sorted[n - 1],
        p50: percentile_nearest_rank(&sorted, 50.0),
        p95: percentile_nearest_rank(&sorted, 95.0),
        p99: percentile_nearest_rank(&sorted, 99.0),
    }
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Nearest-rank percentile: rank = ceil(p/100 * N), then 0-based index.
fn percentile_nearest_rank(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stats_are_zero() {
        let s = sample_stats(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.median, 0.0);
        assert_eq!(s.mad, 0.0);
    }

    #[test]
    fn odd_and_even_median() {
        assert_eq!(sample_stats(&[3.0, 1.0, 2.0]).median, 2.0);
        assert_eq!(sample_stats(&[4.0, 1.0, 2.0, 3.0]).median, 2.5);
    }

    #[test]
    fn order_does_not_matter() {
        let a = sample_stats(&[10.0, 20.0, 30.0, 40.0]);
        let b = sample_stats(&[40.0, 10.0, 30.0, 20.0]);
        assert_eq!(a, b);
    }

    #[test]
    fn mad_of_constant_series_is_zero() {
        let s = sample_stats(&[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(s.median, 5.0);
        assert_eq!(s.mad, 0.0);
        assert_eq!(s.relative_dispersion, 0.0);
    }
}
