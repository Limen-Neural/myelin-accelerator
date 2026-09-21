// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reproducible GPU benchmark harness for myelin-accelerator kernels.
//!
//! Recording, comparing, and refreshing baselines: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md).
//!
//! # Usage
//!
//! ```bash
//! # CPU-only bitpacking benchmarks
//! cargo run --example benchmark --features bench
//!
//! # GPU kernel benchmarks
//! cargo run --example benchmark --features bench,cuda
//!
//! # Informational compare (does not fail the process)
//! cargo run --example benchmark --features bench -- --baseline path/to/baseline.manifest.json
//!
//! # Opt-in hardware budget enforcement (or MYELIN_BENCH_ENFORCE_BUDGET=1)
//! cargo run --example benchmark --features bench -- \
//!   --baseline path/to/baseline.manifest.json --enforce-budget
//!
//! # Custom iteration counts
//! cargo run --example benchmark --features bench -- --warmup 20 --iterations 200
//! ```
//!
//! # Output
//!
//! Emits `benchmark_results.json`, `benchmark_results.csv`, and
//! `benchmark_results.manifest.json` in the current directory. The manifest is
//! versioned, redacted, and parseable even when optional device fields are
//! missing. Legacy JSON/CSV fields (percentiles, throughput, GPU info) are
//! unchanged.
//!
//! # Nsight Profiling
//!
//! For detailed kernel profiling with Nsight Compute:
//!
//! ```bash
//! ncu --set full -o profile cargo run --example benchmark --features bench,cuda
//! ```
//!
//! For timeline profiling with Nsight Systems:
//!
//! ```bash
//! nsys profile -o timeline cargo run --example benchmark --features bench,cuda
//! ```

use myelin_accelerator::bench::{
    BenchmarkManifest, ComparisonCase, DeviceIdentity, MANIFEST_SCHEMA_VERSION, ManifestCase,
    RedactionContext, RegressionBudget, RegressionClass, SampleSource, SampleStats, compare_one,
    comparison_report, enforce_budget_requested, paths_refer_to_same_file, probe_power_clock,
    redact_and_canonicalize, write_canonical_manifest,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ── CLI argument parsing (minimal, no clap dependency) ──────────────────────

struct Config {
    warmup: usize,
    iterations: usize,
    baseline: Option<String>,
    output_prefix: String,
    enforce_budget: bool,
    budget: RegressionBudget,
}

impl Config {
    // `std::env::args` is a developer-facing CLI helper; the only inputs it
    // reads are local flag values, never secrets or auth material, so the
    // "args should not be used for security operations" lint is a false
    // positive here. Suppressed with nosemgrep / opengrep markers below.
    #[allow(clippy::disallowed_methods)]
    fn from_args() -> Self {
        // nosemgrep: rust.lang.security.args.args -- developer CLI flags only
        let args: Vec<String> = std::env::args().collect();
        let parsed = Self::parse_flags(&args).unwrap_or_else(|| {
            eprintln!("Use --help for usage information.");
            std::process::exit(2);
        });
        Self::validate(parsed)
    }

    fn parse_flags(args: &[String]) -> Option<RawConfig> {
        let mut raw = RawConfig::default();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--warmup" => {
                    raw.warmup = Some(consume_usize(args, &mut i, "--warmup")?);
                }
                "--iterations" | "-n" => {
                    raw.iterations = Some(consume_usize(args, &mut i, "--iterations")?);
                }
                "--baseline" => {
                    raw.baseline = Some(consume_string(args, &mut i, "--baseline")?);
                }
                "--output" | "-o" => {
                    raw.output_prefix = Some(consume_string(args, &mut i, "--output")?);
                }
                "--enforce-budget" => {
                    raw.enforce_budget = true;
                }
                "--budget-relative" => {
                    raw.budget_relative = Some(consume_f64(args, &mut i, "--budget-relative")?);
                }
                "--budget-abs-us" => {
                    raw.budget_abs_us = Some(consume_f64(args, &mut i, "--budget-abs-us")?);
                }
                "--min-samples" => {
                    raw.min_samples = Some(consume_usize(args, &mut i, "--min-samples")?);
                }
                "--noisy-dispersion" => {
                    raw.noisy_dispersion = Some(consume_f64(args, &mut i, "--noisy-dispersion")?);
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown argument: {other}");
                    return None;
                }
            }
            i += 1;
        }
        Some(raw)
    }

    fn validate(raw: RawConfig) -> Self {
        if raw.iterations == Some(0) {
            eprintln!("--iterations must be > 0");
            std::process::exit(1);
        }
        let mut budget = RegressionBudget::default();
        if let Some(relative) = raw.budget_relative {
            require_non_negative_finite("--budget-relative", relative);
            budget.relative = relative;
        }
        if let Some(abs_us) = raw.budget_abs_us {
            require_non_negative_finite("--budget-abs-us", abs_us);
            budget.min_absolute_us = abs_us;
        }
        if let Some(min_samples) = raw.min_samples {
            if min_samples == 0 {
                eprintln!("--min-samples must be > 0");
                std::process::exit(1);
            }
            budget.min_samples = min_samples;
        }
        if let Some(noisy) = raw.noisy_dispersion {
            require_non_negative_finite("--noisy-dispersion", noisy);
            budget.noisy_relative_dispersion = noisy;
        }
        Config {
            warmup: raw.warmup.unwrap_or(10),
            iterations: raw.iterations.unwrap_or(100),
            baseline: raw.baseline,
            output_prefix: raw
                .output_prefix
                .unwrap_or_else(|| "benchmark_results".to_string()),
            enforce_budget: raw.enforce_budget || enforce_budget_requested(),
            budget,
        }
    }
}

#[derive(Default)]
struct RawConfig {
    warmup: Option<usize>,
    iterations: Option<usize>,
    baseline: Option<String>,
    output_prefix: Option<String>,
    enforce_budget: bool,
    budget_relative: Option<f64>,
    budget_abs_us: Option<f64>,
    min_samples: Option<usize>,
    noisy_dispersion: Option<f64>,
}

fn consume_string(args: &[String], i: &mut usize, flag: &str) -> Option<String> {
    *i += 1;
    if *i >= args.len() {
        eprintln!("{flag} requires a value");
        return None;
    }
    Some(args[*i].clone())
}

fn consume_usize(args: &[String], i: &mut usize, flag: &str) -> Option<usize> {
    let raw = consume_string(args, i, flag)?;
    match raw.parse() {
        Ok(n) => Some(n),
        Err(_) => {
            eprintln!("{flag} requires a non-negative integer, got \"{raw}\"");
            None
        }
    }
}

fn require_non_negative_finite(flag: &str, value: f64) {
    if !value.is_finite() || value < 0.0 {
        eprintln!("{flag} must be a finite number >= 0, got {value}");
        std::process::exit(1);
    }
}

fn consume_f64(args: &[String], i: &mut usize, flag: &str) -> Option<f64> {
    let raw = consume_string(args, i, flag)?;
    match raw.parse() {
        Ok(n) => Some(n),
        Err(_) => {
            eprintln!("{flag} requires a number, got \"{raw}\"");
            None
        }
    }
}

fn print_help() {
    let budget = RegressionBudget::default();
    println!("Usage: benchmark [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --warmup <N>             Warmup iterations (default: 10)");
    println!("  --iterations <N>         Timed iterations (default: 100)");
    println!("  --baseline <FILE>        Compare against a previous JSON or manifest");
    println!("  --output <PREFIX>        Output file prefix (default: benchmark_results)");
    println!("  --enforce-budget         Exit 1 on Fail (also MYELIN_BENCH_ENFORCE_BUDGET=1)");
    println!(
        "  --budget-relative <F>    Relative median budget (default: {})",
        budget.relative
    );
    println!(
        "  --budget-abs-us <F>      Min absolute median delta µs (default: {})",
        budget.min_absolute_us
    );
    println!(
        "  --min-samples <N>        Min samples per side (default: {})",
        budget.min_samples
    );
    println!(
        "  --noisy-dispersion <F>   Relative MAD above this is noisy (default: {})",
        budget.noisy_relative_dispersion
    );
    println!("  -h, --help               Show this help");
    println!();
    println!("Baseline files are read-only. Refresh by copying a new manifest into git.");
    println!("See docs/BENCHMARKS.md.");
}

// ── Benchmark result types ──────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct BenchmarkResult {
    name: String,
    iterations: usize,
    total_duration_us: f64,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    min_us: f64,
    max_us: f64,
    throughput_ops_per_sec: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct BenchmarkReport {
    timestamp: String,
    gpu_info: Option<GpuInfo>,
    config: RunConfig,
    results: Vec<BenchmarkResult>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct GpuInfo {
    device_name: String,
    driver_version: String,
    cuda_version: String,
    sm_arch: String,
    vram_total_mb: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct RunConfig {
    warmup: usize,
    iterations: usize,
}

// ── Benchmark runner ────────────────────────────────────────────────────────

struct Capture {
    result: BenchmarkResult,
    case: ManifestCase,
}

#[allow(clippy::too_many_arguments)]
fn capture<F: FnMut()>(
    name: &str,
    kernel_variant: &str,
    dims: &[(&str, i64)],
    seed: Option<u64>,
    warmup: usize,
    iterations: usize,
    f: F,
) -> Capture {
    let (result, samples_us) = run_benchmark(name, warmup, iterations, f);
    let input_dimensions: BTreeMap<String, i64> =
        dims.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    Capture {
        result,
        case: ManifestCase::from_samples(
            name,
            kernel_variant,
            input_dimensions,
            seed,
            warmup,
            samples_us,
        ),
    }
}

fn run_benchmark<F: FnMut()>(
    name: &str,
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> (BenchmarkResult, Vec<f64>) {
    // Contract: at least one timed iteration. The CLI validator
    // (Config::validate) already rejects --iterations 0 with a clean
    // exit, but the function may be reused programmatically.
    debug_assert!(iterations > 0, "run_benchmark requires iterations > 0");

    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Timed iterations
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        durations.push(start.elapsed());
    }

    durations.sort();
    let total: Duration = durations.iter().sum();
    let total_us = total.as_secs_f64() * 1e6;
    let mean_us = total_us / iterations as f64;

    let percentile = |p: f64| -> f64 {
        // Nearest-rank method: rank = ceil(p/100 * N), then convert to 0-based index.
        let rank = ((p / 100.0) * iterations as f64).ceil() as usize;
        let idx = rank.saturating_sub(1).min(iterations - 1);
        durations[idx].as_secs_f64() * 1e6
    };

    let min_us = durations[0].as_secs_f64() * 1e6;
    let max_us = durations[iterations - 1].as_secs_f64() * 1e6;
    let p50_us = percentile(50.0);
    let p95_us = percentile(95.0);
    let p99_us = percentile(99.0);
    let throughput = if mean_us > 0.0 {
        1_000_000.0 / mean_us
    } else {
        f64::INFINITY
    };

    let samples_us: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();

    (
        BenchmarkResult {
            name: name.to_string(),
            iterations,
            total_duration_us: total_us,
            mean_us,
            p50_us,
            p95_us,
            p99_us,
            min_us,
            max_us,
            throughput_ops_per_sec: throughput,
        },
        samples_us,
    )
}

// ── GPU info collection ─────────────────────────────────────────────────────

/// Probe device 0 via the CUDA driver API when built with `cuda`.
///
/// `cust::init` is idempotent, so calling this before `GpuAccelerator::new`
/// in the kernel benches does not conflict with later context creation.
#[cfg(feature = "cuda")]
fn collect_gpu_info() -> Option<GpuInfo> {
    use cust::device::{Device, DeviceAttribute};
    use cust::{CudaApiVersion, CudaFlags};

    cust::init(CudaFlags::empty()).ok()?;
    let device = Device::get_device(0).ok()?;
    let device_name = device.name().ok()?;
    let total_bytes = device.total_memory().ok()? as u64;
    let major = device
        .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
        .ok()?;
    let minor = device
        .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
        .ok()?;

    // Driver-supported CUDA API version (cuDriverGetVersion via cust).
    let driver_cuda_api = CudaApiVersion::get()
        .map(|v| format!("{}.{}", v.major(), v.minor()))
        .unwrap_or_else(|_| "unknown".to_string());
    // Toolkit/nvcc version when available (distinct from driver-supported API).
    // Mirror build.rs resolution: CUDA_NVCC, then CUDA_HOME/CUDA_PATH/bin/nvcc,
    // then the nvcc on PATH.
    let nvcc_binary = std::env::var("CUDA_NVCC")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            for var in ["CUDA_HOME", "CUDA_PATH"] {
                if let Ok(root) = std::env::var(var) {
                    if !root.trim().is_empty() {
                        return Some(std::path::PathBuf::from(root).join("bin").join("nvcc"));
                    }
                }
            }
            None
        })
        .unwrap_or_else(|| std::path::PathBuf::from("nvcc"));
    let cuda_toolkit = std::process::Command::new(nvcc_binary)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines().find_map(|line| {
                // e.g. "Cuda compilation tools, release 13.3, V13.3.73"
                line.split("release ")
                    .nth(1)
                    .and_then(|rest| rest.split(',').next())
                    .map(str::trim)
                    .map(str::to_string)
            })
        })
        .unwrap_or_else(|| "unknown".to_string());

    Some(GpuInfo {
        device_name,
        // Field name is historical; stores driver-supported CUDA API, not KMD string.
        driver_version: driver_cuda_api,
        cuda_version: cuda_toolkit,
        sm_arch: format!("sm_{major}{minor}"),
        vram_total_mb: total_bytes / (1024 * 1024),
    })
}

#[cfg(not(feature = "cuda"))]
fn collect_gpu_info() -> Option<GpuInfo> {
    None
}

// ── Bitpacking benchmarks ───────────────────────────────────────────────────

fn bench_bitpacking(config: &Config) -> Vec<Capture> {
    let mut results = Vec::new();
    results.extend(bench_binary_pack(config));
    results.extend(bench_binary_unpack(config));
    results.extend(bench_ternary_pack(config));
    results.extend(bench_ternary_unpack(config));
    results
}

fn bench_binary_pack(config: &Config) -> Vec<Capture> {
    use myelin_accelerator::bitpacking::pack_binary;

    let small: Vec<bool> = (0..256).map(|i| i % 3 == 0).collect();
    let large: Vec<bool> = (0..65536).map(|i| i % 5 < 2).collect();

    vec![
        capture(
            "bitpack_binary_pack_256",
            "host-bitpack",
            &[("n", 256)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = pack_binary(&small);
            },
        ),
        capture(
            "bitpack_binary_pack_65536",
            "host-bitpack",
            &[("n", 65536)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = pack_binary(&large);
            },
        ),
    ]
}

fn bench_binary_unpack(config: &Config) -> Vec<Capture> {
    use myelin_accelerator::bitpacking::{pack_binary, unpack_binary};

    let small_src: Vec<bool> = (0..256).map(|i| i % 3 == 0).collect();
    let large_src: Vec<bool> = (0..65536).map(|i| i % 5 < 2).collect();
    let small = pack_binary(&small_src);
    let large = pack_binary(&large_src);

    vec![
        capture(
            "bitpack_binary_unpack_256",
            "host-bitpack",
            &[("n", 256)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_binary(&small, Some(256));
            },
        ),
        capture(
            "bitpack_binary_unpack_65536",
            "host-bitpack",
            &[("n", 65536)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_binary(&large, Some(65536));
            },
        ),
    ]
}

fn bench_ternary_pack(config: &Config) -> Vec<Capture> {
    use myelin_accelerator::bitpacking::pack_ternary;

    let small: Vec<i8> = ternary_pattern(256);
    let large: Vec<i8> = ternary_pattern(65536);

    vec![
        capture(
            "bitpack_ternary_pack_256",
            "host-bitpack",
            &[("n", 256)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = pack_ternary(&small);
            },
        ),
        capture(
            "bitpack_ternary_pack_65536",
            "host-bitpack",
            &[("n", 65536)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = pack_ternary(&large);
            },
        ),
    ]
}

fn bench_ternary_unpack(config: &Config) -> Vec<Capture> {
    use myelin_accelerator::bitpacking::{pack_ternary, unpack_ternary};

    let small = pack_ternary(&ternary_pattern(256));
    let large = pack_ternary(&ternary_pattern(65536));

    vec![
        capture(
            "bitpack_ternary_unpack_256",
            "host-bitpack",
            &[("n", 256)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_ternary(&small, Some(256));
            },
        ),
        capture(
            "bitpack_ternary_unpack_65536",
            "host-bitpack",
            &[("n", 65536)],
            None,
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_ternary(&large, Some(65536));
            },
        ),
    ]
}

fn ternary_pattern(n: usize) -> Vec<i8> {
    (0..n)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect()
}

// ── GPU kernel benchmarks (requires cuda feature) ───────────────────────────

#[cfg(feature = "cuda")]
fn bench_gpu_kernels(config: &Config) -> Vec<Capture> {
    use myelin_accelerator::{GpuAccelerator, GpuBuffer};

    let mut results = Vec::new();
    let acc = GpuAccelerator::new();
    if !acc.is_ready() {
        eprintln!("[bench] GPU not available, skipping kernel benchmarks");
        return results;
    }

    // Poisson encoding benchmark
    let n = 4096;
    let stimuli = GpuBuffer::from_slice(&vec![0.5f32; n]).unwrap();
    let mut spikes = GpuBuffer::<u32>::alloc(n).unwrap();
    results.push(capture(
        "poisson_encode_4096",
        "poisson-encode",
        &[("n", n as i64)],
        Some(42),
        config.warmup,
        config.iterations,
        || {
            acc.poisson_encode(&stimuli, &mut spikes, 42).unwrap();
        },
    ));

    // Satsolver extract benchmark
    let n_vars = 1024;
    let n_walkers = 256;
    let assignment = GpuBuffer::from_slice(&vec![0u8; n_vars * n_walkers]).unwrap();
    let best_walker = GpuBuffer::from_slice(&[0i32]).unwrap();
    let mut output = GpuBuffer::<u8>::alloc(n_vars).unwrap();
    results.push(capture(
        "satsolver_extract_1024x256",
        "satsolver-extract",
        &[("n_vars", n_vars as i64), ("n_walkers", n_walkers as i64)],
        None,
        config.warmup,
        config.iterations,
        || {
            acc.satsolver_extract(
                &assignment,
                &best_walker,
                &mut output,
                n_vars as i32,
                n_walkers as i32,
            )
            .unwrap();
        },
    ));

    // Packed ternary GEMV / GEMM (group-scaled)
    results.extend(bench_ternary_gpu(&acc, config));

    results
}

#[cfg(feature = "cuda")]
fn bench_ternary_gpu(acc: &myelin_accelerator::GpuAccelerator, config: &Config) -> Vec<Capture> {
    use myelin_accelerator::GpuBuffer;
    use myelin_accelerator::bitpacking::{
        DEFAULT_GROUP_SIZE, pack_ternary_matrix, uniform_group_scales,
    };

    let mut results = Vec::new();

    // Mid-size GEMV: M=1024, K=4096 (embedding-ish strip; not full vocab)
    let m = 1024usize;
    let k = 4096usize;
    let group = DEFAULT_GROUP_SIZE;
    let weights = ternary_pattern(m * k);
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = uniform_group_scales(m, k, group, 1.0);
    let x = vec![0.01f32; k];
    let d_w = GpuBuffer::from_slice(&packed).unwrap();
    let d_s = GpuBuffer::from_slice(&scales).unwrap();
    let d_x = GpuBuffer::from_slice(&x).unwrap();
    let mut d_y = GpuBuffer::<f32>::alloc(m).unwrap();

    results.push(capture(
        "ternary_gemv_1024x4096",
        "ternary-gemv",
        &[("m", m as i64), ("k", k as i64), ("group", group as i64)],
        None,
        config.warmup,
        config.iterations,
        || {
            acc.ternary_gemv(
                &d_w,
                &d_s,
                &d_x,
                &mut d_y,
                m as i32,
                k as i32,
                group as i32,
                false,
            )
            .unwrap();
        },
    ));

    results.push(capture(
        "ternary_gemv_1024x4096_skip_zeros",
        "ternary-gemv",
        &[
            ("m", m as i64),
            ("k", k as i64),
            ("group", group as i64),
            ("skip_zeros", 1),
        ],
        None,
        config.warmup,
        config.iterations,
        || {
            acc.ternary_gemv(
                &d_w,
                &d_s,
                &d_x,
                &mut d_y,
                m as i32,
                k as i32,
                group as i32,
                true,
            )
            .unwrap();
        },
    ));

    // Smaller GEMM tile for latency (M=256, K=1024, N=64)
    let m2 = 256usize;
    let k2 = 1024usize;
    let n2 = 64usize;
    let weights2 = ternary_pattern(m2 * k2);
    let packed2 = pack_ternary_matrix(&weights2, m2, k2);
    let scales2 = uniform_group_scales(m2, k2, group, 1.0);
    let b = vec![0.01f32; k2 * n2];
    let d_w2 = GpuBuffer::from_slice(&packed2).unwrap();
    let d_s2 = GpuBuffer::from_slice(&scales2).unwrap();
    let d_b = GpuBuffer::from_slice(&b).unwrap();
    let mut d_c = GpuBuffer::<f32>::alloc(m2 * n2).unwrap();

    results.push(capture(
        "ternary_gemm_256x1024x64",
        "ternary-gemm",
        &[
            ("m", m2 as i64),
            ("k", k2 as i64),
            ("n", n2 as i64),
            ("group", group as i64),
        ],
        None,
        config.warmup,
        config.iterations,
        || {
            acc.ternary_gemm(
                &d_w2,
                &d_s2,
                &d_b,
                &mut d_c,
                m2 as i32,
                k2 as i32,
                n2 as i32,
                group as i32,
                false,
            )
            .unwrap();
        },
    ));

    // Host-side dense f32 reference GEMV wall time (same shape as gemv bench)
    // for a coarse packed-vs-dense latency comparison (not FLOP-fair).
    // Reuse `y_host` outside the timed loop so allocation is not in the sample.
    let dense_w: Vec<f32> = weights.iter().map(|&t| t as f32).collect();
    let mut y_host = vec![0.0f32; m];
    let host_iters = config.iterations.min(50);
    results.push(capture(
        "dense_f32_gemv_1024x4096_host",
        "dense-f32-gemv-host",
        &[("m", m as i64), ("k", k as i64)],
        None,
        config.warmup,
        host_iters,
        || {
            for mi in 0..m {
                let mut acc_v = 0.0f32;
                let row = mi * k;
                for ki in 0..k {
                    acc_v += dense_w[row + ki] * x[ki];
                }
                y_host[mi] = acc_v;
            }
            std::hint::black_box(&y_host);
        },
    ));

    results
}

#[cfg(not(feature = "cuda"))]
fn bench_gpu_kernels(_config: &Config) -> Vec<Capture> {
    eprintln!("[bench] Built without cuda feature, skipping GPU kernel benchmarks");
    Vec::new()
}

// ── Baseline comparison ─────────────────────────────────────────────────────

fn compare_with_baseline(current: &[ManifestCase], config: &Config) -> i32 {
    let Some(baseline_path) = config.baseline.as_deref() else {
        return 0;
    };

    let Ok(data) = std::fs::read_to_string(baseline_path) else {
        eprintln!("[bench] Could not read baseline file: {baseline_path}");
        return if config.enforce_budget { 1 } else { 0 };
    };

    let rows = match load_baseline_rows(&data) {
        Ok(rows) => rows,
        Err(err) => {
            eprintln!("[bench] Could not parse baseline JSON: {err}");
            return if config.enforce_budget { 1 } else { 0 };
        }
    };

    println!("\n{:=>70}", "");
    println!("  Baseline comparison: {baseline_path}");
    if config.enforce_budget {
        println!("  Enforcement: ON (process fails only on class=fail)");
    } else {
        println!("  Enforcement: off (informational; set --enforce-budget to fail)");
    }
    println!("{:=>70}\n", "");

    println!(
        "{:<40} {:>12} {:>12} {:>10} {:>10} {:>22}",
        "Benchmark", "Base p50(µs)", "Curr p50(µs)", "Rel %", "Abs µs", "Class"
    );
    println!("{:-<110}", "");

    let mut unmatched_baseline: Vec<&str> = Vec::new();
    if config.enforce_budget {
        for base in &rows {
            if !current.iter().any(|c| c.name == base.name) {
                unmatched_baseline.push(base.name.as_str());
            }
        }
        for name in &unmatched_baseline {
            eprintln!("[bench] Baseline case missing from current run: {name}");
        }
    }

    let mut cases: Vec<ComparisonCase> = Vec::new();
    for curr in current {
        let Some(base) = rows.iter().find(|r| r.name == curr.name) else {
            continue;
        };
        let row = compare_one(
            &curr.name,
            base.source(),
            SampleSource::Samples(&curr.samples_us),
            &config.budget,
        );
        let rel_pct = if row.relative_delta.is_finite() {
            row.relative_delta * 100.0
        } else {
            f64::INFINITY
        };
        println!(
            "{:<40} {:>12.2} {:>12.2} {:>+9.1}% {:>10.2} {:>22}",
            row.name,
            row.baseline_median_us,
            row.current_median_us,
            rel_pct,
            row.absolute_delta_us,
            class_label(row.class),
        );
        cases.push(row);
    }

    let report = comparison_report(cases, config.budget.clone(), config.enforce_budget);
    write_comparison(&report, &config.output_prefix);

    if config.enforce_budget && (!unmatched_baseline.is_empty() || report.has_failure()) {
        if !unmatched_baseline.is_empty() {
            eprintln!(
                "[bench] {} baseline case(s) missing from current run (enforcement enabled).",
                unmatched_baseline.len()
            );
        }
        if report.has_failure() {
            eprintln!("[bench] Regression budget exceeded (enforcement enabled).");
        }
        1
    } else {
        0
    }
}

struct BaselineRow {
    name: String,
    samples_us: Vec<f64>,
    stats: SampleStats,
}

impl BaselineRow {
    fn source(&self) -> SampleSource<'_> {
        if self.samples_us.is_empty() {
            SampleSource::Stats(self.stats.clone())
        } else {
            SampleSource::Samples(&self.samples_us)
        }
    }
}

fn load_baseline_rows(data: &str) -> Result<Vec<BaselineRow>, String> {
    if let Ok(manifest) = serde_json::from_str::<BenchmarkManifest>(data) {
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "unsupported benchmark manifest schema version {}; supported version is {}",
                manifest.schema_version, MANIFEST_SCHEMA_VERSION
            ));
        }
        return Ok(manifest
            .cases
            .into_iter()
            .map(|c| BaselineRow {
                name: c.name,
                stats: SampleStats {
                    n: c.samples,
                    mean: c.mean_us,
                    median: c.median_us,
                    mad: c.mad_us,
                    relative_dispersion: c.relative_dispersion,
                    min: c.min_us,
                    max: c.max_us,
                    p50: c.p50_us,
                    p95: c.p95_us,
                    p99: c.p99_us,
                },
                samples_us: c.samples_us,
            })
            .collect());
    }
    let report: BenchmarkReport = serde_json::from_str(data).map_err(|e| e.to_string())?;
    Ok(report
        .results
        .into_iter()
        .map(|r| BaselineRow {
            name: r.name,
            samples_us: Vec::new(),
            stats: SampleStats {
                n: r.iterations,
                mean: r.mean_us,
                median: r.p50_us,
                mad: 0.0,
                relative_dispersion: 0.0,
                min: r.min_us,
                max: r.max_us,
                p50: r.p50_us,
                p95: r.p95_us,
                p99: r.p99_us,
            },
        })
        .collect())
}

fn class_label(class: RegressionClass) -> &'static str {
    match class {
        RegressionClass::Pass => "pass",
        RegressionClass::Fail => "fail",
        RegressionClass::Noisy => "noisy",
        RegressionClass::InsufficientSamples => "insufficient_samples",
    }
}

fn output_collides_with_baseline(prefix: &str, baseline: &str) -> bool {
    let outputs = [
        format!("{prefix}.json"),
        format!("{prefix}.csv"),
        format!("{prefix}.manifest.json"),
        format!("{prefix}.comparison.json"),
    ];
    let base = Path::new(baseline);
    outputs
        .iter()
        .any(|p| paths_refer_to_same_file(Path::new(p), base))
}

fn write_comparison(report: &myelin_accelerator::bench::ComparisonReport, prefix: &str) {
    let path = format!("{prefix}.comparison.json");
    match redact_and_canonicalize(report, &RedactionContext::from_env()) {
        Ok(data) => {
            if let Err(err) = std::fs::write(&path, data) {
                eprintln!("[bench] Could not write {path}: {err}");
            } else {
                println!("[bench] Comparison written to {path}");
            }
        }
        Err(err) => eprintln!("[bench] Could not serialize comparison: {err}"),
    }
}

fn sm_arch_to_cc(sm_arch: &str) -> Option<String> {
    let rest = sm_arch.strip_prefix("sm_")?;
    if rest.is_empty() {
        return None;
    }
    let (maj, min) = rest.split_at(rest.len().saturating_sub(1));
    if maj.is_empty() {
        Some(rest.to_string())
    } else {
        Some(format!("{maj}.{min}"))
    }
}

fn device_from_gpu(info: Option<&GpuInfo>, uuid: Option<String>) -> DeviceIdentity {
    let Some(info) = info else {
        return DeviceIdentity::unavailable();
    };
    DeviceIdentity {
        name: Some(info.device_name.clone()),
        uuid,
        compute_capability: sm_arch_to_cc(&info.sm_arch),
        sm_arch: Some(info.sm_arch.clone()),
        driver_version: Some(info.driver_version.clone()),
        runtime_version: None,
        toolchain_version: Some(info.cuda_version.clone()),
        vram_total_mb: Some(info.vram_total_mb),
    }
}

// ── Output writers ──────────────────────────────────────────────────────────

fn write_json(report: &BenchmarkReport, prefix: &str) {
    let path = format!("{prefix}.json");
    let data = serde_json::to_string_pretty(report).expect("serialize report");
    std::fs::write(&path, data).expect("write JSON");
    println!("[bench] Results written to {path}");
}

fn write_csv(results: &[BenchmarkResult], prefix: &str) {
    let path = format!("{prefix}.csv");
    let mut csv = String::from(
        "name,iterations,mean_us,p50_us,p95_us,p99_us,min_us,max_us,throughput_ops_sec\n",
    );
    for r in results {
        csv.push_str(&format!(
            "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0}\n",
            r.name,
            r.iterations,
            r.mean_us,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.min_us,
            r.max_us,
            r.throughput_ops_per_sec,
        ));
    }
    std::fs::write(&path, csv).expect("write CSV");
    println!("[bench] Results written to {path}");
}

// ── Pretty printer ──────────────────────────────────────────────────────────

fn print_results(results: &[BenchmarkResult]) {
    println!("\n{:=>90}", "");
    println!("  myelin-accelerator benchmark results");
    println!("{:=>90}\n", "");

    println!(
        "{:<40} {:>8} {:>10} {:>10} {:>10} {:>12}",
        "Benchmark", "Iters", "Mean(µs)", "P50(µs)", "P95(µs)", "Ops/sec"
    );
    println!("{:-<92}", "");

    for r in results {
        println!(
            "{:<40} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>12.0}",
            r.name, r.iterations, r.mean_us, r.p50_us, r.p95_us, r.throughput_ops_per_sec,
        );
    }
    println!();
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let config = Config::from_args();

    println!(
        "[bench] Warmup: {}, Iterations: {}",
        config.warmup, config.iterations
    );

    let gpu_info = collect_gpu_info();
    if let Some(ref info) = gpu_info {
        println!(
            "[bench] GPU: {} ({}, {} MB)",
            info.device_name, info.sm_arch, info.vram_total_mb
        );
    } else if cfg!(feature = "cuda") {
        println!("[bench] No CUDA device available (CPU-only benchmarks)");
    } else {
        // Without `cuda`, gpu_stub is compiled — is_available() is always false.
        // This is a feature gate, not a failed hardware probe.
        println!(
            "[bench] Built without `cuda` feature (stub path); \
             use --features bench,cuda for GPU benchmarks"
        );
    }

    let mut captures = Vec::new();
    captures.extend(bench_bitpacking(&config));
    captures.extend(bench_gpu_kernels(&config));

    let results: Vec<BenchmarkResult> = captures.iter().map(|c| c.result.clone()).collect();
    print_results(&results);

    let report = BenchmarkReport {
        timestamp: chrono_now(),
        gpu_info: gpu_info.clone(),
        config: RunConfig {
            warmup: config.warmup,
            iterations: config.iterations,
        },
        results: results.clone(),
    };

    if let Some(ref baseline) = config.baseline
        && output_collides_with_baseline(&config.output_prefix, baseline)
    {
        eprintln!(
            "[bench] Refusing to overwrite baseline {baseline}; choose a different --output prefix"
        );
        std::process::exit(1);
    }

    let (uuid, power_clock) = probe_power_clock();
    let cases: Vec<ManifestCase> = captures.into_iter().map(|c| c.case).collect();
    let mut manifest = BenchmarkManifest::new(
        myelin_accelerator::bench::RunTiming {
            warmup: config.warmup,
            samples: config.iterations,
            seed: None,
        },
        cases,
    );
    manifest.device = device_from_gpu(gpu_info.as_ref(), uuid);
    if manifest.device.toolchain_version.is_none() {
        manifest.device.toolchain_version = manifest.toolchain.nvcc.clone();
    }
    manifest.power_clock = power_clock;

    write_json(&report, &config.output_prefix);
    write_csv(&results, &config.output_prefix);

    let manifest_path = PathBuf::from(format!("{}.manifest.json", config.output_prefix));
    if let Err(err) = write_canonical_manifest(&manifest_path, &manifest) {
        eprintln!("[bench] Could not write manifest: {err}");
        std::process::exit(1);
    }
    println!("[bench] Manifest written to {}", manifest_path.display());

    let mut exit_code = 0;
    if config.baseline.is_some() {
        exit_code = compare_with_baseline(&manifest.cases, &config);
    }

    println!("[bench] Done.");
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

/// Simple timestamp without chrono dependency.
fn chrono_now() -> String {
    // Use system time to produce an ISO-8601-ish timestamp.
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Simple conversion: days since epoch -> Y-M-D (ignores leap seconds)
    let days = secs / 86400;
    let (y, m, d) = days_to_ymd(days as i64 + 719468);
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

fn days_to_ymd(g: i64) -> (i64, u32, u32) {
    let mut y = (10000 * g + 14780) / 3652425;
    let mut doy = g - (365 * y + y / 4 - y / 100 + y / 400);
    if doy < 0 {
        y -= 1;
        doy = g - (365 * y + y / 4 - y / 100 + y / 400);
    }
    let mi = (100 * doy + 52) / 3060;
    let month = (mi + 2) % 12 + 1;
    let year = y + (mi + 2) / 12;
    let day = doy - (mi * 306 + 5) / 10 + 1;
    (year, month as u32, day as u32)
}
