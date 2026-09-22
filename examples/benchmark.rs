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
    BenchmarkManifest, ComparisonCase, ComparisonRejection, ComparisonRejectionReason,
    CudaDeviceUuid, DeviceIdentity, MANIFEST_SCHEMA_VERSION, ManifestCase, PowerClockControls,
    RedactionContext, RegressionBudget, RegressionClass, SampleSource, SampleStats, compare_one,
    comparison_report, enforce_budget_requested, paths_refer_to_same_file, probe_power_clock,
    redact_and_canonicalize, write_atomic_bytes, write_canonical_json, write_canonical_manifest,
};
use std::collections::{BTreeMap, BTreeSet};
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
        let env_enforce_budget = enforce_budget_requested().unwrap_or_else(|message| {
            eprintln!("{message}");
            std::process::exit(1);
        });
        let enforce_budget = raw.enforce_budget || env_enforce_budget;
        if let Err(message) =
            validate_enforcement_configuration(enforce_budget, raw.baseline.as_deref())
        {
            eprintln!("{message}");
            std::process::exit(1);
        }
        Config {
            warmup: raw.warmup.unwrap_or(10),
            iterations: raw.iterations.unwrap_or(100),
            baseline: raw.baseline,
            output_prefix: raw
                .output_prefix
                .unwrap_or_else(|| "benchmark_results".to_string()),
            enforce_budget,
            budget,
        }
    }
}

fn validate_enforcement_configuration(
    enforce_budget: bool,
    baseline: Option<&str>,
) -> Result<(), &'static str> {
    if enforce_budget && baseline.is_none() {
        Err("budget enforcement requires --baseline <FILE>")
    } else {
        Ok(())
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
    println!(
        "  --enforce-budget         Enforce comparison gate; requires --baseline\n\
         \x20                         (also MYELIN_BENCH_ENFORCE_BUDGET=1)"
    );
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
    #[serde(skip)]
    cuda_device_uuid: Option<CudaDeviceUuid>,
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

#[cfg(feature = "cuda")]
fn cuda_device_uuid(device: cust::device::Device) -> Option<CudaDeviceUuid> {
    let mut uuid = cust::sys::CUuuid::default();
    // SAFETY: CUDA is initialized before this helper is called, `device` came
    // from `Device::get_device`, and `uuid` is valid writable storage.
    let result = unsafe { cust::sys::cuDeviceGetUuid_v2(&mut uuid, device.as_raw()) };
    if result != cust::sys::cudaError_enum::CUDA_SUCCESS {
        return None;
    }
    Some(CudaDeviceUuid::from_bytes(
        uuid.bytes.map(|byte| byte as u8),
    ))
}

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
    let cuda_device_uuid = cuda_device_uuid(device);
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
                if let Ok(root) = std::env::var(var)
                    && !root.trim().is_empty()
                {
                    return Some(std::path::PathBuf::from(root).join("bin").join("nvcc"));
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
        cuda_device_uuid,
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
        match acc.fallback() {
            Some(fb) => eprintln!(
                "[bench] GPU not available ({}), skipping kernel benchmarks",
                fb.reason.code()
            ),
            None => eprintln!("[bench] GPU not available, skipping kernel benchmarks"),
        }
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
            for (mi, output) in y_host.iter_mut().enumerate() {
                let mut acc_v = 0.0f32;
                let row = mi * k;
                for ki in 0..k {
                    acc_v += dense_w[row + ki] * x[ki];
                }
                *output = acc_v;
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

fn compare_with_baseline(current: &BenchmarkManifest, config: &Config) -> i32 {
    let Some(baseline_path) = config.baseline.as_deref() else {
        return 0;
    };

    let Ok(data) = std::fs::read_to_string(baseline_path) else {
        eprintln!("[bench] Could not read baseline file: {baseline_path}");
        return write_rejected_comparison(config, ComparisonRejectionReason::BaselineReadFailure);
    };

    let baseline = match load_baseline_rows(&data) {
        Ok(baseline) => baseline,
        Err(err) => {
            eprintln!("[bench] Could not parse baseline JSON: {err}");
            return write_rejected_comparison(
                config,
                ComparisonRejectionReason::BaselineParseFailure,
            );
        }
    };
    let rows = baseline.rows;
    let current_cases = &current.cases;

    println!("\n{:=>70}", "");
    println!("  Baseline comparison: {baseline_path}");
    if config.enforce_budget {
        println!(
            "  Enforcement: ON (fails on regressions, insufficient samples, or rejected comparisons)"
        );
    } else {
        println!("  Enforcement: off (informational; set --enforce-budget to fail)");
    }
    println!("{:=>70}\n", "");

    println!(
        "{:<40} {:>12} {:>12} {:>10} {:>10} {:>22}",
        "Benchmark", "Base p50(µs)", "Curr p50(µs)", "Rel %", "Abs µs", "Class"
    );
    println!("{:-<110}", "");

    let duplicate_baseline_names = duplicate_names(rows.iter().map(|row| row.name.as_str()));
    let duplicate_current_names =
        duplicate_names(current_cases.iter().map(|case| case.name.as_str()));
    for name in &duplicate_baseline_names {
        eprintln!("[bench] Duplicate baseline case name: {name}");
    }
    for name in &duplicate_current_names {
        eprintln!("[bench] Duplicate current case name: {name}");
    }

    let mut rejections = Vec::new();
    let current_toolchain =
        canonicalize_toolchain(&current.toolchain).expect("toolchain provenance must canonicalize");
    let build_profile_mismatch = baseline.build_profile.as_ref().is_some_and(|profile| {
        profile.rustc != current_toolchain.rustc
            || profile.nvcc != current_toolchain.nvcc
            || profile.rustflags != current_toolchain.rustflags
            || profile.target_features != current_toolchain.target_features
            || profile.cargo_profile != current_toolchain.cargo_profile
            || profile.cargo_profile_fingerprint != current_toolchain.cargo_profile_fingerprint
            || profile.cargo_lto != current_toolchain.cargo_lto
            || profile.cargo_codegen_units != current_toolchain.cargo_codegen_units
            || profile.cargo_incremental != current_toolchain.cargo_incremental
            || profile.panic_strategy != current_toolchain.panic_strategy
            || profile.cargo_profile_config != current_toolchain.cargo_profile_config
            || profile.cuda_arch != current_toolchain.cuda_arch
            || profile.ptx_version != current_toolchain.ptx_version
            || profile.target_triple != current_toolchain.target_triple
            || profile.opt_level != current_toolchain.opt_level
            || profile.debug_assertions != current_toolchain.debug_assertions
    });
    if build_profile_mismatch {
        eprintln!(
            "[bench] Build profile mismatch: baseline and current compiler/profile settings differ"
        );
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::BuildProfileMismatch,
            case_name: None,
        });
    }
    let hardware_identity_mismatch = baseline
        .device
        .as_ref()
        .is_some_and(|device| !same_hardware_identity(device, &current.device));
    if hardware_identity_mismatch {
        eprintln!("[bench] Hardware identity mismatch: baseline and current devices differ");
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::HardwareIdentityMismatch,
            case_name: None,
        });
    }
    let cuda_environment_mismatch = baseline
        .device
        .as_ref()
        .is_some_and(|device| !same_cuda_environment(device, &current.device));
    if cuda_environment_mismatch {
        eprintln!(
            "[bench] CUDA environment mismatch: baseline and current driver/runtime/toolkit versions differ"
        );
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::CudaEnvironmentMismatch,
            case_name: None,
        });
    }
    let power_clock_mismatch = baseline
        .power_clock
        .as_ref()
        .is_some_and(|controls| controls != &current.power_clock);
    if power_clock_mismatch {
        eprintln!("[bench] GPU control mismatch: baseline and current power/clock settings differ");
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::PowerClockMismatch,
            case_name: None,
        });
    }
    let host_identity_mismatch = baseline.host_identity.as_ref().is_some_and(|host| {
        host.arch != current.toolchain.host_arch
            || host.os != current.toolchain.host_os
            || host.cpu_model != current.toolchain.cpu_model
    });
    if host_identity_mismatch {
        eprintln!(
            "[bench] Host identity mismatch: baseline and current architecture/OS/CPU differ"
        );
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::HostIdentityMismatch,
            case_name: None,
        });
    }
    let feature_set_mismatch = baseline
        .features
        .as_ref()
        .is_some_and(|features| !same_feature_set(features, &current.features));
    if feature_set_mismatch {
        eprintln!("[bench] Feature-set mismatch: baseline and current Cargo features differ");
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::FeatureSetMismatch,
            case_name: None,
        });
    }
    for name in &duplicate_baseline_names {
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::DuplicateBaselineName,
            case_name: Some(name.clone()),
        });
    }
    for name in &duplicate_current_names {
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::DuplicateCurrentName,
            case_name: Some(name.clone()),
        });
    }

    let unmatched_baseline: Vec<&str> = rows
        .iter()
        .filter(|base| !current_cases.iter().any(|case| case.name == base.name))
        .map(|base| base.name.as_str())
        .collect();
    for name in &unmatched_baseline {
        eprintln!("[bench] Baseline case missing from current run: {name}");
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::MissingCurrentCase,
            case_name: Some((*name).to_string()),
        });
    }
    let unmatched_current: Vec<&str> = current_cases
        .iter()
        .filter(|case| !rows.iter().any(|base| base.name == case.name))
        .map(|case| case.name.as_str())
        .collect();
    for name in &unmatched_current {
        eprintln!("[bench] Current case missing from baseline: {name}");
        rejections.push(ComparisonRejection {
            reason: ComparisonRejectionReason::MissingBaselineCase,
            case_name: Some((*name).to_string()),
        });
    }

    let mut cases: Vec<ComparisonCase> = Vec::new();
    let mut mismatched_workloads = Vec::new();
    let run_provenance_mismatch = build_profile_mismatch
        || hardware_identity_mismatch
        || cuda_environment_mismatch
        || power_clock_mismatch
        || host_identity_mismatch
        || feature_set_mismatch;
    for curr in current_cases {
        if duplicate_baseline_names.contains(&curr.name)
            || duplicate_current_names.contains(&curr.name)
        {
            continue;
        }
        let Some(base) = rows.iter().find(|r| r.name == curr.name) else {
            continue;
        };
        if run_provenance_mismatch {
            continue;
        }
        if base.warmup.is_some_and(|warmup| warmup != curr.warmup) {
            eprintln!(
                "[bench] Warmup mismatch for {}: baseline and current warmup counts differ",
                curr.name
            );
            mismatched_workloads.push(curr.name.as_str());
            rejections.push(ComparisonRejection {
                reason: ComparisonRejectionReason::WarmupMismatch,
                case_name: Some(curr.name.clone()),
            });
            continue;
        }
        if !base.matches_workload(curr) {
            eprintln!(
                "[bench] Workload metadata mismatch for {}: baseline and current kernel variant/input dimensions/seed differ",
                curr.name
            );
            mismatched_workloads.push(curr.name.as_str());
            rejections.push(ComparisonRejection {
                reason: ComparisonRejectionReason::WorkloadMetadataMismatch,
                case_name: Some(curr.name.clone()),
            });
            continue;
        }
        let row = compare_one(
            &curr.name,
            base.source(),
            SampleSource::Samples(&curr.samples_us),
            &config.budget,
        );
        let rel_pct = row
            .relative_delta
            .map(|delta| format!("{:+.1}%", delta * 100.0))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{:<40} {:>12.2} {:>12.2} {:>10} {:>10.2} {:>22}",
            row.name,
            row.baseline_median_us,
            row.current_median_us,
            rel_pct,
            row.absolute_delta_us,
            class_label(row.class),
        );
        cases.push(row);
    }

    let report = comparison_report(
        cases,
        config.budget.clone(),
        config.enforce_budget,
        rejections,
    );
    if let Err(err) = write_comparison(&report, &config.output_prefix) {
        eprintln!("[bench] {err}");
        return 1;
    }
    if config.enforce_budget && !report.gate_passed {
        if !unmatched_baseline.is_empty() {
            eprintln!(
                "[bench] {} baseline case(s) missing from current run (enforcement enabled).",
                unmatched_baseline.len()
            );
        }
        if !unmatched_current.is_empty() {
            eprintln!(
                "[bench] {} current case(s) missing from baseline (enforcement enabled).",
                unmatched_current.len()
            );
        }
        if report.has_failure() {
            eprintln!(
                "[bench] Regression or insufficient-sample case rejected (enforcement enabled)."
            );
        }
        if !mismatched_workloads.is_empty() {
            eprintln!(
                "[bench] {} workload metadata mismatch(es) rejected (enforcement enabled).",
                mismatched_workloads.len()
            );
        }
        if report.cases.is_empty() {
            eprintln!("[bench] No comparable benchmark cases were produced (enforcement enabled).");
        }
        if !duplicate_baseline_names.is_empty() || !duplicate_current_names.is_empty() {
            eprintln!("[bench] Duplicate case names are ambiguous (enforcement enabled).");
        }
        1
    } else {
        0
    }
}

fn write_rejected_comparison(config: &Config, reason: ComparisonRejectionReason) -> i32 {
    let report = comparison_report(
        Vec::new(),
        config.budget.clone(),
        config.enforce_budget,
        vec![ComparisonRejection {
            reason,
            case_name: None,
        }],
    );
    if let Err(err) = write_comparison(&report, &config.output_prefix) {
        eprintln!("[bench] {err}");
        return 1;
    }
    1
}

fn duplicate_names<'a>(names: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            duplicates.insert(name.to_string());
        }
    }
    duplicates
}

struct BaselineRow {
    name: String,
    kernel_variant: Option<String>,
    input_dimensions: Option<BTreeMap<String, i64>>,
    seed: Option<u64>,
    warmup: Option<usize>,
    samples_us: Vec<f64>,
    stats: SampleStats,
    dispersion_known: bool,
}

struct BuildProfile {
    rustc: Option<String>,
    nvcc: Option<String>,
    rustflags: Vec<String>,
    target_features: Vec<String>,
    cargo_profile: Option<String>,
    cargo_profile_fingerprint: Option<String>,
    cargo_lto: Option<String>,
    cargo_codegen_units: Option<String>,
    cargo_incremental: Option<String>,
    panic_strategy: Option<String>,
    cargo_profile_config: Option<String>,
    cuda_arch: Option<String>,
    ptx_version: Option<String>,
    target_triple: Option<String>,
    opt_level: String,
    debug_assertions: bool,
}

struct HostIdentity {
    arch: String,
    os: String,
    cpu_model: Option<String>,
}

struct LoadedBaseline {
    rows: Vec<BaselineRow>,
    build_profile: Option<BuildProfile>,
    device: Option<DeviceIdentity>,
    power_clock: Option<PowerClockControls>,
    host_identity: Option<HostIdentity>,
    features: Option<Vec<String>>,
}

fn same_hardware_identity(baseline: &DeviceIdentity, current: &DeviceIdentity) -> bool {
    baseline.name == current.name
        && baseline.uuid == current.uuid
        && baseline.compute_capability == current.compute_capability
        && baseline.sm_arch == current.sm_arch
        && baseline.vram_total_mb == current.vram_total_mb
}

fn same_cuda_environment(baseline: &DeviceIdentity, current: &DeviceIdentity) -> bool {
    baseline.driver_version == current.driver_version
        && baseline.runtime_version == current.runtime_version
        && baseline.toolchain_version == current.toolchain_version
}

fn same_feature_set(baseline: &[String], current: &[String]) -> bool {
    baseline.iter().collect::<BTreeSet<_>>() == current.iter().collect::<BTreeSet<_>>()
}

fn canonicalize_toolchain(
    toolchain: &myelin_accelerator::bench::ToolchainInfo,
) -> Result<myelin_accelerator::bench::ToolchainInfo, String> {
    let text = redact_and_canonicalize(toolchain, &RedactionContext::from_env())
        .map_err(|err| err.to_string())?;
    serde_json::from_str(&text).map_err(|err| err.to_string())
}

fn validate_manifest_case(case: &ManifestCase) -> Result<SampleStats, String> {
    if case.samples_us.is_empty() {
        return Err(format!(
            "manifest case {:?} has no latency samples",
            case.name
        ));
    }
    if case.samples != case.samples_us.len() {
        return Err(format!(
            "manifest case {:?} declares {} samples but contains {}",
            case.name,
            case.samples,
            case.samples_us.len()
        ));
    }
    if case.samples_us.iter().any(|sample| !sample.is_finite()) {
        return Err(format!(
            "manifest case {:?} contains non-finite samples",
            case.name
        ));
    }
    let computed = myelin_accelerator::bench::sample_stats(&case.samples_us);
    let aggregates = [
        ("mean_us", case.mean_us, computed.mean),
        ("median_us", case.median_us, computed.median),
        ("mad_us", case.mad_us, computed.mad),
        (
            "relative_dispersion",
            case.relative_dispersion,
            computed.relative_dispersion,
        ),
        ("min_us", case.min_us, computed.min),
        ("max_us", case.max_us, computed.max),
        ("p50_us", case.p50_us, computed.p50),
        ("p95_us", case.p95_us, computed.p95),
        ("p99_us", case.p99_us, computed.p99),
    ];
    for (field, stored, expected) in aggregates {
        let tolerance = 1e-12 * stored.abs().max(expected.abs()).max(1.0);
        if !stored.is_finite() || (stored - expected).abs() > tolerance {
            return Err(format!(
                "manifest case {:?} has inconsistent {field}: stored {stored}, computed {expected}",
                case.name
            ));
        }
    }
    Ok(computed)
}

impl BaselineRow {
    fn matches_workload(&self, current: &ManifestCase) -> bool {
        match (&self.kernel_variant, &self.input_dimensions) {
            (Some(variant), Some(dimensions)) => {
                variant == &current.kernel_variant
                    && dimensions == &current.input_dimensions
                    && self.seed == current.seed
            }
            _ => true,
        }
    }

    fn source(&self) -> SampleSource<'_> {
        if !self.dispersion_known {
            SampleSource::StatsWithoutDispersion(self.stats.clone())
        } else if self.samples_us.is_empty() {
            SampleSource::Stats(self.stats.clone())
        } else {
            SampleSource::Samples(&self.samples_us)
        }
    }
}

fn load_baseline_rows(data: &str) -> Result<LoadedBaseline, String> {
    if let Ok(manifest) = serde_json::from_str::<BenchmarkManifest>(data) {
        if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "unsupported benchmark manifest schema version {}; supported version is {}",
                manifest.schema_version, MANIFEST_SCHEMA_VERSION
            ));
        }
        for case in &manifest.cases {
            validate_manifest_case(case)?;
        }
        let toolchain = canonicalize_toolchain(&manifest.toolchain)?;
        let build_profile = BuildProfile {
            rustc: toolchain.rustc.clone(),
            nvcc: toolchain.nvcc.clone(),
            rustflags: toolchain.rustflags.clone(),
            target_features: toolchain.target_features.clone(),
            cargo_profile: toolchain.cargo_profile.clone(),
            cargo_profile_fingerprint: toolchain.cargo_profile_fingerprint.clone(),
            cargo_lto: toolchain.cargo_lto.clone(),
            cargo_codegen_units: toolchain.cargo_codegen_units.clone(),
            cargo_incremental: toolchain.cargo_incremental.clone(),
            panic_strategy: toolchain.panic_strategy.clone(),
            cargo_profile_config: toolchain.cargo_profile_config.clone(),
            cuda_arch: toolchain.cuda_arch.clone(),
            ptx_version: toolchain.ptx_version.clone(),
            target_triple: toolchain.target_triple.clone(),
            opt_level: toolchain.opt_level,
            debug_assertions: toolchain.debug_assertions,
        };
        let host_identity = HostIdentity {
            arch: toolchain.host_arch.clone(),
            os: toolchain.host_os.clone(),
            cpu_model: toolchain.cpu_model.clone(),
        };
        let device = manifest.device;
        let power_clock = manifest.power_clock;
        let features = manifest.features;
        let rows = manifest
            .cases
            .into_iter()
            .map(|c| {
                let stats = myelin_accelerator::bench::sample_stats(&c.samples_us);
                BaselineRow {
                    name: c.name,
                    kernel_variant: Some(c.kernel_variant),
                    input_dimensions: Some(c.input_dimensions),
                    seed: c.seed,
                    warmup: Some(c.warmup),
                    stats,
                    samples_us: c.samples_us,
                    dispersion_known: true,
                }
            })
            .collect();
        return Ok(LoadedBaseline {
            rows,
            build_profile: Some(build_profile),
            device: Some(device),
            power_clock: Some(power_clock),
            host_identity: Some(host_identity),
            features: Some(features),
        });
    }
    let report: BenchmarkReport = serde_json::from_str(data).map_err(|e| e.to_string())?;
    let warmup = report.config.warmup;
    let rows = report
        .results
        .into_iter()
        .map(|r| BaselineRow {
            name: r.name,
            kernel_variant: None,
            input_dimensions: None,
            seed: None,
            warmup: Some(warmup),
            samples_us: Vec::new(),
            dispersion_known: false,
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
        .collect();
    Ok(LoadedBaseline {
        rows,
        build_profile: None,
        device: None,
        power_clock: None,
        host_identity: None,
        features: None,
    })
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

fn write_comparison(
    report: &myelin_accelerator::bench::ComparisonReport,
    prefix: &str,
) -> Result<(), String> {
    let path = format!("{prefix}.comparison.json");
    write_canonical_json(Path::new(&path), report)
        .map_err(|err| format!("Could not write {path}: {err}"))?;
    println!("[bench] Comparison written to {path}");
    Ok(())
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

fn device_from_gpu(
    info: Option<&GpuInfo>,
    uuid: Option<String>,
    driver_version: Option<String>,
    toolchain_version: Option<String>,
) -> DeviceIdentity {
    let Some(info) = info else {
        return DeviceIdentity::unavailable();
    };
    DeviceIdentity {
        name: Some(info.device_name.clone()),
        uuid,
        compute_capability: sm_arch_to_cc(&info.sm_arch),
        sm_arch: Some(info.sm_arch.clone()),
        driver_version,
        runtime_version: None,
        toolchain_version,
        vram_total_mb: Some(info.vram_total_mb),
    }
}

// ── Output writers ──────────────────────────────────────────────────────────

fn write_json(report: &BenchmarkReport, prefix: &str) {
    let path = format!("{prefix}.json");
    write_canonical_json(Path::new(&path), report).expect("write JSON");
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
    write_atomic_bytes(Path::new(&path), csv).expect("write CSV");
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

    let (uuid, driver_version, power_clock) = probe_power_clock(
        gpu_info
            .as_ref()
            .and_then(|info| info.cuda_device_uuid.as_ref()),
    );
    let cases: Vec<ManifestCase> = captures.into_iter().map(|c| c.case).collect();
    let mut manifest = BenchmarkManifest::new(
        myelin_accelerator::bench::RunTiming {
            warmup: config.warmup,
            samples: config.iterations,
            seed: None,
        },
        cases,
    );
    manifest.device = device_from_gpu(
        gpu_info.as_ref(),
        uuid,
        driver_version,
        manifest.toolchain.nvcc.clone(),
    );
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
        exit_code = compare_with_baseline(&manifest, &config);
    }

    println!("[bench] Done.");
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforcement_requires_a_baseline() {
        assert!(validate_enforcement_configuration(false, None).is_ok());
        assert!(validate_enforcement_configuration(false, Some("baseline.json")).is_ok());
        assert!(validate_enforcement_configuration(true, Some("baseline.json")).is_ok());
        assert_eq!(
            validate_enforcement_configuration(true, None),
            Err("budget enforcement requires --baseline <FILE>")
        );
    }

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "myelin-benchmark-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn manifest_case(name: &str, variant: &str, n: i64, value: f64) -> ManifestCase {
        manifest_case_with_seed(name, variant, n, value, None)
    }

    fn manifest_case_with_seed(
        name: &str,
        variant: &str,
        n: i64,
        value: f64,
        seed: Option<u64>,
    ) -> ManifestCase {
        ManifestCase::from_samples(
            name,
            variant,
            BTreeMap::from([("n".to_string(), n)]),
            seed,
            1,
            vec![value; 8],
        )
    }

    fn comparison_json(prefix: &str) -> serde_json::Value {
        let path = format!("{prefix}.comparison.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("read comparison report"))
            .expect("parse comparison report")
    }

    fn rejection_reasons(report: &serde_json::Value) -> Vec<&str> {
        report["rejections"]
            .as_array()
            .expect("comparison rejections")
            .iter()
            .map(|rejection| rejection["reason"].as_str().expect("rejection reason"))
            .collect()
    }

    fn compare_cases_with_baseline(current: &[ManifestCase], config: &Config) -> i32 {
        let manifest = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: config.warmup,
                samples: config.iterations,
                seed: None,
            },
            current.to_vec(),
        );
        compare_with_baseline(&manifest, config)
    }

    fn legacy_result(name: &str, median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            name: name.to_string(),
            iterations: 8,
            total_duration_us: median_us * 8.0,
            mean_us: median_us,
            p50_us: median_us,
            p95_us: median_us,
            p99_us: median_us,
            min_us: median_us,
            max_us: median_us,
            throughput_ops_per_sec: 1_000_000.0 / median_us,
        }
    }

    #[test]
    fn device_manifest_uses_driver_probe_and_build_toolchain() {
        let info = GpuInfo {
            device_name: "test gpu".to_string(),
            sm_arch: "sm_120".to_string(),
            vram_total_mb: 16_384,
            driver_version: "runtime-api-should-not-be-driver".to_string(),
            cuda_version: "runtime-path-nvcc-should-not-win".to_string(),
            cuda_device_uuid: None,
        };

        let device = device_from_gpu(
            Some(&info),
            Some("GPU-test".to_string()),
            Some("610.43.03".to_string()),
            Some("Cuda compilation tools, release 13.3, V13.3.101".to_string()),
        );

        assert_eq!(device.driver_version.as_deref(), Some("610.43.03"));
        assert_eq!(
            device.toolchain_version.as_deref(),
            Some("Cuda compilation tools, release 13.3, V13.3.101")
        );
        assert_eq!(device.runtime_version, None);
    }

    #[test]
    fn legacy_baseline_without_dispersion_is_not_enforced_as_stable() {
        let legacy = BenchmarkReport {
            timestamp: "2026-09-21T00:00:00Z".to_string(),
            gpu_info: None,
            config: RunConfig {
                warmup: 1,
                iterations: 8,
            },
            results: vec![legacy_result("legacy", 100.0)],
        };
        let baseline =
            load_baseline_rows(&serde_json::to_string(&legacy).expect("serialize legacy"))
                .expect("load legacy baseline");
        let current = vec![130.0; 8];
        let comparison = compare_one(
            "legacy",
            baseline.rows[0].source(),
            SampleSource::Samples(&current),
            &RegressionBudget::default(),
        );

        assert_eq!(comparison.class, RegressionClass::InsufficientSamples);
        assert_eq!(comparison.baseline_relative_dispersion, None);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_workload_identity() {
        let dir = temp_dir("identity-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![manifest_case("same-name", "variant-a", 128, 100.0)],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let current = vec![manifest_case("same-name", "variant-b", 256, 100.0)];

        let exit_code = compare_cases_with_baseline(&current, &config);

        assert_eq!(exit_code, 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["workload_metadata_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_build_profile() {
        let dir = temp_dir("build-profile-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        baseline.toolchain.opt_level = format!("{}-different", baseline.toolchain.opt_level);
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["build_profile_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_device_identity() {
        let dir = temp_dir("device-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        baseline.device.name = Some("definitely-not-the-current-device".to_string());
        baseline.device.uuid = Some("GPU-00000000-0000-0000-0000-000000000000".to_string());
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["hardware_identity_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_feature_set() {
        let dir = temp_dir("feature-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        baseline.features = vec!["definitely-not-enabled".to_string()];
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["feature_set_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_power_clock_controls() {
        let dir = temp_dir("power-clock-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        baseline.power_clock.power_limit_w = Some(300.0);
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["power_clock_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_host_identity() {
        let dir = temp_dir("host-identity-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        baseline.toolchain.host_arch = "definitely-not-the-current-arch".to_string();
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["host_identity_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_warmup_count() {
        let dir = temp_dir("warmup-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let mut baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        baseline_case.warmup = 2;
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 2,
                samples: 8,
                seed: None,
            },
            vec![baseline_case],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let current = manifest_case("same-name", "variant", 128, 100.0);

        assert_eq!(compare_cases_with_baseline(&[current], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["warmup_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_seed_identity() {
        let dir = temp_dir("seed-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![manifest_case_with_seed(
                "stochastic",
                "variant",
                128,
                100.0,
                Some(1),
            )],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let current = vec![manifest_case_with_seed(
            "stochastic",
            "variant",
            128,
            100.0,
            Some(2),
        )];

        assert_eq!(compare_cases_with_baseline(&current, &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert!(rejection_reasons(&report).contains(&"workload_metadata_mismatch"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn comparison_artifact_write_failure_makes_run_fail() {
        let dir = temp_dir("comparison-write-failure");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let output_prefix = dir.join("current");
        std::fs::create_dir(output_prefix.with_extension("comparison.json"))
            .expect("create colliding comparison directory");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: output_prefix.to_string_lossy().into_owned(),
            enforce_budget: false,
            budget: RegressionBudget::default(),
        };

        let exit_code = compare_cases_with_baseline(&[baseline_case], &config);

        assert_eq!(exit_code, 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_an_empty_baseline_and_current_run() {
        let dir = temp_dir("empty-enforced-comparison");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        let exit_code = compare_cases_with_baseline(&[], &config);

        assert_eq!(exit_code, 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert!(rejection_reasons(&report).contains(&"no_comparable_cases"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_records_unreadable_baseline() {
        let dir = temp_dir("unreadable-baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(dir.join("missing.json").to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["baseline_read_failure", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_records_invalid_baseline() {
        let dir = temp_dir("invalid-baseline");
        let baseline_path = dir.join("invalid.json");
        std::fs::write(&baseline_path, "not-json").expect("write invalid baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["baseline_parse_failure", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn informational_comparison_fails_for_unreadable_baseline() {
        let dir = temp_dir("informational-unreadable-baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(dir.join("missing.json").to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: false,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[], &config), 1);
        assert_eq!(
            rejection_reasons(&comparison_json(&config.output_prefix)),
            vec!["baseline_read_failure", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn informational_comparison_fails_for_invalid_baseline() {
        let dir = temp_dir("informational-invalid-baseline");
        let baseline_path = dir.join("invalid.json");
        std::fs::write(&baseline_path, "not-json").expect("write invalid baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: false,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[], &config), 1);
        assert_eq!(
            rejection_reasons(&comparison_json(&config.output_prefix)),
            vec!["baseline_parse_failure", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn manifest_baseline_requires_samples_and_consistent_aggregates() {
        for mutation in ["empty-samples", "wrong-median"] {
            let dir = temp_dir(mutation);
            let baseline_path = dir.join("baseline.manifest.json");
            let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
            let mut baseline = BenchmarkManifest::new(
                myelin_accelerator::bench::RunTiming {
                    warmup: 1,
                    samples: 8,
                    seed: None,
                },
                vec![baseline_case.clone()],
            );
            if mutation == "empty-samples" {
                baseline.cases[0].samples_us.clear();
            } else {
                baseline.cases[0].median_us = 1.0;
            }
            std::fs::write(
                &baseline_path,
                serde_json::to_vec(&baseline).expect("serialize manifest"),
            )
            .expect("write baseline");
            let config = Config {
                warmup: 1,
                iterations: 8,
                baseline: Some(baseline_path.to_string_lossy().into_owned()),
                output_prefix: dir.join("current").to_string_lossy().into_owned(),
                enforce_budget: true,
                budget: RegressionBudget::default(),
            };

            assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
            assert_eq!(
                rejection_reasons(&comparison_json(&config.output_prefix)),
                vec!["baseline_parse_failure", "no_comparable_cases"]
            );
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn enforced_comparison_records_baseline_cases_missing_from_current() {
        let dir = temp_dir("missing-current-case");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![manifest_case("missing", "variant", 128, 100.0)],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert!(rejection_reasons(&report).contains(&"missing_current_case"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_current_cases_missing_from_baseline() {
        let dir = temp_dir("missing-baseline-case");
        let baseline_path = dir.join("baseline.manifest.json");
        let matching = manifest_case("matching", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![matching.clone()],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let current = vec![matching, manifest_case("new-case", "variant", 64, 100.0)];

        assert_eq!(compare_cases_with_baseline(&current, &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(rejection_reasons(&report), vec!["missing_baseline_case"]);
        assert_eq!(report["cases"].as_array().expect("cases").len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_cuda_versions() {
        let dir = temp_dir("cuda-version-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        baseline.device.driver_version = Some("different-driver".to_string());
        baseline.device.toolchain_version = Some("different-cuda-toolkit".to_string());
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["cuda_environment_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_rust_codegen_flags() {
        let dir = temp_dir("rust-codegen-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        let mut baseline_json = serde_json::to_value(&baseline).expect("serialize manifest");
        baseline_json["toolchain"]["rustflags"] =
            serde_json::json!(["-C", "target-cpu=definitely-not-current"]);
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline_json).expect("serialize manifest JSON"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["build_profile_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_accepts_equally_redacted_codegen_paths() {
        let dir = temp_dir("redacted-codegen-path");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let mut baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case],
        );
        let home = std::env::var("HOME").expect("HOME for redaction test");
        baseline.toolchain.rustflags = vec![format!("-Lnative={home}/lib")];
        write_canonical_manifest(&baseline_path, &baseline).expect("write canonical baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_with_baseline(&baseline, &config), 0);
        assert!(rejection_reasons(&comparison_json(&config.output_prefix)).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_cargo_profile_settings() {
        let dir = temp_dir("cargo-profile-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        let mut baseline_json = serde_json::to_value(&baseline).expect("serialize manifest");
        baseline_json["toolchain"]["cargo_lto"] = serde_json::json!("fat");
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline_json).expect("serialize manifest JSON"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        assert!(
            rejection_reasons(&comparison_json(&config.output_prefix))
                .contains(&"build_profile_mismatch")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_cargo_profile_fingerprint() {
        let dir = temp_dir("cargo-profile-fingerprint-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        let mut baseline_json = serde_json::to_value(&baseline).expect("serialize manifest");
        baseline_json["toolchain"]["cargo_profile_fingerprint"] =
            serde_json::json!("definitely-not-current");
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline_json).expect("serialize manifest JSON"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        assert!(
            rejection_reasons(&comparison_json(&config.output_prefix))
                .contains(&"build_profile_mismatch")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_target_triple() {
        let dir = temp_dir("target-triple-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        let mut baseline_json = serde_json::to_value(&baseline).expect("serialize manifest");
        baseline_json["toolchain"]["target_triple"] =
            serde_json::json!("x86_64-unknown-linux-definitely-not-current");
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline_json).expect("serialize manifest JSON"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        assert!(
            rejection_reasons(&comparison_json(&config.output_prefix))
                .contains(&"build_profile_mismatch")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_cuda_codegen_settings() {
        let dir = temp_dir("cuda-codegen-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        let mut baseline_json = serde_json::to_value(&baseline).expect("serialize manifest");
        baseline_json["toolchain"]["cuda_arch"] = serde_json::json!("sm_999");
        baseline_json["toolchain"]["ptx_version"] = serde_json::json!("999.0");
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline_json).expect("serialize manifest JSON"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        assert!(
            rejection_reasons(&comparison_json(&config.output_prefix))
                .contains(&"build_profile_mismatch")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_mismatched_cpu_identity() {
        let dir = temp_dir("cpu-identity-mismatch");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("same-name", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        let mut baseline_json = serde_json::to_value(&baseline).expect("serialize manifest");
        baseline_json["toolchain"]["cpu_model"] =
            serde_json::Value::String("definitely-not-the-current-cpu".to_string());
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline_json).expect("serialize manifest JSON"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };

        assert_eq!(compare_cases_with_baseline(&[baseline_case], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(
            rejection_reasons(&report),
            vec!["host_identity_mismatch", "no_comparable_cases"]
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_insufficient_samples() {
        let dir = temp_dir("insufficient-current-samples");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("under-sampled", "variant", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 1,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let mut current = baseline_case;
        current.samples_us = vec![3_000.0];
        current.samples = 1;

        assert_eq!(compare_cases_with_baseline(&[current], &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert_eq!(report["cases"][0]["class"], "insufficient_samples");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_duplicate_baseline_case_names() {
        let dir = temp_dir("duplicate-baseline");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![
                manifest_case("duplicate", "variant-a", 128, 100.0),
                manifest_case("duplicate", "variant-b", 256, 100.0),
            ],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let current = vec![manifest_case("duplicate", "variant-a", 128, 100.0)];

        assert_eq!(compare_cases_with_baseline(&current, &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert!(rejection_reasons(&report).contains(&"duplicate_baseline_name"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn enforced_comparison_rejects_duplicate_current_case_names() {
        let dir = temp_dir("duplicate-current");
        let baseline_path = dir.join("baseline.manifest.json");
        let baseline_case = manifest_case("duplicate", "variant-a", 128, 100.0);
        let baseline = BenchmarkManifest::new(
            myelin_accelerator::bench::RunTiming {
                warmup: 1,
                samples: 8,
                seed: None,
            },
            vec![baseline_case.clone()],
        );
        std::fs::write(
            &baseline_path,
            serde_json::to_vec(&baseline).expect("serialize manifest"),
        )
        .expect("write baseline");
        let config = Config {
            warmup: 1,
            iterations: 8,
            baseline: Some(baseline_path.to_string_lossy().into_owned()),
            output_prefix: dir.join("current").to_string_lossy().into_owned(),
            enforce_budget: true,
            budget: RegressionBudget::default(),
        };
        let current = vec![baseline_case.clone(), baseline_case];

        assert_eq!(compare_cases_with_baseline(&current, &config), 1);
        let report = comparison_json(&config.output_prefix);
        assert_eq!(report["gate_passed"], false);
        assert!(rejection_reasons(&report).contains(&"duplicate_current_name"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn legacy_json_output_does_not_follow_final_path_symlink() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir("legacy-json-symlink");
        let prefix = dir.join("results");
        let output = prefix.with_extension("json");
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, "do not overwrite").expect("victim");
        symlink(&victim, &output).expect("output symlink");
        let report = BenchmarkReport {
            timestamp: "2026-09-22T00:00:00Z".to_string(),
            gpu_info: None,
            config: RunConfig {
                warmup: 1,
                iterations: 8,
            },
            results: vec![legacy_result("case", 100.0)],
        };

        write_json(&report, &prefix.to_string_lossy());

        assert_eq!(
            std::fs::read_to_string(&victim).expect("read victim"),
            "do not overwrite"
        );
        assert!(
            !std::fs::symlink_metadata(&output)
                .expect("output metadata")
                .file_type()
                .is_symlink()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn legacy_csv_output_does_not_follow_final_path_symlink() {
        use std::os::unix::fs::symlink;

        let dir = temp_dir("legacy-csv-symlink");
        let prefix = dir.join("results");
        let output = prefix.with_extension("csv");
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, "do not overwrite").expect("victim");
        symlink(&victim, &output).expect("output symlink");

        write_csv(&[legacy_result("case", 100.0)], &prefix.to_string_lossy());

        assert_eq!(
            std::fs::read_to_string(&victim).expect("read victim"),
            "do not overwrite"
        );
        assert!(
            !std::fs::symlink_metadata(&output)
                .expect("output metadata")
                .file_type()
                .is_symlink()
        );
        let _ = std::fs::remove_dir_all(dir);
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
