// Copyright 2026 Raul Mc
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reproducible GPU benchmark harness for myelin-accelerator kernels.
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
//! # Compare against a baseline
//! cargo run --example benchmark --features bench -- --baseline results.json
//!
//! # Custom iteration counts
//! cargo run --example benchmark --features bench -- --warmup 20 --iterations 200
//! ```
//!
//! # Output
//!
//! Emits `benchmark_results.json` and `benchmark_results.csv` in the current
//! directory. Results include latency percentiles (p50, p95, p99), throughput,
//! and GPU info when available.
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

use std::time::{Duration, Instant};

// ── CLI argument parsing (minimal, no clap dependency) ──────────────────────

struct Config {
    warmup: usize,
    iterations: usize,
    baseline: Option<String>,
    output_prefix: String,
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
        Config {
            warmup: raw.warmup.unwrap_or(10),
            iterations: raw.iterations.unwrap_or(100),
            baseline: raw.baseline,
            output_prefix: raw
                .output_prefix
                .unwrap_or_else(|| "benchmark_results".to_string()),
        }
    }
}

#[derive(Default)]
struct RawConfig {
    warmup: Option<usize>,
    iterations: Option<usize>,
    baseline: Option<String>,
    output_prefix: Option<String>,
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

fn print_help() {
    println!("Usage: benchmark [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --warmup <N>        Warmup iterations (default: 10)");
    println!("  --iterations <N>    Timed iterations (default: 100)");
    println!("  --baseline <FILE>   Compare against a previous JSON result");
    println!("  --output <PREFIX>   Output file prefix (default: benchmark_results)");
    println!("  -h, --help          Show this help");
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

fn run_benchmark<F: FnMut()>(
    name: &str,
    warmup: usize,
    iterations: usize,
    mut f: F,
) -> BenchmarkResult {
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
    }
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

fn bench_bitpacking(config: &Config) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    results.extend(bench_binary_pack(config));
    results.extend(bench_binary_unpack(config));
    results.extend(bench_ternary_pack(config));
    results.extend(bench_ternary_unpack(config));
    results
}

fn bench_binary_pack(config: &Config) -> Vec<BenchmarkResult> {
    use myelin_accelerator::bitpacking::pack_binary;

    let small: Vec<bool> = (0..256).map(|i| i % 3 == 0).collect();
    let large: Vec<bool> = (0..65536).map(|i| i % 5 < 2).collect();

    vec![
        run_benchmark(
            "bitpack_binary_pack_256",
            config.warmup,
            config.iterations,
            || {
                let _ = pack_binary(&small);
            },
        ),
        run_benchmark(
            "bitpack_binary_pack_65536",
            config.warmup,
            config.iterations,
            || {
                let _ = pack_binary(&large);
            },
        ),
    ]
}

fn bench_binary_unpack(config: &Config) -> Vec<BenchmarkResult> {
    use myelin_accelerator::bitpacking::{pack_binary, unpack_binary};

    let small_src: Vec<bool> = (0..256).map(|i| i % 3 == 0).collect();
    let large_src: Vec<bool> = (0..65536).map(|i| i % 5 < 2).collect();
    let small = pack_binary(&small_src);
    let large = pack_binary(&large_src);

    vec![
        run_benchmark(
            "bitpack_binary_unpack_256",
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_binary(&small, Some(256));
            },
        ),
        run_benchmark(
            "bitpack_binary_unpack_65536",
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_binary(&large, Some(65536));
            },
        ),
    ]
}

fn bench_ternary_pack(config: &Config) -> Vec<BenchmarkResult> {
    use myelin_accelerator::bitpacking::pack_ternary;

    let small: Vec<i8> = ternary_pattern(256);
    let large: Vec<i8> = ternary_pattern(65536);

    vec![
        run_benchmark(
            "bitpack_ternary_pack_256",
            config.warmup,
            config.iterations,
            || {
                let _ = pack_ternary(&small);
            },
        ),
        run_benchmark(
            "bitpack_ternary_pack_65536",
            config.warmup,
            config.iterations,
            || {
                let _ = pack_ternary(&large);
            },
        ),
    ]
}

fn bench_ternary_unpack(config: &Config) -> Vec<BenchmarkResult> {
    use myelin_accelerator::bitpacking::{pack_ternary, unpack_ternary};

    let small = pack_ternary(&ternary_pattern(256));
    let large = pack_ternary(&ternary_pattern(65536));

    vec![
        run_benchmark(
            "bitpack_ternary_unpack_256",
            config.warmup,
            config.iterations,
            || {
                let _ = unpack_ternary(&small, Some(256));
            },
        ),
        run_benchmark(
            "bitpack_ternary_unpack_65536",
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
fn bench_gpu_kernels(config: &Config) -> Vec<BenchmarkResult> {
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
    results.push(run_benchmark(
        "poisson_encode_4096",
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
    results.push(run_benchmark(
        "satsolver_extract_1024x256",
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
fn bench_ternary_gpu(
    acc: &myelin_accelerator::GpuAccelerator,
    config: &Config,
) -> Vec<BenchmarkResult> {
    use myelin_accelerator::GpuBuffer;
    use myelin_accelerator::bitpacking::{
        DEFAULT_GROUP_SIZE, pack_ternary_matrix, uniform_group_scales,
    };

    let mut results = Vec::new();

    // Mid-size GEMV: M=1024, K=4096 (embedding-ish strip; not full vocab)
    let m = 1024usize;
    let k = 4096usize;
    let group = DEFAULT_GROUP_SIZE;
    let weights: Vec<i8> = (0..m * k)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect();
    let packed = pack_ternary_matrix(&weights, m, k);
    let scales = uniform_group_scales(m, k, group, 1.0);
    let x = vec![0.01f32; k];
    let d_w = GpuBuffer::from_slice(&packed).unwrap();
    let d_s = GpuBuffer::from_slice(&scales).unwrap();
    let d_x = GpuBuffer::from_slice(&x).unwrap();
    let mut d_y = GpuBuffer::<f32>::alloc(m).unwrap();

    results.push(run_benchmark(
        "ternary_gemv_1024x4096",
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

    results.push(run_benchmark(
        "ternary_gemv_1024x4096_skip_zeros",
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
    let weights2: Vec<i8> = (0..m2 * k2)
        .map(|i| match i % 3 {
            0 => 0i8,
            1 => 1i8,
            _ => -1i8,
        })
        .collect();
    let packed2 = pack_ternary_matrix(&weights2, m2, k2);
    let scales2 = uniform_group_scales(m2, k2, group, 1.0);
    let b = vec![0.01f32; k2 * n2];
    let d_w2 = GpuBuffer::from_slice(&packed2).unwrap();
    let d_s2 = GpuBuffer::from_slice(&scales2).unwrap();
    let d_b = GpuBuffer::from_slice(&b).unwrap();
    let mut d_c = GpuBuffer::<f32>::alloc(m2 * n2).unwrap();

    results.push(run_benchmark(
        "ternary_gemm_256x1024x64",
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
    let dense_w: Vec<f32> = weights.iter().map(|&t| t as f32).collect();
    results.push(run_benchmark(
        "dense_f32_gemv_1024x4096_host",
        config.warmup,
        config.iterations.min(50),
        || {
            let mut y = vec![0.0f32; m];
            for mi in 0..m {
                let mut acc_v = 0.0f32;
                let row = mi * k;
                for ki in 0..k {
                    acc_v += dense_w[row + ki] * x[ki];
                }
                y[mi] = acc_v;
            }
            std::hint::black_box(y);
        },
    ));

    results
}

#[cfg(not(feature = "cuda"))]
fn bench_gpu_kernels(_config: &Config) -> Vec<BenchmarkResult> {
    eprintln!("[bench] Built without cuda feature, skipping GPU kernel benchmarks");
    Vec::new()
}

// ── Baseline comparison ─────────────────────────────────────────────────────

fn compare_with_baseline(current: &[BenchmarkResult], baseline_path: &str) {
    let Ok(data) = std::fs::read_to_string(baseline_path) else {
        eprintln!("[bench] Could not read baseline file: {baseline_path}");
        return;
    };
    let Ok(report) = serde_json::from_str::<BenchmarkReport>(&data) else {
        eprintln!("[bench] Could not parse baseline JSON");
        return;
    };

    println!("\n{:=>70}", "");
    println!("  Baseline comparison: {baseline_path}");
    println!("{:=>70}\n", "");

    println!(
        "{:<40} {:>12} {:>12} {:>10}",
        "Benchmark", "Baseline(µs)", "Current(µs)", "Change"
    );
    println!("{:-<76}", "");

    for curr in current {
        if let Some(base) = report.results.iter().find(|r| r.name == curr.name) {
            let change = if base.mean_us > 0.0 {
                ((curr.mean_us - base.mean_us) / base.mean_us) * 100.0
            } else {
                0.0
            };
            let indicator = if change > 5.0 {
                "⚠ SLOWER"
            } else if change < -5.0 {
                "✓ faster"
            } else {
                "  ~same"
            };
            println!(
                "{:<40} {:>12.2} {:>12.2} {:>+8.1}% {}",
                curr.name, base.mean_us, curr.mean_us, change, indicator
            );
        }
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

    let mut results = Vec::new();
    results.extend(bench_bitpacking(&config));
    results.extend(bench_gpu_kernels(&config));

    print_results(&results);

    let report = BenchmarkReport {
        timestamp: chrono_now(),
        gpu_info,
        config: RunConfig {
            warmup: config.warmup,
            iterations: config.iterations,
        },
        results: results.clone(),
    };

    write_json(&report, &config.output_prefix);
    write_csv(&results, &config.output_prefix);

    if let Some(ref baseline) = config.baseline {
        compare_with_baseline(&results, baseline);
    }

    println!("[bench] Done.");
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
