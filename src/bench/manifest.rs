// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Versioned, machine-readable benchmark provenance.

use crate::bench::redact::{RedactionContext, redact_and_canonicalize};
use crate::bench::stats::{SampleStats, sample_stats};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Manifest schema version emitted by this crate.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Provenance + results for one benchmark run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    pub schema_version: u32,
    pub git: GitProvenance,
    pub features: Vec<String>,
    pub toolchain: ToolchainInfo,
    pub device: DeviceIdentity,
    pub power_clock: PowerClockControls,
    pub run: RunTiming,
    pub cases: Vec<ManifestCase>,
}

impl BenchmarkManifest {
    /// Construct a manifest with required fields; optional device data may be empty.
    pub fn new(run: RunTiming, cases: Vec<ManifestCase>) -> Self {
        Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            git: capture_git(),
            features: enabled_features(),
            toolchain: capture_toolchain(),
            device: DeviceIdentity::unavailable(),
            power_clock: PowerClockControls::unavailable(),
            run,
            cases,
        }
    }
}

/// Git commit and dirty flag. Both are `None` when git is unavailable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitProvenance {
    pub commit: Option<String>,
    pub dirty: Option<bool>,
}

/// Compiler / CUDA toolkit identity (versions only, never absolute paths).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolchainInfo {
    pub rustc: Option<String>,
    pub nvcc: Option<String>,
    pub host_arch: String,
    pub host_os: String,
    pub opt_level: String,
    pub debug_assertions: bool,
    pub crate_version: String,
}

/// Device identity. Every field is optional so CPU-only runs stay parseable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceIdentity {
    pub name: Option<String>,
    pub uuid: Option<String>,
    pub compute_capability: Option<String>,
    pub sm_arch: Option<String>,
    pub driver_version: Option<String>,
    pub runtime_version: Option<String>,
    pub toolchain_version: Option<String>,
    pub vram_total_mb: Option<u64>,
}

impl DeviceIdentity {
    pub fn unavailable() -> Self {
        Self {
            name: None,
            uuid: None,
            compute_capability: None,
            sm_arch: None,
            driver_version: None,
            runtime_version: None,
            toolchain_version: None,
            vram_total_mb: None,
        }
    }
}

/// Power / clock knobs from `nvidia-smi` when present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerClockControls {
    pub persistence_mode: Option<String>,
    pub graphics_clock_mhz: Option<f64>,
    pub memory_clock_mhz: Option<f64>,
    pub power_limit_w: Option<f64>,
}

impl PowerClockControls {
    pub fn unavailable() -> Self {
        Self {
            persistence_mode: None,
            graphics_clock_mhz: None,
            memory_clock_mhz: None,
            power_limit_w: None,
        }
    }
}

/// Warmup / sample counts and the run-level RNG seed (when one is used).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunTiming {
    pub warmup: usize,
    pub samples: usize,
    pub seed: Option<u64>,
}

/// One named kernel/workload with input shape and latency samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifestCase {
    pub name: String,
    pub kernel_variant: String,
    pub input_dimensions: BTreeMap<String, i64>,
    pub seed: Option<u64>,
    pub warmup: usize,
    pub samples: usize,
    pub samples_us: Vec<f64>,
    pub median_us: f64,
    pub mad_us: f64,
    pub relative_dispersion: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
}

impl ManifestCase {
    pub fn from_samples(
        name: impl Into<String>,
        kernel_variant: impl Into<String>,
        input_dimensions: BTreeMap<String, i64>,
        seed: Option<u64>,
        warmup: usize,
        samples_us: Vec<f64>,
    ) -> Self {
        let stats: SampleStats = sample_stats(&samples_us);
        let mut sorted = samples_us;
        sorted.sort_by(|a, b| a.total_cmp(b));
        Self {
            name: name.into(),
            kernel_variant: kernel_variant.into(),
            input_dimensions,
            seed,
            warmup,
            samples: stats.n,
            samples_us: sorted,
            median_us: stats.median,
            mad_us: stats.mad,
            relative_dispersion: stats.relative_dispersion,
            mean_us: stats.mean,
            p50_us: stats.p50,
            p95_us: stats.p95,
            p99_us: stats.p99,
            min_us: stats.min,
            max_us: stats.max,
        }
    }
}

/// Cargo features compiled into this crate, sorted.
pub fn enabled_features() -> Vec<String> {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda".into());
    }
    if cfg!(feature = "bench") {
        features.push("bench".into());
    }
    features.sort();
    features
}

/// `git rev-parse` / `git status --porcelain` from the crate root when possible.
pub fn capture_git() -> GitProvenance {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let commit = git_stdout(&dir, &["rev-parse", "HEAD"]).map(|s| s.trim().to_string());
    let dirty = git_stdout(&dir, &["status", "--porcelain"]).map(|s| !s.trim().is_empty());
    GitProvenance { commit, dirty }
}

/// rustc / nvcc versions and host identity. Paths are never stored.
pub fn capture_toolchain() -> ToolchainInfo {
    ToolchainInfo {
        rustc: command_stdout("rustc", &["--version"]).map(trim_line),
        nvcc: nvcc_release(),
        host_arch: std::env::consts::ARCH.to_string(),
        host_os: std::env::consts::OS.to_string(),
        opt_level: option_env!("OPT_LEVEL").unwrap_or("unknown").to_string(),
        debug_assertions: cfg!(debug_assertions),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Probe `nvidia-smi` for UUID and power/clock controls. Missing binary → empty.
pub fn probe_power_clock() -> (Option<String>, PowerClockControls) {
    let Some(raw) = command_stdout(
        "nvidia-smi",
        &[
            "--query-gpu=uuid,persistence_mode,clocks.current.graphics,clocks.current.memory,power.limit",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
    ) else {
        return (None, PowerClockControls::unavailable());
    };
    let line = raw.lines().next().unwrap_or("").trim();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 5 {
        return (None, PowerClockControls::unavailable());
    }
    let uuid = optional_smi(parts[0]);
    let power_clock = PowerClockControls {
        persistence_mode: optional_smi(parts[1]),
        graphics_clock_mhz: optional_smi(parts[2]).and_then(|s| s.parse().ok()),
        memory_clock_mhz: optional_smi(parts[3]).and_then(|s| s.parse().ok()),
        power_limit_w: optional_smi(parts[4]).and_then(|s| s.parse().ok()),
    };
    (uuid, power_clock)
}

/// Write a redacted, key-sorted pretty JSON manifest. Never used as a baseline overwrite helper.
pub fn write_canonical_manifest(path: &Path, manifest: &BenchmarkManifest) -> std::io::Result<()> {
    let ctx = RedactionContext::from_env();
    let text = redact_and_canonicalize(manifest, &ctx)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, text)
}

fn git_stdout(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

fn command_stdout(bin: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

fn trim_line(s: String) -> String {
    s.lines().next().unwrap_or("").trim().to_string()
}

fn nvcc_release() -> Option<String> {
    let binary = nvcc_binary();
    let out = Command::new(binary).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().find_map(|line| {
        line.split("release ")
            .nth(1)
            .and_then(|rest| rest.split(',').next())
            .map(str::trim)
            .map(str::to_string)
    })
}

fn nvcc_binary() -> PathBuf {
    std::env::var("CUDA_NVCC")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            for var in ["CUDA_HOME", "CUDA_PATH"] {
                if let Ok(root) = std::env::var(var)
                    && !root.trim().is_empty()
                {
                    return Some(PathBuf::from(root).join("bin").join("nvcc"));
                }
            }
            None
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"))
}

fn optional_smi(raw: &str) -> Option<String> {
    let t = raw.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("n/a") || t.eq_ignore_ascii_case("[n/a]") {
        None
    } else {
        Some(t.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::redact::canonicalize_json_value;

    #[test]
    fn empty_optional_fields_still_roundtrip() {
        let manifest = BenchmarkManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            git: GitProvenance {
                commit: None,
                dirty: None,
            },
            features: vec![],
            toolchain: ToolchainInfo {
                rustc: None,
                nvcc: None,
                host_arch: "x86_64".into(),
                host_os: "linux".into(),
                opt_level: "0".into(),
                debug_assertions: true,
                crate_version: "0.0.0".into(),
            },
            device: DeviceIdentity::unavailable(),
            power_clock: PowerClockControls::unavailable(),
            run: RunTiming {
                warmup: 0,
                samples: 0,
                seed: None,
            },
            cases: vec![],
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let parsed: BenchmarkManifest = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed.schema_version, MANIFEST_SCHEMA_VERSION);
        assert!(parsed.device.name.is_none());
        assert!(parsed.power_clock.persistence_mode.is_none());
        assert!(parsed.cases.is_empty());
    }

    #[test]
    fn schema_keys_are_stable_when_canonicalized() {
        let manifest = BenchmarkManifest::new(
            RunTiming {
                warmup: 1,
                samples: 2,
                seed: Some(42),
            },
            vec![ManifestCase::from_samples(
                "noop",
                "host-bitpack",
                BTreeMap::from([("n".into(), 4)]),
                Some(42),
                1,
                vec![1.0, 2.0],
            )],
        );
        let value = serde_json::to_value(&manifest).expect("value");
        let ctx = RedactionContext::default();
        let canonical = canonicalize_json_value(value, &ctx);
        let obj = canonical.as_object().expect("object");
        let keys: Vec<_> = obj.keys().cloned().collect();
        assert_eq!(
            keys,
            vec![
                "cases",
                "device",
                "features",
                "git",
                "power_clock",
                "run",
                "schema_version",
                "toolchain",
            ]
        );
        assert_eq!(obj["schema_version"], MANIFEST_SCHEMA_VERSION);
    }
}
