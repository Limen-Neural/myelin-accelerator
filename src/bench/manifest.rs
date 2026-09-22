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
        rustc: option_env!("MYELIN_BUILD_RUSTC_VERSION").map(str::to_string),
        nvcc: option_env!("MYELIN_BUILD_NVCC_VERSION").map(str::to_string),
        host_arch: std::env::consts::ARCH.to_string(),
        host_os: std::env::consts::OS.to_string(),
        opt_level: option_env!("OPT_LEVEL").unwrap_or("unknown").to_string(),
        debug_assertions: cfg!(debug_assertions),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Probe `nvidia-smi` for UUID, driver, and configured power/clock controls.
///
/// Stable identity/power fields and optional application clocks are queried
/// separately so drivers that reject the deprecated clock fields do not erase
/// otherwise available provenance.
pub fn probe_power_clock() -> (Option<String>, Option<String>, PowerClockControls) {
    probe_power_clock_with_command(Path::new("nvidia-smi"))
}

fn smi_query(binary: &Path, fields: &str) -> Option<String> {
    Command::new(binary)
        .args([
            &format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
}

fn probe_power_clock_with_command(
    binary: &Path,
) -> (Option<String>, Option<String>, PowerClockControls) {
    probe_power_clock_with_query(|fields| smi_query(binary, fields))
}

fn probe_power_clock_with_query(
    mut query: impl FnMut(&str) -> Option<String>,
) -> (Option<String>, Option<String>, PowerClockControls) {
    let mut controls = PowerClockControls::unavailable();
    let mut uuid = None;
    let mut driver_version = None;

    if let Some(raw) = query("uuid,driver_version,persistence_mode,power.limit") {
        let parts: Vec<&str> = raw
            .lines()
            .next()
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .collect();
        if parts.len() >= 4 {
            uuid = optional_smi(parts[0]);
            driver_version = optional_smi(parts[1]);
            controls.persistence_mode = optional_smi(parts[2]);
            controls.power_limit_w = optional_smi(parts[3]).and_then(|s| s.parse().ok());
        }
    }

    if let Some(raw) = query("clocks.applications.graphics,clocks.applications.memory") {
        let parts: Vec<&str> = raw
            .lines()
            .next()
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .collect();
        if parts.len() >= 2 {
            controls.graphics_clock_mhz = optional_smi(parts[0]).and_then(|s| s.parse().ok());
            controls.memory_clock_mhz = optional_smi(parts[1]).and_then(|s| s.parse().ok());
        }
    }

    (uuid, driver_version, controls)
}

/// True when `a` and `b` name the same filesystem path after resolving `.` /
/// `..` and canonicalizing existing components. Used so `--output ./x` cannot
/// clobber a `--baseline x` file via raw `Path` inequality.
pub fn paths_refer_to_same_file(a: &Path, b: &Path) -> bool {
    resolved_path(a) == resolved_path(b)
}

fn resolved_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let file_name = path.file_name();
    if let Some(name) = file_name {
        if parent.as_os_str().is_empty() {
            if let Ok(cwd) = std::env::current_dir().and_then(|d| d.canonicalize()) {
                return cwd.join(name);
            }
        } else if let Ok(parent_c) = parent.canonicalize() {
            return parent_c.join(name);
        }
    }
    lexical_absolute(path)
}

fn lexical_absolute(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut out = PathBuf::new();
    for component in abs.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                let _ = out.pop();
            }
            rest => out.push(rest.as_os_str()),
        }
    }
    out
}

fn write_atomic(path: &Path, contents: impl AsRef<[u8]>) -> std::io::Result<()> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("manifest.json"));
    let tmp = dir.join(format!(
        ".{}.tmp.{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    if let Err(err) = std::fs::write(&tmp, contents) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            Err(err)
        }
    }
}

/// Write any serializable value as redacted, key-sorted pretty JSON.
///
/// Bytes land in a same-directory temp file and are renamed into place so an
/// interrupt cannot leave a truncated artifact.
pub fn write_canonical_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let ctx = RedactionContext::from_env();
    let text = redact_and_canonicalize(value, &ctx)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    write_atomic(path, text)
}

/// Write a redacted, key-sorted pretty JSON manifest atomically.
pub fn write_canonical_manifest(path: &Path, manifest: &BenchmarkManifest) -> std::io::Result<()> {
    write_canonical_json(path, manifest)
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

    #[test]
    fn dotted_relative_paths_refer_to_same_file() {
        assert!(paths_refer_to_same_file(
            Path::new("./bench.manifest.json"),
            Path::new("bench.manifest.json"),
        ));
        assert!(paths_refer_to_same_file(
            Path::new("nested/../bench.manifest.json"),
            Path::new("bench.manifest.json"),
        ));
        assert!(!paths_refer_to_same_file(
            Path::new("a.manifest.json"),
            Path::new("b.manifest.json"),
        ));
    }

    #[test]
    fn write_canonical_manifest_replaces_via_rename() {
        let dir = std::env::temp_dir().join(format!(
            "myelin-manifest-atomic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("out.manifest.json");
        std::fs::write(&path, "{truncated").expect("pre-existing truncated file");
        let manifest = BenchmarkManifest::new(
            RunTiming {
                warmup: 0,
                samples: 0,
                seed: None,
            },
            vec![],
        );
        write_canonical_manifest(&path, &manifest).expect("atomic write");
        let text = std::fs::read_to_string(&path).expect("read");
        serde_json::from_str::<BenchmarkManifest>(&text).expect("complete JSON");
        let leftover = std::fs::read_dir(&dir)
            .expect("list")
            .filter_map(|e| e.ok())
            .count();
        assert_eq!(leftover, 1, "temp sibling must be renamed away");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn power_clock_probe_preserves_stable_fields_when_application_clocks_fail() {
        let (uuid, driver_version, controls) =
            probe_power_clock_with_query(|fields| match fields {
                "uuid,driver_version,persistence_mode,power.limit" => {
                    Some("GPU-test, 610.43.03, Enabled, 360.00\n".to_string())
                }
                "clocks.applications.graphics,clocks.applications.memory" => None,
                unexpected => panic!("unexpected nvidia-smi query: {unexpected}"),
            });

        assert_eq!(uuid.as_deref(), Some("GPU-test"));
        assert_eq!(driver_version.as_deref(), Some("610.43.03"));
        assert_eq!(controls.persistence_mode.as_deref(), Some("Enabled"));
        assert_eq!(controls.graphics_clock_mhz, None);
        assert_eq!(controls.memory_clock_mhz, None);
        assert_eq!(controls.power_limit_w, Some(360.0));
    }

    #[test]
    fn toolchain_versions_are_the_compilers_recorded_at_build_time() {
        let toolchain = capture_toolchain();

        assert!(
            toolchain
                .rustc
                .as_deref()
                .is_some_and(|version| version.starts_with("rustc "))
        );
        if cfg!(feature = "cuda") {
            assert!(
                toolchain
                    .nvcc
                    .as_deref()
                    .is_some_and(|version| version.contains("release "))
            );
        } else {
            assert_eq!(toolchain.nvcc, None);
        }
    }
}
