#!/usr/bin/env python3
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Inspect and exercise a v0.2.0 Cargo candidate; never tag or publish it.

Requires Python 3.11.4+ for tomllib and tarfile's safe extraction filter.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11, 4):
    raise SystemExit("prepare_crate.py requires Python 3.11.4 or newer")

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed local Cargo/Git commands, never a shell
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.2.0"
CRATE_NAME = "myelin-accelerator"
ARCHIVE_ROOT = f"{CRATE_NAME}-{VERSION}"
AUTHOR = "Raul Cardenas Montoya"
MANIFEST_NAME = "Cargo.toml"
GIT = shutil.which("git")
REQUIRED_PATHS = frozenset(
    {
        MANIFEST_NAME,
        "Cargo.lock",
        "README.md",
        "REVIEW.md",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        "build.rs",
        "src/lib.rs",
        "src/gpu_stub.rs",
        "src/gpu/accelerator.rs",
        "cu/common.cuh",
        "cu/spiking_network.cu",
        "cu/vector_similarity.cu",
        "cu/satsolver.cu",
        "cu/ternary_gemm.cu",
        "scripts/ptx_entries.txt",
        "scripts/sanitize_lifecycle.sh",
        "examples/benchmark.rs",
        "docs/SNN_COMPATIBILITY.md",
        "tests/fixtures/snn/spikenaut/fixture.json",
        "tests/fixtures/snn/spikenaut/parameters_weights.mem",
        "tests/fixtures/snn/synfire_lifneuron/fixture.json",
    }
)
FORBIDDEN_DIRECTORIES = {".codex", ".agents", "actions-runner", "target", ".git"}
FORBIDDEN_NAMES = {
    "Dockerfile",
    ".dockerignore",
    ".env",
    ".credentials",
    ".credentials_rsaparams",
    "id_rsa",
    "id_ed25519",
    ".pypirc",
}
FORBIDDEN_SUFFIXES = (".pem", ".key", ".p12", ".ptx", ".cubin")


def inspect_archive(archive_path: Path) -> set[str]:
    """Reject missing build inputs and obvious local/credential artifacts."""
    files: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = archive_member_file(member)
            if path is not None:
                files.add(path)
    missing = REQUIRED_PATHS - files
    if missing:
        raise ValueError(f"missing required archive paths: {', '.join(sorted(missing))}")
    return files


def check_tracked_inventory(files: set[str], tracked: set[str]) -> None:
    cargo_generated = {".cargo_vcs_info.json", "Cargo.toml.orig"}
    unexpected = files - tracked - cargo_generated
    if unexpected:
        raise ValueError(f"untracked archive paths: {', '.join(sorted(unexpected))}")


def archive_member_file(member: tarfile.TarInfo) -> str | None:
    if member.isdir() and member.name.rstrip("/") == ARCHIVE_ROOT:
        return None
    path = archive_relative_path(member.name)
    if member.isdir():
        return None
    if not member.isfile():
        raise ValueError(f"non-file archive member: {path}")
    return path


def archive_relative_path(member_name: str) -> str:
    prefix = f"{ARCHIVE_ROOT}/"
    if not member_name.startswith(prefix):
        raise ValueError(f"outside crate root: {member_name}")
    raw_path = member_name[len(prefix) :]
    check_safe_path(raw_path, member_name)
    check_allowed_path(raw_path)
    return raw_path


def check_safe_path(raw_path: str, member_name: str) -> None:
    invalid = not raw_path or raw_path.startswith("/")
    invalid |= bool({"", ".", ".."}.intersection(raw_path.split("/")))
    if invalid:
        raise ValueError(f"outside crate root: {member_name}")


def check_allowed_path(raw_path: str) -> None:
    parts = PurePosixPath(raw_path).parts
    for part in parts[:-1]:
        forbidden = part in FORBIDDEN_DIRECTORIES
        forbidden |= part.startswith("cmake-build-")
        if forbidden:
            raise ValueError(f"forbidden archive path: {raw_path}")
    forbidden_name = parts[-1] in FORBIDDEN_NAMES
    forbidden_name |= raw_path.endswith(FORBIDDEN_SUFFIXES)
    forbidden_name |= parts[-1].startswith(".env")
    forbidden_name |= parts[-1].startswith("id_rsa")
    forbidden_name |= parts[-1].startswith("id_ed25519")
    if forbidden_name:
        raise ValueError(f"forbidden archive path: {raw_path}")


@dataclass
class RunContext:
    env: dict[str, str]
    output: Path

    def run(self, name: str, args: list[str], cwd: Path) -> None:
        log_path = self.output / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(args, cwd=cwd, env=self.env, stdout=log, stderr=subprocess.STDOUT)  # nosec B603 - fixed local commands
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'} ({log_path})", flush=True)
        if result.returncode:
            print("\n".join(log_path.read_text().splitlines()[-30:]), file=sys.stderr)
            raise RuntimeError(f"{name} failed with exit {result.returncode}")


def check_extracted_crate(unpacked: Path, context: RunContext) -> None:
    manifest = str(unpacked / MANIFEST_NAME)
    context.run("artifact-cpu-tests", ["cargo", "test", "--locked", "--manifest-path", manifest], unpacked)
    context.run("artifact-cuda-build", ["cargo", "build", "--locked", "--features", "cuda", "--manifest-path", manifest], unpacked)
    context.run("artifact-gpu-tests", ["cargo", "test", "--locked", "--features", "cuda", "--manifest-path", manifest, "--", "--ignored", "--test-threads=1"], unpacked)
    context.run("artifact-memcheck", ["bash", str(unpacked / "scripts" / "sanitize_lifecycle.sh")], unpacked)


def write_consumer(consumer: Path) -> None:
    (consumer / "src").mkdir(parents=True)
    (consumer / MANIFEST_NAME).write_text(
        "[package]\nname = \"myelin-release-consumer\"\nversion = \"0.0.0\"\nedition = \"2024\"\n"
        "[features]\ncuda = [\"myelin-accelerator/cuda\"]\n"
        "[dependencies]\nmyelin-accelerator = { path = \"../artifact/myelin-accelerator-0.2.0\" }\n",
        encoding="utf-8",
    )
    (consumer / "src" / "main.rs").write_text(
        "use myelin_accelerator::{bitpacking, GpuAccelerator, GpuBuffer};\n"
        "fn main() {\n"
        " assert_eq!(bitpacking::unpack_ternary(&bitpacking::pack_ternary(&[-1, 0, 1]), Some(3)), vec![-1, 0, 1]);\n"
        " #[cfg(not(feature = \"cuda\"))] { assert!(!GpuAccelerator::new().is_ready()); assert_eq!(GpuBuffer::<u32>::alloc(1).unwrap().to_vec().unwrap(), [0]); }\n"
        " #[cfg(feature = \"cuda\")] { let acc = GpuAccelerator::require_gpu().unwrap(); let rates = GpuBuffer::from_slice(&[0.0_f32, 1.0]).unwrap(); let mut spikes = GpuBuffer::<u32>::alloc(2).unwrap(); acc.poisson_encode(&rates, &mut spikes, 42).unwrap(); assert_eq!(spikes.to_vec().unwrap(), [0, 1]); }\n"
        "}\n",
        encoding="utf-8",
    )


def check_consumer(consumer: Path, context: RunContext) -> None:
    write_consumer(consumer)
    manifest = str(consumer / MANIFEST_NAME)
    context.run("consumer-lock", ["cargo", "generate-lockfile", "--manifest-path", manifest], consumer)
    context.run("consumer-cpu", ["cargo", "run", "--locked", "--manifest-path", manifest], consumer)
    context.run("consumer-cuda", ["cargo", "run", "--locked", "--features", "cuda", "--manifest-path", manifest], consumer)


def validate_candidate(parser: argparse.ArgumentParser, candidate_sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
        parser.error("--candidate-sha must be a full lowercase 40-digit SHA")
    if not GIT:
        parser.error("git is required")
    head = subprocess.check_output([GIT, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()  # nosec B603 - fixed local command
    if head != candidate_sha:
        parser.error(f"checkout HEAD {head} differs from candidate {candidate_sha}")
    status = subprocess.check_output([GIT, "status", "--porcelain"], cwd=ROOT, text=True)  # nosec B603 - fixed local command
    if status:
        parser.error("working tree must be clean before preparing a crate")

    package = tomllib.loads((ROOT / MANIFEST_NAME).read_text(encoding="utf-8"))["package"]
    if (package["name"], package["version"]) != (CRATE_NAME, VERSION):
        parser.error(f"expected {CRATE_NAME} {VERSION} in {MANIFEST_NAME}")
    if not any(author.split(" <", 1)[0] == AUTHOR for author in package["authors"]):
        parser.error(f"Cargo authors must include {AUTHOR}")
    return head


def output_directory(parser: argparse.ArgumentParser, requested: Path | None) -> Path:
    output = (requested or Path(tempfile.mkdtemp(prefix="myelin-v0.2.0-prep-"))).resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output-dir must be outside the repository")
    if requested:
        output.mkdir(parents=True, exist_ok=False)
    return output


def cuda_environment(parser: argparse.ArgumentParser) -> dict[str, str]:
    env = os.environ.copy()
    nvcc = Path(env.get("CUDA_NVCC", "/usr/local/cuda/bin/nvcc")).resolve()
    if not nvcc.is_file():
        parser.error(f"nvcc is required for the advertised cuda feature: {nvcc}")
    env["CUDA_NVCC"] = str(nvcc)
    env.setdefault("CUDA_HOME", str(nvcc.parent.parent))
    env.setdefault("CUDA_LIBRARY_PATH", str(nvcc.parent.parent))
    return env


def prepare_archive(context: RunContext) -> tuple[Path, set[str]]:
    context.run("package-list", ["cargo", "package", "--locked", "--list"], ROOT)
    context.run("package", ["cargo", "package", "--locked"], ROOT)
    package_archive = Path(context.env["CARGO_TARGET_DIR"]) / "package" / f"{ARCHIVE_ROOT}.crate"
    files = inspect_archive(package_archive)
    tracked = set(subprocess.check_output([GIT, "ls-files", "-z"], cwd=ROOT).decode().strip("\0").split("\0"))  # nosec B603 - fixed local command
    check_tracked_inventory(files, tracked)
    archived_copy = context.output / package_archive.name
    shutil.copyfile(package_archive, archived_copy)
    context.run("publish-dry-run", ["cargo", "publish", "--dry-run", "--locked"], ROOT)
    return archived_copy, files


def check_archive_consumers(context: RunContext, archived_copy: Path) -> None:
    extracted = context.output / "artifact"
    with tarfile.open(archived_copy, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    check_extracted_crate(extracted / ARCHIVE_ROOT, context)
    check_consumer(context.output / "consumer", context)


def write_summary(output: Path, head: str, archived_copy: Path, files: set[str]) -> None:
    digest = hashlib.sha256(archived_copy.read_bytes()).hexdigest()
    summary = {
        "candidate_sha": head,
        "crate_version": VERSION,
        "cargo_author": AUTHOR,
        "archive_sha256": digest,
        "file_count": len(files),
        "files": sorted(files),
        "status": "preparation passed; no registry upload or tag performed",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {ARCHIVE_ROOT}: {len(files)} files, sha256 {digest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sha", required=True, help="full git SHA to prepare")
    parser.add_argument("--output-dir", type=Path, help="new directory outside the repository")
    args = parser.parse_args()
    head = validate_candidate(parser, args.candidate_sha)
    output = output_directory(parser, args.output_dir)
    print(f"Preparing {head} into {output}", flush=True)
    env = cuda_environment(parser)
    with tempfile.TemporaryDirectory(prefix="myelin-cargo-build-") as build_dir:
        env["CARGO_TARGET_DIR"] = str(Path(build_dir) / "target")
        context = RunContext(env, output)
        archived_copy, files = prepare_archive(context)
        check_archive_consumers(context, archived_copy)
    write_summary(output, head, archived_copy, files)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
