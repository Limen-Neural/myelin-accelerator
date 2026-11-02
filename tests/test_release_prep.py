# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Archive boundary checks for the v0.2.0 Cargo release preflight."""

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_crate import REQUIRED_PATHS, check_tracked_inventory, inspect_archive, prepare_archive  # noqa: E402


class ArchiveBoundaryTests(unittest.TestCase):
    def archive(self, root: Path, paths: set[str], symlink: str | None = None) -> Path:
        output = root / "myelin-accelerator-0.2.0.crate"
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(paths):
                payload = b"fixture\n"
                info = tarfile.TarInfo(f"myelin-accelerator-0.2.0/{path}")
                if path == symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../outside"
                    archive.addfile(info)
                    continue
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return output

    def test_accepts_minimal_required_source_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            files = inspect_archive(self.archive(Path(temp), set(REQUIRED_PATHS)))
        self.assertEqual(files, set(REQUIRED_PATHS))

    def test_rejects_missing_cuda_header(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(Path(temp), set(REQUIRED_PATHS) - {"cu/common.cuh"})
            with self.assertRaisesRegex(ValueError, "cu/common.cuh"):
                inspect_archive(archive)

    def test_rejects_credential_shaped_file(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(Path(temp), set(REQUIRED_PATHS) | {"src/.env"})
            with self.assertRaisesRegex(ValueError, "src/.env"):
                inspect_archive(archive)

    def test_rejects_path_outside_expected_crate_root(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(Path(temp), set(REQUIRED_PATHS) | {"../outside.txt"})
            with self.assertRaisesRegex(ValueError, "outside crate root"):
                inspect_archive(archive)

    def test_rejects_absolute_path_inside_archive_root(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(Path(temp), set(REQUIRED_PATHS) | {"/outside.txt"})
            with self.assertRaisesRegex(ValueError, "outside crate root"):
                inspect_archive(archive)

    def test_rejects_untracked_file_even_under_an_included_directory(self):
        with self.assertRaisesRegex(ValueError, "src/ignored.rs"):
            check_tracked_inventory(set(REQUIRED_PATHS) | {"src/ignored.rs"}, set(REQUIRED_PATHS))

    def test_allows_cargo_generated_metadata(self):
        check_tracked_inventory(
            set(REQUIRED_PATHS) | {".cargo_vcs_info.json", "Cargo.toml.orig"},
            set(REQUIRED_PATHS),
        )

    def test_rejects_common_private_key_name(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(Path(temp), set(REQUIRED_PATHS) | {"src/id_ed25519"})
            with self.assertRaisesRegex(ValueError, "id_ed25519"):
                inspect_archive(archive)

    def test_rejects_symlink_member(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(Path(temp), set(REQUIRED_PATHS) | {"src/linked.rs"}, "src/linked.rs")
            with self.assertRaisesRegex(ValueError, "non-file archive member"):
                inspect_archive(archive)

    def test_prepare_archive_runs_package_boundary_checker(self):
        class Context:
            def __init__(self, root: Path):
                self.env = {"CARGO_TARGET_DIR": str(root / "target")}
                self.output = root / "output"
                self.output.mkdir()
                self.pipelines: list[tuple[str, list[str], list[str], Path]] = []

            def run(self, _name: str, _args: list[str], _cwd: Path) -> None:
                # Archive creation and copying are mocked; only pipeline wiring is under test.
                return None

            def run_pipeline(self, name: str, producer: list[str], consumer: list[str], cwd: Path) -> None:
                self.pipelines.append((name, producer, consumer, cwd))

        with tempfile.TemporaryDirectory() as temp:
            context = Context(Path(temp))
            tracked = "\0".join(sorted(REQUIRED_PATHS)).encode() + b"\0"
            with (
                patch("prepare_crate.inspect_archive", return_value=set(REQUIRED_PATHS)),
                patch("prepare_crate.subprocess.check_output", return_value=tracked),
                patch("prepare_crate.shutil.copyfile"),
            ):
                prepare_archive(context)

        self.assertEqual(len(context.pipelines), 1)
        name, producer, consumer, _cwd = context.pipelines[0]
        self.assertEqual(name, "package-boundary")
        self.assertEqual(producer, ["cargo", "package", "--locked", "--list"])
        self.assertEqual(Path(consumer[0]).name, "check-package-boundary.sh")


if __name__ == "__main__":
    unittest.main()
