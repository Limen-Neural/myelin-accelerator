# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Archive boundary checks for the v0.2.0 Cargo release preflight."""

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_crate import REQUIRED_PATHS, inspect_archive  # noqa: E402


class ArchiveBoundaryTests(unittest.TestCase):
    def archive(self, root: Path, paths: set[str]) -> Path:
        output = root / "myelin-accelerator-0.2.0.crate"
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(paths):
                payload = b"fixture\n"
                info = tarfile.TarInfo(f"myelin-accelerator-0.2.0/{path}")
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


if __name__ == "__main__":
    unittest.main()
