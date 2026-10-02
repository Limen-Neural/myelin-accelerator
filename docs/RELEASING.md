# v0.2.0 crate preparation

Publication is tracked in [LIM-1460](https://linear.app/rpd-34/issue/LIM-1460/release-publish-myelin-accelerator-v020-foundation). Its recorded qualified commit is `a1a24773a4fc457e778cfc65bb2deba8e0315935`. This guide prepares a new candidate; it does not qualify that commit for publication or upload the crate.

The manifest has an explicit source allowlist. Cargo must include the Rust sources, CUDA translation units and common header, build script, examples, licenses, lockfile, documentation, tests, pinned SNN fixtures, and verification scripts. Local metadata, CI configuration, generated PTX, and credentials do not belong in the crate. Check the actual archive rather than relying on the manifest pattern alone.

On a clean checkout with CUDA toolkit 13.2+ and an `sm_120` device, run:

```bash
python3 -m unittest discover -s tests -p test_release_prep.py -v
python3 scripts/prepare_crate.py \
  --candidate-sha "$(git rev-parse HEAD)" \
  --output-dir "/tmp/myelin-v0.2.0-$(git rev-parse --short HEAD)"
```

The preflight records Cargo's file list, creates and inspects a `.crate`, runs `cargo publish --dry-run --locked`, tests the extracted archive on CPU and GPU, runs its two lifecycle suites under Compute Sanitizer, and compiles and runs independent CPU/CUDA path consumers. `summary.json` records the full commit SHA, archive SHA-256, and file list; adjacent logs record each command's output. A failure prevents a success summary.

Review the archive, logs, and final CI/review state on the **exact** candidate commit. Run the remaining release gate in `REVIEW.md` §6–§7 and reconcile the pin and evidence in LIM-1460 before starting a real `cargo publish` transaction. Packaging and dry-run success alone do not replace qualification. Keep the old qualification evidence attached to its original SHA.
