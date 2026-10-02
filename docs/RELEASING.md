# Crate preparation and Linear release reporting

## Prepare and qualify a v0.2.0 candidate

Publication is tracked in [LIM-1460](https://linear.app/rpd-34/issue/LIM-1460/release-publish-myelin-accelerator-v020-foundation). Its recorded qualified commit was `a1a24773a4fc457e778cfc65bb2deba8e0315935`; check the issue for the current release pin before acting. Merging later PRs does not transfer qualification to their new commit. This guide prepares a candidate; it does not upload the crate or tag a release.

The manifest has an explicit source allowlist. Cargo must include Rust and CUDA sources, the common CUDA header, build script, examples, licenses, lockfile, documentation, tests, pinned SNN fixtures, and verification scripts. Local metadata, generated PTX, and credentials do not belong in the crate. The preflight inspects the actual archive and compares its inventory with tracked files.

On a clean checkout with Python 3.11.4+, CUDA toolkit 13.2+, Compute Sanitizer (or `COMPUTE_SANITIZER` pointing to it), and an `sm_120` device, run:

```bash
python3 -B -m unittest discover -s tests -p test_release_prep.py -v
python3 -B scripts/prepare_crate.py \
  --candidate-sha "$(git rev-parse HEAD)"
```

The preflight prints a fresh output directory. It records Cargo's file list, creates and inspects a `.crate`, runs `cargo publish --dry-run --locked`, tests the extracted archive on CPU and GPU, runs both lifecycle suites under Compute Sanitizer, and runs separate CPU/CUDA consumers. `summary.json` records the full commit SHA, archive SHA-256, and file list; adjacent logs record each command's output. CI also uploads the archive and logs as a candidate artifact. A failure prevents a success summary.

Review the archive, logs, final CI and review state on the **exact** candidate commit. Run the remaining release gate in [REVIEW.md](../REVIEW.md) §6–§7 and reconcile the pin and evidence in LIM-1460 before starting the publication transaction. Keep earlier qualification evidence attached to its original SHA.

## Publish, then report to Linear

The `myelin-accelerator` Linear pipeline is a scheduled production pipeline. Normal pushes and merges to `main` do not report a release. After exact-commit qualification, publish the crate to crates.io and verify its metadata and archive. Then publish a stable GitHub release for the `v<crate version>` tag pointing to the **same commit** used to package and publish the crate. The existing `v0.2.0` candidate tag points to an older commit; reconcile its target before publication. Do not publish the GitHub release before the crate is available on crates.io.

The [release-reporting workflow](https://github.com/Limen-Neural/myelin-accelerator/blob/main/.github/workflows/linear-release.yml) runs when GitHub releases a stable version, including promotion from prerelease. It checks the tag against `Cargo.toml`, verifies the downloaded crate's checksum and clean Cargo VCS revision, and checks that earlier stable releases were successfully reported. It scans commits since the latest reported stable release, including commits before unpublished candidate or prerelease tags. For the first release it scans the full history. Put LIM identifiers in shipped commits or link their PRs to Linear issues.

The pinned Linear action syncs that version and its shipped issues, links the crate, commit, and GitHub release, and completes the same production version. A missing crate, mismatched revision, failed prior report, or failed Linear operation fails the job. Inspect and retry that run before calling the Linear release complete. The workflow reads the existing `LINEAR_ACCESS_KEY` Actions secret only during this post-publication report.
