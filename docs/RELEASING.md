# Crate preparation and Linear release reporting

## Prepare and qualify a candidate

v0.2.0 was [published](https://crates.io/crates/myelin-accelerator/0.2.0) from qualified commit `6cfb49c60fb7f95818e87d0ea0e1ce76c2360fb5`; [LIM-1460](https://linear.app/rpd-34/issue/LIM-1460/release-publish-myelin-accelerator-v020-foundation) and [GH #37](https://github.com/Limen-Neural/myelin-accelerator/issues/37#issuecomment-5945443818) retain the evidence. Merging a later PR does not transfer qualification to its new commit. The preflight below is pinned to v0.2.0 for reproduction; update its version contract before using it for a future release. It prepares a candidate but never uploads or tags.

The manifest has an explicit source allowlist. Cargo must include Rust and CUDA sources, the common CUDA header, build script, examples, licenses, lockfile, documentation, tests, pinned SNN fixtures, and verification scripts. Local metadata, generated PTX, and credentials do not belong in the crate. The preflight inspects the actual archive and compares its inventory with tracked files.

On a clean checkout with Python 3.11.4+, CUDA toolkit 13.2+, Compute Sanitizer (or `COMPUTE_SANITIZER` pointing to it), and an `sm_120` device, run:

```bash
python3 -B -m unittest discover -s tests -p test_release_prep.py -v
python3 -B scripts/prepare_crate.py \
  --candidate-sha "$(git rev-parse HEAD)"
```

The preflight prints a fresh output directory. It records Cargo's file list, creates and inspects a `.crate`, runs `cargo publish --dry-run --locked`, tests the extracted archive on CPU and GPU, runs both lifecycle suites under Compute Sanitizer, and runs separate CPU/CUDA consumers. `summary.json` records the full commit SHA, archive SHA-256, and file list; adjacent logs record each command's output. CI also uploads the archive and logs as a candidate artifact. A failure prevents a success summary.

Review the archive, logs, final CI and review state on the **exact** candidate commit. Run the remaining release gate in [REVIEW.md](../REVIEW.md) §6–§7 and record the pin and evidence in the issue for that version before starting its publication transaction. Keep earlier qualification evidence attached to its original SHA.

## Publish, then report to Linear

The `myelin-accelerator` Linear pipeline is a scheduled production pipeline. Normal pushes and merges to `main` do not report a release. After exact-commit qualification, publish the crate to crates.io and verify its metadata and archive. Then publish a stable GitHub release for the `v<crate version>` tag pointing to the **same commit** used to package and publish the crate. Do not publish the GitHub release before the crate is available on crates.io. The published `v0.2.0` tag points to the qualified `6cfb49c` commit.

The [release-reporting workflow](https://github.com/Limen-Neural/myelin-accelerator/blob/main/.github/workflows/linear-release.yml) runs when GitHub releases a stable version, including promotion from prerelease. It checks the tag against `Cargo.toml`, verifies the downloaded crate's checksum and clean Cargo VCS revision, and checks that earlier stable releases were successfully reported. It scans commits since the latest reported stable release, including commits before unpublished candidate or prerelease tags. On a first release, the repository root commit is the exclusive base; it has no LIM identifier, so all shipped issue-bearing commits are scanned. Put LIM identifiers in shipped commit subjects or link their PRs to Linear issues.

The pinned Linear action syncs that version and its shipped issues, links the crate, commit, and GitHub release, and completes the same production version. A missing crate, mismatched revision, failed prior report, or failed Linear operation fails the job. The first [v0.2.0 reporting run](https://github.com/Limen-Neural/myelin-accelerator/actions/runs/36964728688) failed before sync because its synthetic base was not an ancestor; the intended release was reconciled manually. The workflow has a narrow exception for that exact tag/SHA, recorded in [GH #60](https://github.com/Limen-Neural/myelin-accelerator/issues/60), while later releases still require successful reports. The workflow reads the existing `LINEAR_ACCESS_KEY` Actions secret only during post-publication reporting. Before any manual CLI recovery, confirm that the local access key belongs to the intended pipeline; a workspace key for another pipeline can mutate the wrong release.
