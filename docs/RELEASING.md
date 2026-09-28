# crates.io and Linear releases

The `myelin-accelerator` Linear pipeline is a **scheduled production** pipeline.
Normal pushes and merges to `main` do not report a release. The
[`Report published crate to Linear`](../.github/workflows/linear-release.yml)
workflow runs when GitHub marks a release as stable, including when a published
prerelease is promoted to stable. Metadata edits do not report a release.

After the v0.2.0 qualification and publication steps tracked in
[LIM-1443](https://linear.app/rpd-34/issue/LIM-1443/release-qualify-myelin-accelerator-v020-for-cratesio),
publish a GitHub release for the `v<crate version>` tag. The tag must point to
the **exact commit used to package and publish the crate**; check this before
publishing the GitHub release. The existing `v0.2.0` candidate tag points to an
older commit, so verify or correct its target as part of that qualification.
Do not publish the GitHub release before the crate is available on crates.io.

The workflow checks the tag against `Cargo.toml`, records the checked-out tag
commit SHA, and waits for the matching, unyanked crates.io version. It verifies
the downloaded crate archive against crates.io's checksum and requires its
Cargo VCS revision to match the clean tag commit before reporting to Linear.
It scans commits since the previous published, stable production version tag
(including commits before unpublished candidate and prerelease tags). For the
first release, it scans the full history, including the repository's first
commit. Put LIM identifiers in shipped commit subjects or link the corresponding
pull requests to Linear issues. The official Linear action
then syncs that version and its issues to the pipeline, links the crate,
commit, and GitHub release, and completes the same version. Linear's pipeline
settings generate release notes and move open issues on completion.

The workflow fails if the crate is unavailable, Linear sync produces no matching
release, or completion fails. Check the Actions run and retry it after fixing
the cause; a failed run must not be treated as a completed Linear release. The
pipeline key is supplied only through the existing `LINEAR_ACCESS_KEY` Actions
secret. Keep the release history and tag available for issue attribution.
