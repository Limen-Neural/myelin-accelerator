# SNN model fixtures (LIM-1462 / #44)

Pinned, offline fixtures used by `tests/snn_fixtures.rs` (CPU) and
`tests/snn_fixtures_gpu.rs` (CUDA, `#[ignore]`). Both are classified
**workload-only** for v0.2.0: they supply real model shapes, weights, and
parameters, but the tests execute myelin's fixed v0.2.0 LIF dynamics, not the
source models' own dynamics. See `docs/SNN_COMPATIBILITY.md`.

| Directory | Source | Checked-in content |
|-----------|--------|--------------------|
| `spikenaut/` | `rmems/Spikenaut-SNN` @ `6965e12a` (HF `da93893d`) | `parameters_weights.mem` byte-for-byte + `fixture.json` |
| `synfire_lifneuron/` | Synfire `pabogdan/lifneuron:1.0.0` (= NIR `paper/01_lif/lif_norse.nir` @ `da0551da`) | `fixture.json` with derived scalars only; raw `model.nir` is **not** checked in |

Every checksum, revision, orientation and parameter in the `fixture.json`
files was measured from the pinned artifacts. Re-check with:

```bash
python3 scripts/snn_fixtures/verify_fixtures.py            # offline
python3 scripts/snn_fixtures/verify_fixtures.py --network  # re-download + audit (needs h5py for NIR params)
```

Do not edit fixture bytes or recorded values by hand. If an upstream revision
changes, re-pin it deliberately, rerun the network audit, and record the new
values.

## Attribution

- `spikenaut/parameters_weights.mem` — from Spikenaut-SNN,
  Copyright (c) 2026 Raul Montoya Cardenas, dual-licensed MIT OR Apache-2.0
  (the same terms as this crate; see `LICENSE-MIT` / `LICENSE-APACHE` in the
  upstream repository).
- `synfire_lifneuron/fixture.json` — scalar parameters and the `d0` stimulus
  derived from the NIR project's `paper/01_lif` example (BSD-3-Clause, NIR
  Team), published on Synfire as `pabogdan/lifneuron:1.0.0` (BSD-3-Clause).
  Cite: Pedersen et al., *Neuromorphic Intermediate Representation*,
  Nature Communications (2024), doi:10.1038/s41467-024-52259-9.
