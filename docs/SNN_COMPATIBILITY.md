# SNN fixture compatibility (v0.2.0)

This page records which external SNN models the v0.2.0 kernels are exercised
with, and exactly how far that validation goes. Tracking: GH
[#44](https://github.com/rmems/myelin-accelerator/issues/44) /
[LIM-1462](https://linear.app/rpd-34/issue/LIM-1462).

**Bottom line:** both fixtures are **workload-only**. They supply real model
shapes, weights, and affine parameters to the raw `lif_step` /
`lif_step_weighted` symbols. Both the CPU oracle and the GPU execute the
**fixed myelin v0.2.0 LIF dynamics**. The tests prove the kernels implement
those dynamics bit-exactly on model-shaped, multi-timestep workloads. They do
**not** prove faithful Spikenaut or NIR/Synfire execution. Exact external-model
interoperability is v0.3.0 work, tracked in GH
[#43](https://github.com/rmems/myelin-accelerator/issues/43) /
[LIM-1461](https://linear.app/rpd-34/issue/LIM-1461).

## v0.2.0 LIF contract (what is executed)

Per neuron, per tick (`cu/spiking_network.cu`, mirrored by
`myelin_accelerator::oracle::{lif_step_oracle, lif_step_weighted_oracle}`):

1. If `refract > 0`: `v = 0`, `refract -= 1`, input ignored, no spike.
2. Else `v = fma(0.85, v, I)`. For `lif_step_weighted`, `I` is accumulated
   from `0` in input-index order, `I = fma(w[n,j], x[j], I)`, over a row-major
   `[n_neurons × n_inputs]` matrix.
3. If `v >= 1.0`: spike, `v = 0`, `refract = 2`.

Decay 0.85, threshold 1.0, reset 0.0, and 2 refractory ticks are compile-time
constants. Neither kernel has a stable `GpuAccelerator` wrapper in v0.2.0;
tests launch the registered raw symbols.

## Compatibility matrix

| | Spikenaut-SNN `merged_v2` hidden layer | Synfire `pabogdan/lifneuron:1.0.0` |
|---|---|---|
| Source / revision | [rmems/Spikenaut-SNN](https://github.com/rmems/Spikenaut-SNN) @ `6965e12a6e46d29783e69f6b0fda3dda1058bd29`. Same bytes on HF `rmems/Spikenaut-SNN` @ `da93893d53dc2c28b58d0b48db7bd78a39ba8f8d` | Synfire release `01a01fa8-902e-4517-b47a-bc5fa6b55d2b`. `model.nir` sha256 `0dd9143e…b7b6` is byte-identical to NIR `paper/01_lif/lif_norse.nir` @ `da0551dad03c4818689fdae7a48bfcff0be07b81` |
| Artifact checksum | `parameters_weights.mem` sha256 `825969873444d215d09920e69592700a2cc8594d92e7b14c4059eef8919fe4f3` | `model.nir` `0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6`; `nir-card.json` `1ab401bc3082f3f816cb07222ec3db45ce1d239ee53b6a9247e35ce36b99f59d` |
| License | MIT OR Apache-2.0 (weights checked in verbatim) | BSD-3-Clause (only derived scalars checked in) |
| Topology | 16 inputs → 16 LIF (→ 16×3 readout, not exercised) | `Input(1) → Affine(1) → LIF(1) → Output(1)` |
| Dimensions tested | `n_neurons = 16`, `n_inputs = 16` | `n_neurons = 1`; Affine 1×1 (`w = 1.0`, `b = 0.0`) |
| Timestep | Model: 1 ms (`clock_hz = 1000`). Test: 128 discrete ticks | Model: `dt = 1e-4` s (card / paper; not stored in the graph). Test: 100 ticks (`d0`) and 128 ticks (ramp) |
| Input modality | Model: analog current on axons 0–4. **Test: deterministic 0/1 spikes** on axons 0–4 (`CaseRng` seed 68, rates `[0.35, 0.5, 0.3, 0.6, 0.45]`); axons 5–15 held at 0 (zero-weight unused width) | Model: dense float current through Affine. Test: host Affine `I = w·x + b` on (a) the NIR paper's binary `d0` pattern (before the paper's ×10 zero-upsampling) and (b) a myelin ramp `x[t] = (t+1)/256` |
| Neuron model | Discrete LIF `v = decay·v + W·x` | NIR continuous LIF `tau·dv/dt = (v_leak − v) + r·I` |
| Decay | Model: `0x00DA` = 0.8515625 per neuron. **Executed: 0.85** | Model: `tau = 0.0025` s (≈ 0.96 per 1e-4 s step), `r = 1`, `v_leak = 0`. **Executed: 0.85, no `(1 − decay)` input gain** |
| Threshold | Model: per neuron 1.6015625 (×10), 1.59765625 (×2), 0.44921875 (×4). **Executed: 1.0** | Model: `v_threshold = 0.1`. **Executed: 1.0** |
| Reset | Model: zero baseline (NIR `v_reset` default). Executed: 0.0 | Model: `v_reset` absent from the artifact (NIR default 0). Executed: 0.0 |
| Refractory | Model: none documented. **Executed: 2 ticks** | Model: none in NIR LIF. **Executed: 2 ticks** |
| Weight layout / precision | Signed Q8.8, neuron-major (= row-major `[n_neurons × n_inputs]`, no transpose). Verified 256/256 against `snn_model.json`; transposed reading matches 140/256. Decoded exactly to f32 (`code / 256`) | f32 scalars recorded with exact bit patterns (`tau` `0x3b23d70a`, `v_threshold` `0x3dcccccd`) |
| Kernel exercised | `lif_step_weighted` (raw symbol), block 32 (1 block) and block 8 (2 blocks, strided shared-memory fill), 64 B dynamic shared memory | `lif_step` (raw symbol), block 32 |
| Classification | **workload-only** | **workload-only** |

### Not validated

- Spikenaut's own dynamics: per-neuron decay (0.8515625) and thresholds,
  analog-current input, reset/refractory rules, the 16×3 readout, and its
  telemetry encoder. The real bank uses analog current and its own
  parameters. This test uses the real model shape and weights as a kernel
  workload and does **not** show faithful Spikenaut execution.
- NIR/Synfire dynamics: the continuous-time LIF (`tau`, `r`, `v_leak`,
  `v_threshold = 0.1`), its timestep, and the published voltage/spike traces
  in `paper/01_lif/*.csv`. The NIR LIF parameters are recorded but never
  applied. Only the Affine node is evaluated on the host. The v0.2 test is
  workload-shaped. Exact NIR semantic execution belongs in #43 / LIM-1461.
- Any claim that other LIF/SNN models are supported.

Representing either model exactly needs configurable per-neuron
decay/threshold, configurable (or absent) refractory behavior, analog-current
input scaling, and public wrappers. That is outside the fixed v0.2.0 contract.
It is already scoped in #43 / LIM-1461, with wrapper work referenced there
via LIM-1304.

### Deferred fixtures (not exercised)

| Fixture | Reason |
|---|---|
| Synfire `pabogdan/ifsynfire:0.1.0` | Optional in #44. Graph and semantics not inspected for v0.2.0 |
| HF `Krishnav1234/neurocuda-mlp-mnist-snn` | Stretch in #44. Not a v0.2.0 release blocker |

## Comparison policy

- Spikes and refractory counters: exact, every tick, every neuron.
- Membrane: **bit-exact** (`f32::to_bits`) every tick. Justification, checked
  against the sm_120 PTX built with the repository's flags (`--use_fast_math`):
  - The `lif_step` update compiles to a single `fma.rn.ftz.f32` with the
    constant `0f3F59999A`, identical to host `0.85f32` (`0x3f59999a`).
  - `lif_step_weighted` uses `fma.rn.ftz.f32` for the unrolled dot product and
    for the update.
  - The threshold test is `setp.ltu.ftz.f32 v, 1.0`.
  - The host oracle uses `f32::mul_add` (single rounding, round-to-nearest)
    in the same order.

  The only remaining divergence is flush-to-zero, so every trace asserts that
  no oracle membrane or current is subnormal. Q8.8 × {0,1} partial sums are
  multiples of 1/256, and 128 ticks of 0.85 decay cannot reach the subnormal
  range. A one-ulp change to the oracle decay fails at timestep 1.
- Coverage guards: each GPU run requires at least one spike and at least one
  refractory tick. The Spikenaut run also requires non-zero sub-threshold
  and negative membrane states. Totals must agree with the oracle.
- Mismatch messages include fixture id, compatibility label, kernel,
  timestep, neuron index, expected/actual value (and bits), dimensions,
  stimulus identity, seed, and block size.

## Files

| Path | Role |
|---|---|
| `src/oracle.rs` | `LIF_*` constants, `lif_step_oracle`, `lif_step_weighted_oracle` |
| `tests/oracle.rs` | Hand-computed LIF traces (CPU) |
| `tests/fixtures/snn/` | Pinned fixtures, `fixture.json` provenance, attribution |
| `tests/snn_support/mod.rs` | Offline loaders (`include_str!`), stimulus generation, per-tick comparator |
| `tests/snn_fixtures.rs` | CPU integrity + oracle coverage (ordinary `cargo test`) |
| `tests/snn_fixtures_gpu.rs` | CUDA differential tests (`--features cuda -- --ignored`) |
| `scripts/snn_fixtures/verify_fixtures.py` | Offline re-check; `--network` re-downloads and audits pinned upstreams (manual only) |

Ordinary `cargo test` never touches the network.
