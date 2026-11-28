# SNN external-model interoperability (v0.3.0)

Tracking: GH #43 / LIM-1461. This page is the durable numerical contract for
the exact external-model executions in `tests/external_semantics_gpu.rs`.
It complements, rather than replaces, the workload-only v0.2.0 evidence in
`SNN_COMPATIBILITY.md`.

## Pre-declared implementation plan and comparison rules

The implementation uses only supported public `GpuAccelerator` methods. Raw
`KernelModule::get_function` launches are diagnostic coverage and do not count
toward this interoperability result.

### Synfire / NIR single LIF

- Model: Synfire `pabogdan/lifneuron:1.0.0`, release id
  `01a01fa8-902e-4517-b47a-bc5fa6b55d2b`.
- Independent upstream artifact: NIR `paper/01_lif/lif_norse.nir` at
  `neuromorphs/NIR` commit `da0551dad03c4818689fdae7a48bfcff0be07b81`;
  `model.nir` SHA-256 `0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6`.
- Independent output oracle: NIR `paper/01_lif/lif_norse.csv` at the same
  commit, checked in as `tests/fixtures/snn/synfire_lifneuron/lif_norse.csv`
  with SHA-256 `300d80dc21d67137740722caefb6d9a2f267331bfe682051ed99ac41a9cd0448`.
- Graph/input: `Input(1) -> Affine(1) -> LIF(1) -> Output(1)`, affine weight
  1.0 and bias 0.0. The input is the paper's binary `d0` after its documented
  10x zero-upsampling, 1000 ticks at `dt = 1e-4` s.
- Neuron semantics: Norse `LIFBoxCell`, `tau_mem_inv = 400`, `dt = 1e-4`,
  threshold 0.1, zero reset, no refractory period. In the discrete public
  myelin recurrence this is decay `1 - dt/tau = 0.96` and input gain
  `dt/tau = 0.04` after the affine projection.
- Public execution: `lif_step_with_dynamics` with validated
  `TemporalDynamics { lif_decay: 0.96, lif_threshold: 0.1, lif_reset: 0.0,
  lif_refractory_ticks: 0, ..Default::default() }`.
- Rule declared before evaluation: spike decisions must match exactly at all
  1000 upsampled ticks; membrane voltage may differ by at most `1e-6`
  absolute. The tolerance covers decimal round-trip in the CSV and different
  but algebraically equivalent f32 operation ordering between Norse and the
  myelin FMA recurrence. It is not a model-quality tolerance and was not
  relaxed after the run.

### Spikenaut merged_v2 bank

- Model: `rmems/Spikenaut-SNN` `dataset/merged_v2`, hidden-weight commit
  `6965e12a6e46d29783e69f6b0fda3dda1058bd29`; Hugging Face revision
  `da93893d53dc2c28b58d0b48db7bd78a39ba8f8d`.
- Independent reference implementation: the standard-library Python keep-LIF
  stepper in `tools/hamming_lif.py` at the same commit. Its documented
  `learn=false` recurrence is analog current `W @ stim`, optional inhibitory
  drive, `v = decay * v + input`, spike at `v >= threshold`, then zero reset.
- Parameter artifacts are the signed Q8.8 `.mem` files, decoded exactly as
  two's-complement `code / 256`:
  - hidden weights `parameters_weights.mem`, SHA-256 `825969873444d215d09920e69592700a2cc8594d92e7b14c4059eef8919fe4f3`;
  - thresholds `parameters.mem`, SHA-256 `2aff910820757f7015f7097dae0aa00f631fc49d2a5df912b35ba92e309f7a70`;
  - decays `parameters_decay.mem`, SHA-256 `bbb575d31ebd5e0386037c3421d17212b3645522f4e64d02272bb5d6abc0610d`;
  - readout weights `parameters_output_weights.mem`, SHA-256 `6577e1c5958cb04d0e064a23a8356f54d73739378ed7fdd925a367627014cf72`.
- Input: deterministic analog current on live axons 0-4 and zero on unused
  axons 5-15. It is not Poisson encoding and does not use the unrelated
  telemetry channel map. The pinned exp-024 inhibitory drive `0.05` is
  represented as a constant seventeenth input channel whose weight is zero for
  excitatory rows and `0.05` for inhibitory rows 12-15. This is an exact
  linear-algebra embedding of the reference `input[12:16] += 0.05` term.
- Hidden public execution: `lif_step_weighted_per_neuron` with the decoded
  per-neuron decay and threshold vectors, zero reset, and zero refractory
  ticks. Input values are continuous analog currents.
- Readout: the pinned 16x3 neuron-major output weights are transposed to the
  row-major `[3 x 16]` form required by the same public weighted-LIF primitive.
  Setting decay to exactly 1.0, reset to 0.0, refractory to 0, and threshold
  to `f32::MAX` makes that primitive a linear spike-weight accumulator,
  `output += W_out @ hidden_spikes`. The upstream artifact does not assign a
  class/decision meaning to rows 0-2; this test therefore compares numerical
  accumulated output values only and makes no classifier or quality claim.
- Rule declared before evaluation: hidden spike vectors must match exactly at
  every tick. Hidden membrane values may differ by at most `2e-5` absolute
  because the Python reference rounds each multiply and addition separately
  while the CUDA kernel uses fused multiply-add in the dot product and state
  update. Final readout values may differ by at most `1e-5` absolute for the
  same reason. Tolerances were not relaxed after observing the result.

## Hardware evidence recorded for this work

On the local RTX 5080 / `sm_120` host with CUDA 13.3, the command

```bash
cargo test --locked --features cuda --test external_semantics_gpu -- \
  --ignored --nocapture --test-threads=1
```

passed both ignored GPU tests. CPU-only `cargo test` does not claim device
execution; the ignored test output is the hardware evidence.

## Compatibility matrix

| | Synfire/NIR lifneuron | Spikenaut merged_v2 |
|---|---|---|
| Topology | 1 input -> 1x1 affine -> 1 LIF -> output | 16 analog inputs -> 16 LIF -> 16x3 linear readout accumulator |
| Input | Binary `d0`, 10x zero-upsampled, affine identity, `dt/tau` current gain | Deterministic analog currents on axons 0-4; axons 5-15 zero; constant inhibitory drive embedded as channel 16 |
| Timestep | `dt = 1e-4` s, 1000 ticks | 1 kHz discrete ticks, 96 replay ticks in the qualification test |
| Neuron rule | Euler LIF equivalent: `v = 0.96*v + 0.04*I`; threshold 0.1; reset 0; no refractory | `v = decay_n*v + W_n @ stim`; per-neuron Q8.8 decay/threshold; reset 0; no refractory |
| Public operation | `lif_step_with_dynamics` | `lif_step_weighted_per_neuron` for hidden layer and readout accumulator |
| Spike rule | exact equality, all ticks | exact equality, all ticks |
| State/output rule | membrane abs tolerance `1e-6` | membrane abs `2e-5`; final readout abs `1e-5` |
| Oracle | upstream `lif_norse.csv` at pinned NIR commit | direct Rust port of upstream `tools/hamming_lif.py` recurrence at pinned Spikenaut commit |

## Unsupported / non-claims

- No SAAQ semantics or telemetry policy are part of myelin.
- No Hugging Face downloader, broad importer, training loop, dataset
  orchestration, convolution support, or full NIR runtime is included.
- Spikenaut K-WTA is train-time sidecar metadata and is not part of the
  `learn=false` health-eval recurrence used here.
- Spikenaut's upstream repository explicitly does not define a decision
  contract for readout rows 0-2. The qualification compares numerical readout
  accumulations only; it does not claim a classifier decision or accuracy.
- Passing these two pinned fixtures does not imply support for all LIF/SNN
  models.
