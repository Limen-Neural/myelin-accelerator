// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared, offline loaders for the pinned SNN fixtures (LIM-1462 / #44).
//!
//! Fixtures are embedded with `include_str!`: no runtime paths, no network.
//! Provenance and checksums live in each `fixture.json`; see
//! `tests/fixtures/snn/README.md` and `docs/SNN_COMPATIBILITY.md`.
//!
//! Both fixtures are **workload-only**: they supply real model shapes, weights
//! and parameters, while the oracle and the GPU execute the fixed myelin v0.2.0
//! LIF dynamics. Nothing here executes Spikenaut or NIR dynamics.

use myelin_accelerator::oracle::{CaseRng, lif_step_oracle, lif_step_weighted_oracle};
use serde::Deserialize;

const SPIKENAUT_JSON: &str = include_str!("../fixtures/snn/spikenaut/fixture.json");
const SPIKENAUT_MEM: &str = include_str!("../fixtures/snn/spikenaut/parameters_weights.mem");
const SYNFIRE_JSON: &str = include_str!("../fixtures/snn/synfire_lifneuron/fixture.json");

/// The only classification either fixture may carry in v0.2.0.
pub const WORKLOAD_ONLY: &str = "workload-only";

// ── Spikenaut ───────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SpikenautCard {
    fixture_id: String,
    compatibility: String,
    kernel: String,
    source: SpikenautSource,
    weights: SpikenautWeights,
    stimulus: SpikenautStimulus,
}

#[derive(Deserialize)]
struct SpikenautSource {
    commit: String,
    sha256: String,
}

#[derive(Deserialize)]
struct SpikenautWeights {
    n_neurons: usize,
    n_inputs: usize,
    word_count: usize,
    raw_code_sum: i64,
    nonzero_count: usize,
    min_code: i32,
    max_code: i32,
}

#[derive(Deserialize)]
struct SpikenautStimulus {
    identity: String,
    seed: u64,
    rates: Vec<f32>,
    steps: usize,
}

/// Decoded Spikenaut-shaped weighted-LIF workload.
pub struct SpikenautFixture {
    pub fixture_id: String,
    pub compatibility: String,
    pub kernel: String,
    pub commit: String,
    pub sha256: String,
    pub n_neurons: usize,
    pub n_inputs: usize,
    /// Raw signed Q8.8 codes in file (neuron-major) order.
    pub codes: Vec<i32>,
    /// `codes / 256` as f32, row-major `[n_neurons × n_inputs]`.
    pub weights: Vec<f32>,
    pub expected_word_count: usize,
    pub expected_code_sum: i64,
    pub expected_nonzero: usize,
    pub expected_min_code: i32,
    pub expected_max_code: i32,
    pub stimulus_id: String,
    pub seed: u64,
    pub rates: Vec<f32>,
    pub steps: usize,
}

/// Decode one signed two's-complement Q8.8 hex word (4 hex digits).
pub fn decode_q88_word(word: &str) -> i32 {
    assert_eq!(
        word.len(),
        4,
        "Q8.8 word must be 4 hex digits, got {word:?}"
    );
    let raw =
        u16::from_str_radix(word, 16).unwrap_or_else(|e| panic!("bad Q8.8 word {word:?}: {e}"));
    i32::from(raw as i16)
}

/// Load the pinned Spikenaut fixture. The `.mem` is neuron-major, which is
/// already the `lif_step_weighted` row-major layout, so no transpose occurs.
pub fn spikenaut() -> SpikenautFixture {
    let card: SpikenautCard = serde_json::from_str(SPIKENAUT_JSON).expect("spikenaut fixture.json");
    let codes: Vec<i32> = SPIKENAUT_MEM.lines().map(decode_q88_word).collect();
    // Q8.8 codes are |code| ≤ 2^15, so code / 256 is exact in f32.
    let weights = codes.iter().map(|&c| c as f32 / 256.0).collect();
    SpikenautFixture {
        fixture_id: card.fixture_id,
        compatibility: card.compatibility,
        kernel: card.kernel,
        commit: card.source.commit,
        sha256: card.source.sha256,
        n_neurons: card.weights.n_neurons,
        n_inputs: card.weights.n_inputs,
        codes,
        weights,
        expected_word_count: card.weights.word_count,
        expected_code_sum: card.weights.raw_code_sum,
        expected_nonzero: card.weights.nonzero_count,
        expected_min_code: card.weights.min_code,
        expected_max_code: card.weights.max_code,
        stimulus_id: card.stimulus.identity,
        seed: card.stimulus.seed,
        rates: card.stimulus.rates,
        steps: card.stimulus.steps,
    }
}

impl SpikenautFixture {
    /// Deterministic 0/1 input spikes, `steps × n_inputs`. One `CaseRng`
    /// stream seeded with `seed`; per tick, inputs are drawn in index order.
    pub fn inputs(&self) -> Vec<Vec<f32>> {
        assert_eq!(self.rates.len(), self.n_inputs, "rate table width");
        let mut rng = CaseRng::new(self.seed);
        (0..self.steps)
            .map(|_| {
                self.rates
                    .iter()
                    .map(|&rate| f32::from(u8::from(rng.next_rate_f32() < rate)))
                    .collect()
            })
            .collect()
    }

    pub fn shape(&self) -> String {
        format!("n_neurons={} n_inputs={}", self.n_neurons, self.n_inputs)
    }
}

// ── Synfire / NIR lifneuron ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct SynfireCard {
    fixture_id: String,
    compatibility: String,
    kernel: String,
    source: SynfireSource,
    graph: SynfireGraph,
    stimuli: Vec<SynfireStimulus>,
}

#[derive(Deserialize)]
struct SynfireSource {
    files: Vec<SynfireFile>,
}

#[derive(Deserialize)]
struct SynfireFile {
    name: String,
    sha256: String,
}

#[derive(Deserialize)]
struct SynfireGraph {
    topology: String,
    affine_weight: f32,
    affine_weight_f32_bits: String,
    affine_weight_shape: Vec<usize>,
    affine_bias: f32,
    affine_bias_f32_bits: String,
    lif_tau: f32,
    lif_tau_f32_bits: String,
    lif_r: f32,
    lif_v_leak: f32,
    lif_v_threshold: f32,
    lif_v_threshold_f32_bits: String,
    lif_v_reset: Option<f32>,
}

#[derive(Deserialize)]
struct SynfireStimulus {
    identity: String,
    steps: usize,
    #[serde(default)]
    d0: Option<Vec<u8>>,
}

/// One deterministic stimulus for the 1×1 workload.
pub struct SynfireStimulusCase {
    pub identity: String,
    /// `x[t]` fed into the Affine node.
    pub x: Vec<f32>,
}

/// Derived scalars of `pabogdan/lifneuron:1.0.0` plus its stimuli.
pub struct SynfireFixture {
    pub fixture_id: String,
    pub compatibility: String,
    pub kernel: String,
    pub model_nir_sha256: String,
    pub topology: String,
    pub affine_weight: f32,
    pub affine_weight_bits: u32,
    pub affine_weight_shape: Vec<usize>,
    pub affine_bias: f32,
    pub affine_bias_bits: u32,
    /// Recorded for the compatibility matrix; **not executed** in v0.2.0.
    pub nir_tau: f32,
    pub nir_tau_bits: u32,
    pub nir_r: f32,
    pub nir_v_leak: f32,
    pub nir_v_threshold: f32,
    pub nir_v_threshold_bits: u32,
    pub nir_v_reset: Option<f32>,
    pub stimuli: Vec<SynfireStimulusCase>,
}

fn parse_bits(hex: &str) -> u32 {
    let digits = hex.strip_prefix("0x").expect("f32 bits must start with 0x");
    u32::from_str_radix(digits, 16).unwrap_or_else(|e| panic!("bad f32 bits {hex:?}: {e}"))
}

pub fn synfire() -> SynfireFixture {
    let card: SynfireCard = serde_json::from_str(SYNFIRE_JSON).expect("synfire fixture.json");
    let model_nir_sha256 = card
        .source
        .files
        .iter()
        .find(|f| f.name == "model.nir")
        .map(|f| f.sha256.clone())
        .expect("model.nir checksum recorded");
    let stimuli = card
        .stimuli
        .into_iter()
        .map(|s| {
            let x: Vec<f32> = match (s.identity.as_str(), s.d0) {
                ("nir-paper-d0", Some(d0)) => d0.into_iter().map(f32::from).collect(),
                // x[t] = (t + 1) / 256: exact dyadic values.
                ("myelin-ramp-256", None) => (0..s.steps).map(|t| (t + 1) as f32 / 256.0).collect(),
                (other, _) => panic!("unknown synfire stimulus {other:?}"),
            };
            assert_eq!(x.len(), s.steps, "stimulus {} length", s.identity);
            SynfireStimulusCase {
                identity: s.identity,
                x,
            }
        })
        .collect();
    let g = card.graph;
    SynfireFixture {
        fixture_id: card.fixture_id,
        compatibility: card.compatibility,
        kernel: card.kernel,
        model_nir_sha256,
        topology: g.topology,
        affine_weight: g.affine_weight,
        affine_weight_bits: parse_bits(&g.affine_weight_f32_bits),
        affine_weight_shape: g.affine_weight_shape,
        affine_bias: g.affine_bias,
        affine_bias_bits: parse_bits(&g.affine_bias_f32_bits),
        nir_tau: g.lif_tau,
        nir_tau_bits: parse_bits(&g.lif_tau_f32_bits),
        nir_r: g.lif_r,
        nir_v_leak: g.lif_v_leak,
        nir_v_threshold: g.lif_v_threshold,
        nir_v_threshold_bits: parse_bits(&g.lif_v_threshold_f32_bits),
        nir_v_reset: g.lif_v_reset,
        stimuli,
    }
}

impl SynfireFixture {
    /// Host-side Affine node only: `I_ext[t] = w · x[t] + b` (one fused op).
    /// The NIR LIF parameters are intentionally not applied.
    pub fn currents(&self, stim: &SynfireStimulusCase) -> Vec<f32> {
        stim.x
            .iter()
            .map(|&x| self.affine_weight.mul_add(x, self.affine_bias))
            .collect()
    }
}

// ── Oracle traces and per-timestep comparison ───────────────────────────────

/// State after one tick.
#[derive(Clone)]
pub struct StepState {
    pub membrane: Vec<f32>,
    pub refract: Vec<u32>,
    pub spikes: Vec<u32>,
}

/// Coverage counters so a trivial trace cannot pass silently.
#[derive(Default, Debug, Clone, Copy)]
pub struct Coverage {
    pub spikes: usize,
    /// Neuron-ticks spent in the refractory branch.
    pub refractory_ticks: usize,
    /// Refractory neuron-ticks whose input current was nonzero (ignored input).
    pub ignored_input_ticks: usize,
    /// Neuron-ticks ending with a nonzero, non-spiking membrane.
    pub subthreshold_nonzero: usize,
    /// Neuron-ticks ending with a negative membrane.
    pub negative_membrane: usize,
}

impl Coverage {
    fn record(&mut self, prev_refract: &[u32], currents: &[f32], state: &StepState) {
        for i in 0..state.membrane.len() {
            self.spikes += state.spikes[i] as usize;
            if prev_refract[i] > 0 {
                self.refractory_ticks += 1;
                if currents[i] != 0.0 {
                    self.ignored_input_ticks += 1;
                }
            }
            if state.spikes[i] == 0 && state.membrane[i] != 0.0 {
                self.subthreshold_nonzero += 1;
            }
            if state.membrane[i] < 0.0 {
                self.negative_membrane += 1;
            }
        }
    }
}

/// Stateful fixed-v0.2.0 reference for `lif_step_weighted`.
pub struct WeightedOracle<'a> {
    weights: &'a [f32],
    n_inputs: usize,
    membrane: Vec<f32>,
    refract: Vec<u32>,
    pub coverage: Coverage,
}

impl<'a> WeightedOracle<'a> {
    pub fn new(weights: &'a [f32], n_neurons: usize, n_inputs: usize) -> Self {
        Self {
            weights,
            n_inputs,
            membrane: vec![0.0; n_neurons],
            refract: vec![0; n_neurons],
            coverage: Coverage::default(),
        }
    }

    pub fn step(&mut self, input: &[f32]) -> StepState {
        let prev = self.refract.clone();
        // Per-neuron current, only for the "ignored input" counter.
        let currents: Vec<f32> = self
            .weights
            .chunks(self.n_inputs)
            .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
            .collect();
        let spikes = lif_step_weighted_oracle(
            &mut self.membrane,
            self.weights,
            input,
            &mut self.refract,
            self.n_inputs,
        );
        let state = StepState {
            membrane: self.membrane.clone(),
            refract: self.refract.clone(),
            spikes,
        };
        self.coverage.record(&prev, &currents, &state);
        state
    }
}

/// Stateful fixed-v0.2.0 reference for `lif_step`.
pub struct UnweightedOracle {
    membrane: Vec<f32>,
    refract: Vec<u32>,
    pub coverage: Coverage,
}

impl UnweightedOracle {
    pub fn new(n_neurons: usize) -> Self {
        Self {
            membrane: vec![0.0; n_neurons],
            refract: vec![0; n_neurons],
            coverage: Coverage::default(),
        }
    }

    pub fn step(&mut self, i_ext: &[f32]) -> StepState {
        let prev = self.refract.clone();
        let spikes = lif_step_oracle(&mut self.membrane, i_ext, &mut self.refract);
        let state = StepState {
            membrane: self.membrane.clone(),
            refract: self.refract.clone(),
            spikes,
        };
        self.coverage.record(&prev, i_ext, &state);
        state
    }
}

/// Identity printed with every mismatch.
pub struct TraceContext<'a> {
    pub fixture_id: &'a str,
    pub compatibility: &'a str,
    pub kernel: &'a str,
    pub shape: String,
    pub stimulus: &'a str,
    pub seed: Option<u64>,
    /// CUDA block size, `None` for CPU-only runs.
    pub block: Option<u32>,
}

impl TraceContext<'_> {
    pub fn describe(&self) -> String {
        let seed = self
            .seed
            .map_or_else(|| "n/a".to_string(), |s| s.to_string());
        let block = self
            .block
            .map_or_else(|| "cpu".to_string(), |b| b.to_string());
        format!(
            "fixture={} label={} kernel={} {} stimulus={} seed={} block={}",
            self.fixture_id,
            self.compatibility,
            self.kernel,
            self.shape,
            self.stimulus,
            seed,
            block
        )
    }
}

/// Compare one tick. Spikes and refractory counters are exact; membrane is
/// compared **bit-for-bit** (see `docs/SNN_COMPATIBILITY.md`, comparison
/// policy). Returns the first mismatch with full replay context.
pub fn check_step(
    ctx: &TraceContext<'_>,
    t: usize,
    got: &StepState,
    expected: &StepState,
) -> Result<(), String> {
    let n = expected.membrane.len();
    for (field, glen) in [
        ("membrane", got.membrane.len()),
        ("refract", got.refract.len()),
        ("spikes", got.spikes.len()),
    ] {
        if glen != n {
            return Err(format!(
                "length mismatch in {field}: expected {n} got {glen} at timestep={t} ({})",
                ctx.describe()
            ));
        }
    }
    for i in 0..n {
        if got.spikes[i] != expected.spikes[i] {
            return Err(format!(
                "spike mismatch at timestep={t} neuron={i}: expected {} got {} ({})",
                expected.spikes[i],
                got.spikes[i],
                ctx.describe()
            ));
        }
        if got.refract[i] != expected.refract[i] {
            return Err(format!(
                "refract mismatch at timestep={t} neuron={i}: expected {} got {} ({})",
                expected.refract[i],
                got.refract[i],
                ctx.describe()
            ));
        }
        let (e, g) = (expected.membrane[i], got.membrane[i]);
        if e.to_bits() != g.to_bits() {
            return Err(format!(
                "membrane mismatch at timestep={t} neuron={i}: expected {e:e} (0x{:08x}) got {g:e} (0x{:08x}) \
                 (bit-exact policy; {})",
                e.to_bits(),
                g.to_bits(),
                ctx.describe()
            ));
        }
    }
    Ok(())
}

/// The device is built with `--use_fast_math` (flush-to-zero). The bit-exact
/// membrane policy is only sound while no subnormal enters or leaves the LIF
/// FMA, so every fixture trace asserts this on the oracle side.
pub fn assert_no_subnormals(ctx: &TraceContext<'_>, t: usize, values: &[f32], what: &str) {
    for (i, v) in values.iter().enumerate() {
        assert!(
            !v.is_subnormal(),
            "{what} is subnormal at timestep={t} index={i}: {v:e} — bit-exact policy \
             invalid under device flush-to-zero ({})",
            ctx.describe()
        );
    }
}
