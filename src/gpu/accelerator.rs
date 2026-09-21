// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::bitpacking::TERNARY_VALUES_PER_WORD;
#[cfg(feature = "saaq")]
use crate::gif::{GIF_ADAPTATION_SCALE, GIF_BLOCK_SIZE, SnapshotChannels, gif_saaq_grid};
use crate::gpu::context::GpuContext;
use crate::gpu::error::{GpuError, GpuResult};
#[cfg(feature = "saaq")]
use crate::gpu::ffi;
use crate::gpu::kernel::KernelModule;
use crate::gpu::memory::GpuBuffer;
use crate::launch_hook::{LaunchFailure, LaunchType, report_launch_failure};
use cust::launch;
use cust::stream::{Stream, StreamFlags};
use nvtx::{range_pop, range_push};
use std::cell::RefCell;
use tracing::warn;

const SATSOLVER_BLOCK_SIZE: u32 = 256;
const SATSOLVER_SHARED_MEM_BYTES: u32 = 0;
#[cfg(feature = "saaq")]
const TEMPORAL_BLOCK_SIZE: u32 = GIF_BLOCK_SIZE;
#[cfg(feature = "saaq")]
const TEMPORAL_SHARED_MEM_BYTES: u32 = 0;
#[cfg(feature = "saaq")]
const SNAPSHOT_CHANNELS: usize = 4;

#[cfg(feature = "saaq")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SynapsePrecision {
    None,
    F32,
    F16,
}

#[cfg(feature = "saaq")]
struct TemporalState {
    neuron_count: usize,
    n_inputs: usize,
    membrane: GpuBuffer<f32>,
    refractory: GpuBuffer<u32>,
    spikes_out: GpuBuffer<u32>,
    input_current: GpuBuffer<f32>,
    input_spikes: GpuBuffer<f32>,
    adaptation: GpuBuffer<f32>,
    weights_f32: GpuBuffer<f32>,
    weights_f16: Option<GpuBuffer<u16>>,
    synapse_precision: SynapsePrecision,
    synapse_signature: Option<String>,
    snapshot: GpuBuffer<f32>,
    best_walker: GpuBuffer<u32>,
    saaq_partial_scores: GpuBuffer<f32>,
    saaq_partial_walkers: GpuBuffer<u32>,
}

pub struct GpuAccelerator {
    _ctx: Option<GpuContext>,
    modules: Option<KernelModule>,
    stream: Option<Stream>,
    #[cfg(feature = "saaq")]
    temporal_state: Option<TemporalState>,
    aux_partial_scores: RefCell<Option<GpuBuffer<i32>>>,
    aux_partial_walkers: RefCell<Option<GpuBuffer<i32>>>,
}

impl GpuAccelerator {
    pub fn new() -> Self {
        match GpuContext::init() {
            Ok(ctx) => {
                let modules = match KernelModule::load() {
                    Ok(modules) => Some(modules),
                    Err(e) => {
                        #[cfg(feature = "saaq")]
                        warn!("[GPU] fatbin/PTX load failed (shim-only if stream is up): {e}");
                        #[cfg(not(feature = "saaq"))]
                        warn!("[GPU] fatbin/PTX load failed (CPU fallback): {e}");
                        None
                    }
                };
                let stream = match Stream::new(StreamFlags::DEFAULT, None) {
                    Ok(stream) => Some(stream),
                    Err(e) => {
                        warn!("[GPU] stream creation failed (CPU fallback): {e:?}");
                        None
                    }
                };
                Self {
                    _ctx: Some(ctx),
                    modules,
                    stream,
                    #[cfg(feature = "saaq")]
                    temporal_state: None,
                    aux_partial_scores: RefCell::new(None),
                    aux_partial_walkers: RefCell::new(None),
                }
            }
            Err(e) => {
                warn!("[GPU] No CUDA device (CPU fallback): {e}");
                Self {
                    _ctx: None,
                    modules: None,
                    stream: None,
                    #[cfg(feature = "saaq")]
                    temporal_state: None,
                    aux_partial_scores: RefCell::new(None),
                    aux_partial_walkers: RefCell::new(None),
                }
            }
        }
    }

    /// `true` when a CUDA context and stream exist.
    ///
    /// Fatbin/PTX helpers still require [`Self::kernels`] / [`Self::kernels_ready`].
    /// With `--features saaq`, the C-ABI shim path (F16 GIF + SAAQ) can run
    /// with context+stream alone if module load failed.
    pub fn is_ready(&self) -> bool {
        self._ctx.is_some() && self.stream.is_some()
    }

    /// `true` when [`Self::is_ready`] and fatbin/PTX modules loaded successfully.
    pub fn kernels_ready(&self) -> bool {
        self.is_ready() && self.modules.is_some()
    }

    #[cfg(feature = "saaq")]
    fn has_context(&self) -> bool {
        self._ctx.is_some()
    }

    fn ptx_launch_error(
        kernel_name: &str,
        grid: u32,
        block: u32,
        shared_mem: u32,
        neuron_count: Option<usize>,
        error: impl std::fmt::Debug,
    ) -> GpuError {
        Self::reported_launch_error(
            kernel_name,
            LaunchType::PtxFatbin,
            grid,
            block,
            shared_mem,
            neuron_count,
            error,
        )
    }

    fn reported_launch_error(
        kernel_name: &str,
        launch_type: LaunchType,
        grid: u32,
        block: u32,
        shared_mem: u32,
        neuron_count: Option<usize>,
        error: impl std::fmt::Debug,
    ) -> GpuError {
        let gpu_error = GpuError::LaunchFailed(format!("{kernel_name} launch: {error:?}"));
        report_launch_failure(LaunchFailure {
            kernel_name: kernel_name.to_string(),
            launch_type,
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_mem,
            neuron_count,
            error: gpu_error.to_string(),
            jit_error_log: None,
            jit_info_log: None,
        });
        gpu_error
    }

    #[cfg(feature = "saaq")]
    fn temporal_grid(neuron_count: usize) -> GpuResult<u32> {
        gif_saaq_grid(neuron_count).map_err(GpuError::LaunchFailed)
    }

    pub fn kernels(&self) -> GpuResult<&KernelModule> {
        if self._ctx.is_none() {
            return Err(GpuError::NoGpu);
        }
        self.modules.as_ref().ok_or_else(|| {
            #[cfg(feature = "saaq")]
            {
                GpuError::ModuleLoadFailed(
                    "fatbin/PTX modules are not loaded; C-ABI shim launches may still work when is_ready()"
                        .into(),
                )
            }
            #[cfg(not(feature = "saaq"))]
            {
                GpuError::ModuleLoadFailed("fatbin/PTX modules are not loaded".into())
            }
        })
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        stream
            .synchronize()
            .map_err(|e| GpuError::LaunchFailed(format!("stream sync: {e:?}")))
    }

    /// Allocate (or reuse) resident GIF/SAAQ device buffers for `neuron_count`.
    ///
    /// Uses full connectivity (`n_inputs == neuron_count`). Reallocates when
    /// the count changes. Production corinth-canal size is 2048.
    #[cfg(feature = "saaq")]
    pub fn ensure_temporal_state(&mut self, neuron_count: usize) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        if neuron_count == 0 {
            return Err(GpuError::LaunchFailed(
                "temporal state requires neuron_count > 0".into(),
            ));
        }
        let _ = Self::temporal_grid(neuron_count)?;

        let needs_realloc = self
            .temporal_state
            .as_ref()
            .is_none_or(|state| state.neuron_count != neuron_count);

        if needs_realloc {
            self.temporal_state = Some(Self::build_temporal_state(neuron_count)?);
        }

        Ok(())
    }

    /// Project a 4-channel snapshot into per-neuron `input_current`.
    ///
    /// The next [`Self::gif_step_weighted_tick`] (f32 or f16) adds that current
    /// to the synaptic drive so telemetry projection changes GIF dynamics.
    #[cfg(feature = "saaq")]
    pub fn project_snapshot_current(
        &mut self,
        snapshot: SnapshotChannels,
        neuron_count: usize,
    ) -> GpuResult<()> {
        self.ensure_temporal_state(neuron_count)?;

        let modules = self.kernels()?;
        let project_snapshot_current = modules.get_function("project_snapshot_current")?;
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;

        state.snapshot.upload(&snapshot.as_array())?;

        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid = Self::temporal_grid(neuron_count)?;
        unsafe {
            launch!(project_snapshot_current<<<grid, TEMPORAL_BLOCK_SIZE, TEMPORAL_SHARED_MEM_BYTES, stream>>>(
                state.snapshot.as_device_ptr(),
                state.input_current.as_device_ptr(),
                neuron_count as i32
            ))
            .map_err(|e| Self::ptx_launch_error(
                "project_snapshot_current",
                grid,
                TEMPORAL_BLOCK_SIZE,
                TEMPORAL_SHARED_MEM_BYTES,
                Some(neuron_count),
                e,
            ))?;
        }

        self.synchronize()
    }

    /// Download the current GIF spike vector.
    ///
    /// `neuron_count` must equal the resident temporal size.
    #[cfg(feature = "saaq")]
    pub fn temporal_spikes_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<u32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::require_state_neuron_count(state, neuron_count)?;
        state.spikes_out.to_vec()
    }

    /// Download the current GIF membrane vector.
    ///
    /// `neuron_count` must equal the resident temporal size.
    #[cfg(feature = "saaq")]
    pub fn temporal_membrane_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<f32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::require_state_neuron_count(state, neuron_count)?;
        state.membrane.to_vec()
    }

    /// Download the current GIF adaptation vector.
    ///
    /// `neuron_count` must equal the resident temporal size.
    #[cfg(feature = "saaq")]
    pub fn temporal_adaptation_to_vec(&self, neuron_count: usize) -> GpuResult<Vec<f32>> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        Self::require_state_neuron_count(state, neuron_count)?;
        state.adaptation.to_vec()
    }

    /// Upload a per-neuron (or per-input) vector into the resident `input_spikes` buffer.
    #[cfg(feature = "saaq")]
    pub fn upload_temporal_input_spikes(&mut self, input_spikes: &[f32]) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        if input_spikes.len() != state.n_inputs {
            return Err(GpuError::MemoryError(format!(
                "temporal input_spikes length mismatch: expected {}, got {}",
                state.n_inputs,
                input_spikes.len()
            )));
        }
        state
            .input_spikes
            .upload(input_spikes)
            .map_err(|e| GpuError::MemoryError(format!("input_spikes upload failed: {e}")))
    }

    /// Load an f32 synapse matrix. Signature defaults to `"host-f32"`.
    #[cfg(feature = "saaq")]
    pub fn load_synapse_weights(&mut self, weights: &[f32]) -> GpuResult<()> {
        self.load_synapse_weights_named("host-f32", weights)
    }

    /// Load an f32 synapse matrix, skipping the upload when `signature` is already resident.
    #[cfg(feature = "saaq")]
    pub fn load_synapse_weights_named(
        &mut self,
        signature: &str,
        weights: &[f32],
    ) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        let expected = state.neuron_count * state.n_inputs;
        if weights.len() != expected {
            return Err(GpuError::MemoryError(format!(
                "weights length mismatch: expected {} ({}x{}), got {}",
                expected,
                state.neuron_count,
                state.n_inputs,
                weights.len()
            )));
        }
        if state.synapse_precision == SynapsePrecision::F32
            && state.synapse_signature.as_deref() == Some(signature)
        {
            return Ok(());
        }
        state
            .weights_f32
            .upload(weights)
            .map_err(|e| GpuError::MemoryError(format!("synapse weights upload failed: {e}")))?;
        state.synapse_precision = SynapsePrecision::F32;
        state.synapse_signature = Some(signature.to_owned());
        Ok(())
    }

    /// Load an IEEE f16 synapse matrix from host `u16` bits.
    #[cfg(feature = "saaq")]
    pub fn load_synapse_weights_f16_registered(
        &mut self,
        signature: &str,
        weights: &[u16],
    ) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        let expected = state.neuron_count * state.n_inputs;
        if weights.len() != expected {
            return Err(GpuError::MemoryError(format!(
                "f16 weights length mismatch: expected {} ({}x{}), got {}",
                expected,
                state.neuron_count,
                state.n_inputs,
                weights.len()
            )));
        }

        if state.synapse_precision == SynapsePrecision::F16
            && state.synapse_signature.as_deref() == Some(signature)
        {
            return Ok(());
        }

        let f16_weights = match state.weights_f16.as_mut() {
            Some(weights_f16) => weights_f16,
            _ => {
                state.weights_f16 = Some(GpuBuffer::<u16>::alloc(expected)?);
                state
                    .weights_f16
                    .as_mut()
                    .expect("weights_f16 was just inserted")
            }
        };
        f16_weights.upload(weights).map_err(|e| {
            GpuError::MemoryError(format!("registered f16 synapse upload failed: {e}"))
        })?;
        state.synapse_precision = SynapsePrecision::F16;
        state.synapse_signature = Some(signature.to_owned());
        Ok(())
    }

    /// One GIF-weighted tick, then on-device SAAQ selection.
    ///
    /// Returns the best-walker index. Requires synapse weights to be loaded.
    #[cfg(feature = "saaq")]
    pub fn gif_step_weighted_tick(&mut self, neuron_count: usize) -> GpuResult<u32> {
        self.ensure_temporal_state(neuron_count)?;

        let precision = self
            .temporal_state
            .as_ref()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?
            .synapse_precision;
        if precision == SynapsePrecision::None {
            return Err(GpuError::MemoryError(
                "synapse weights must be loaded before gif_step_weighted_tick".into(),
            ));
        }

        let n_inputs = self
            .temporal_state
            .as_ref()
            .expect("temporal state checked above")
            .n_inputs;
        let grid = Self::temporal_grid(neuron_count)?;
        let shared_bytes = (n_inputs * 4) as u32;

        {
            let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
            match precision {
                SynapsePrecision::F32 => {
                    let gif_step = self.kernels()?.get_function("gif_step_weighted")?;
                    let state = self
                        .temporal_state
                        .as_ref()
                        .expect("temporal state checked above");
                    unsafe {
                        launch!(gif_step<<<grid, TEMPORAL_BLOCK_SIZE, shared_bytes, stream>>>(
                            state.membrane.as_device_ptr(),
                            state.adaptation.as_device_ptr(),
                            state.weights_f32.as_device_ptr(),
                            state.input_spikes.as_device_ptr(),
                            state.input_current.as_device_ptr(),
                            state.refractory.as_device_ptr(),
                            state.spikes_out.as_device_ptr(),
                            neuron_count as i32,
                            n_inputs as i32
                        ))
                        .map_err(|e| {
                            Self::ptx_launch_error(
                                "gif_step_weighted",
                                grid,
                                TEMPORAL_BLOCK_SIZE,
                                shared_bytes,
                                Some(neuron_count),
                                e,
                            )
                        })?;
                    }
                }
                SynapsePrecision::F16 => {
                    let state = self
                        .temporal_state
                        .as_ref()
                        .expect("temporal state checked above");
                    let weights_f16 = state.weights_f16.as_ref().ok_or_else(|| {
                        GpuError::MemoryError("f16 synapse buffer not initialised".into())
                    })?;
                    ffi::launch_gif_step_weighted_f16(
                        stream,
                        grid,
                        TEMPORAL_BLOCK_SIZE,
                        shared_bytes,
                        state.membrane.as_device_ptr(),
                        state.adaptation.as_device_ptr(),
                        weights_f16.as_device_ptr(),
                        state.input_spikes.as_device_ptr(),
                        state.input_current.as_device_ptr(),
                        state.refractory.as_device_ptr(),
                        state.spikes_out.as_device_ptr(),
                        neuron_count as i32,
                        n_inputs as i32,
                    )?;
                }
                SynapsePrecision::None => unreachable!("validated above"),
            }
        }

        self.saaq_find_best_walker(neuron_count)
    }

    /// Zero resident GIF state (membrane, adaptation, spikes, inputs, SAAQ winner).
    #[cfg(feature = "saaq")]
    pub fn reset_temporal_state(&mut self) -> GpuResult<()> {
        if !self.has_context() {
            return Err(GpuError::NoGpu);
        }
        if self.temporal_state.is_none() {
            return Ok(());
        }

        let neuron_count = self
            .temporal_state
            .as_ref()
            .expect("checked above")
            .neuron_count;
        if self.modules.is_some() {
            let reset_membrane = self.kernels()?.get_function("reset_membrane")?;
            let grid = Self::temporal_grid(neuron_count)?;
            {
                let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
                let state = self.temporal_state.as_ref().expect("checked above");
                unsafe {
                    launch!(reset_membrane<<<grid, TEMPORAL_BLOCK_SIZE, TEMPORAL_SHARED_MEM_BYTES, stream>>>(
                        state.membrane.as_device_ptr(),
                        neuron_count as i32,
                        0.0f32
                    ))
                    .map_err(|e| Self::ptx_launch_error(
                        "reset_membrane",
                        grid,
                        TEMPORAL_BLOCK_SIZE,
                        TEMPORAL_SHARED_MEM_BYTES,
                        Some(neuron_count),
                        e,
                    ))?;
                }
            }
            self.synchronize()?;
        } else {
            let state = self.temporal_state.as_mut().expect("checked above");
            state.membrane.zero_prefix(neuron_count)?;
        }

        let state = self.temporal_state.as_mut().expect("checked above");
        state
            .refractory
            .upload(&vec![0u32; state.neuron_count])
            .map_err(|e| GpuError::MemoryError(format!("reset refractory upload failed: {e}")))?;
        state
            .spikes_out
            .upload(&vec![0u32; state.neuron_count])
            .map_err(|e| GpuError::MemoryError(format!("reset spikes upload failed: {e}")))?;
        state.input_current.zero_prefix(state.neuron_count)?;
        state.input_spikes.zero_prefix(state.n_inputs)?;
        state.adaptation.zero_prefix(state.neuron_count)?;
        state
            .best_walker
            .upload(&[0u32; 1])
            .map_err(|e| GpuError::MemoryError(format!("reset best_walker upload failed: {e}")))?;

        Ok(())
    }

    /// Currently loaded synapse signature, if any.
    #[cfg(feature = "saaq")]
    pub fn synapse_signature(&self) -> Option<&str> {
        self.temporal_state
            .as_ref()
            .and_then(|state| state.synapse_signature.as_deref())
    }

    /// On-device SAAQ two-pass reduction over resident membrane/adaptation.
    ///
    /// Public so LIM-955 can profile the unfused baseline independently of GIF.
    #[cfg(feature = "saaq")]
    pub fn saaq_find_best_walker(&mut self, neuron_count: usize) -> GpuResult<u32> {
        self.ensure_temporal_state(neuron_count)?;

        let state = self
            .temporal_state
            .as_mut()
            .ok_or_else(|| GpuError::MemoryError("temporal state not initialised".into()))?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid = Self::temporal_grid(neuron_count)?;

        ffi::launch_saaq_find_best_walker(
            stream,
            grid,
            TEMPORAL_BLOCK_SIZE,
            0,
            state.membrane.as_device_ptr(),
            state.adaptation.as_device_ptr(),
            state.saaq_partial_scores.as_device_ptr(),
            state.saaq_partial_walkers.as_device_ptr(),
            state.best_walker.as_device_ptr(),
            neuron_count as i32,
            GIF_ADAPTATION_SCALE,
        )?;

        stream.synchronize().map_err(|e| {
            Self::reported_launch_error(
                "saaq_find_best_walker",
                LaunchType::CAbiShim,
                grid,
                TEMPORAL_BLOCK_SIZE,
                0,
                Some(neuron_count),
                e,
            )
        })?;

        let best = state.best_walker.to_vec()?;
        Ok(best[0])
    }

    #[cfg(feature = "saaq")]
    fn build_temporal_state(neuron_count: usize) -> GpuResult<TemporalState> {
        let n_inputs = neuron_count;
        let weight_size = neuron_count * n_inputs;
        let saaq_partials_len = Self::temporal_grid(neuron_count)? as usize;
        Ok(TemporalState {
            neuron_count,
            n_inputs,
            membrane: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            refractory: GpuBuffer::<u32>::from_slice(&vec![0u32; neuron_count])?,
            spikes_out: GpuBuffer::<u32>::from_slice(&vec![0u32; neuron_count])?,
            input_current: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            input_spikes: GpuBuffer::<f32>::from_slice(&vec![0.0f32; n_inputs])?,
            adaptation: GpuBuffer::<f32>::from_slice(&vec![0.0f32; neuron_count])?,
            weights_f32: GpuBuffer::<f32>::from_slice(&vec![0.0f32; weight_size])?,
            weights_f16: None,
            synapse_precision: SynapsePrecision::None,
            synapse_signature: None,
            snapshot: GpuBuffer::<f32>::from_slice(&[0.0f32; SNAPSHOT_CHANNELS])?,
            best_walker: GpuBuffer::<u32>::from_slice(&[0u32; 1])?,
            saaq_partial_scores: GpuBuffer::<f32>::from_slice(&vec![0.0f32; saaq_partials_len])?,
            saaq_partial_walkers: GpuBuffer::<u32>::from_slice(&vec![0u32; saaq_partials_len])?,
        })
    }

    pub fn satsolver_extract(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        self.satsolver_extract_async(assignment, best_walker, output, n_vars, n_walkers)?;
        self.synchronize()
    }

    pub fn satsolver_extract_async(
        &self,
        assignment: &GpuBuffer<u8>,
        best_walker: &GpuBuffer<i32>,
        output: &mut GpuBuffer<u8>,
        n_vars: i32,
        n_walkers: i32,
    ) -> GpuResult<()> {
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_extract: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars == 0 {
            return Ok(());
        }

        let n_vars = n_vars as usize;
        let n_walkers_usize = n_walkers as usize;
        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars),
        )?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len("output", output.len(), n_vars)?;

        let kernels = self.kernels()?;
        let satsolver_extract = kernels.get_function("satsolver_extract")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid = Self::ceil_div_u32(n_vars as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;

        unsafe {
            launch!(satsolver_extract<<<grid, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                best_walker.as_device_ptr(),
                output.as_device_ptr(),
                n_vars as i32,
                n_walkers,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_extract launch: {e:?}")))?;
        }

        Ok(())
    }

    // 11 parameters mirror the satsolver_aux_update CUDA kernel ABI;
    // grouping them would require a context struct on every call site
    // without simplifying the launch. Allow the clippy lint.
    #[allow(clippy::too_many_arguments)]
    pub fn satsolver_aux_reduce_best(
        &self,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        best_score: &mut GpuBuffer<i32>,
        best_walker: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
    ) -> GpuResult<()> {
        self.satsolver_aux_reduce_best_async(
            assignment,
            sat_flags,
            scores,
            best_score,
            best_walker,
            clauses,
            n_walkers,
            n_vars,
            n_clauses,
            clause_len,
        )?;
        self.synchronize()
    }

    // 11 parameters mirror the satsolver_aux_update CUDA kernel ABI;
    // grouping them would require a context struct on every call site
    // without simplifying the launch. Allow the clippy lint.
    #[allow(clippy::too_many_arguments)]
    pub fn satsolver_aux_reduce_best_async(
        &self,
        assignment: &GpuBuffer<u8>,
        sat_flags: &mut GpuBuffer<u8>,
        scores: &GpuBuffer<i32>,
        best_score: &mut GpuBuffer<i32>,
        best_walker: &mut GpuBuffer<i32>,
        clauses: &GpuBuffer<i32>,
        n_walkers: i32,
        n_vars: i32,
        n_clauses: i32,
        clause_len: i32,
    ) -> GpuResult<()> {
        if n_walkers <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_walkers must be > 0, got {n_walkers}"
            )));
        }
        if n_vars < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_vars must be >= 0, got {n_vars}"
            )));
        }
        if n_clauses < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: n_clauses must be >= 0, got {n_clauses}"
            )));
        }
        if clause_len < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "satsolver_aux_reduce_best: clause_len must be >= 0, got {clause_len}"
            )));
        }

        let n_walkers_usize = n_walkers as usize;
        let n_vars_usize = n_vars as usize;
        let n_clauses_usize = n_clauses as usize;
        let clause_len_usize = clause_len as usize;

        Self::expect_len(
            "assignment",
            assignment.len(),
            n_walkers_usize.saturating_mul(n_vars_usize),
        )?;
        Self::expect_len(
            "sat_flags",
            sat_flags.len(),
            n_walkers_usize.saturating_mul(n_clauses_usize),
        )?;
        Self::expect_len("scores", scores.len(), n_walkers_usize)?;
        Self::expect_len("best_score", best_score.len(), 1)?;
        Self::expect_len("best_walker", best_walker.len(), 1)?;
        Self::expect_len(
            "clauses",
            clauses.len(),
            n_clauses_usize.saturating_mul(clause_len_usize),
        )?;

        let kernels = self.kernels()?;
        let satsolver_aux_update = kernels.get_function("satsolver_aux_update")?;
        let satsolver_best_reduce_pass2 = kernels.get_function("satsolver_best_reduce_pass2")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let grid_x = Self::ceil_div_u32(n_walkers as u32, SATSOLVER_BLOCK_SIZE);
        let block = SATSOLVER_BLOCK_SIZE;
        let partial_len = grid_x as usize;
        let mut partial_scores = self.aux_partial_scores.borrow_mut();
        let mut partial_walkers = self.aux_partial_walkers.borrow_mut();
        let need_partial_realloc = partial_scores
            .as_ref()
            .is_none_or(|b| b.len() < partial_len)
            || partial_walkers
                .as_ref()
                .is_none_or(|b| b.len() < partial_len);
        if need_partial_realloc {
            stream.synchronize().map_err(|e| {
                GpuError::LaunchFailed(format!("stream sync before partial realloc: {e:?}"))
            })?;
            let scores = GpuBuffer::<i32>::alloc(partial_len)?;
            let walkers = GpuBuffer::<i32>::alloc(partial_len)?;
            *partial_scores = Some(scores);
            *partial_walkers = Some(walkers);
        }
        let partial_scores = partial_scores.as_ref().expect("partial_scores buffer");
        let partial_walkers = partial_walkers.as_ref().expect("partial_walkers buffer");

        unsafe {
            launch!(satsolver_aux_update<<<grid_x, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                assignment.as_device_ptr(),
                sat_flags.as_device_ptr(),
                scores.as_device_ptr(),
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                clauses.as_device_ptr(),
                n_walkers,
                n_vars,
                n_clauses,
                clause_len,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("satsolver_aux_update launch: {e:?}")))?;

            launch!(satsolver_best_reduce_pass2<<<1u32, block, SATSOLVER_SHARED_MEM_BYTES, stream>>>(
                partial_scores.as_device_ptr(),
                partial_walkers.as_device_ptr(),
                best_score.as_device_ptr(),
                best_walker.as_device_ptr(),
                partial_len as i32,
            ))
            .map_err(|e| {
                GpuError::LaunchFailed(format!("satsolver_best_reduce_pass2 launch: {e:?}"))
            })?;
        }

        Ok(())
    }

    pub fn poisson_encode(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        self.poisson_encode_async(stimuli, spikes, seed)?;
        self.synchronize()
    }

    pub fn poisson_encode_async(
        &self,
        stimuli: &GpuBuffer<f32>,
        spikes: &mut GpuBuffer<u32>,
        seed: u32,
    ) -> GpuResult<()> {
        let n = stimuli.len();
        Self::expect_len("spikes", spikes.len(), n)?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("poisson_encode")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;

        let block = 256;
        let grid = Self::ceil_div_u32(n as u32, block);

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                stimuli.as_device_ptr(),
                spikes.as_device_ptr(),
                n as i32,
                seed,
            ))
            .map_err(|e| GpuError::LaunchFailed(format!("poisson_encode launch: {e:?}")))?;
        }

        Ok(())
    }

    /// Group-scaled ternary GEMV: `y = scale(W) @ x`.
    ///
    /// `packed_w` is row-major `M × ternary_word_count(K)` words.
    /// `scales` is `M × groups_per_row(K, group_size)` f32.
    /// When `skip_zeros` is true, zero trits are skipped on device.
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemv(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        self.ternary_gemv_async(packed_w, scales, x, y, m, k, group_size, skip_zeros)?;
        self.synchronize()
    }

    /// Async variant of [`Self::ternary_gemv`].
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemv_async(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        if m < 0 || k < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemv: m and k must be >= 0, got m={m} k={k}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemv: group_size must be > 0, got {group_size}"
            )));
        }

        let m_u = m as usize;
        let k_u = k as usize;
        Self::expect_len("y", y.len(), m_u)?;

        // Match host-ref: empty product is the zero vector (not "leave y untouched").
        if m == 0 {
            return Ok(());
        }
        if k == 0 {
            // Device memset — no host-sized staging buffer; preserve pooled tail.
            y.zero_prefix(m_u)?;
            return Ok(());
        }

        let group_u = group_size as usize;
        let words_per_row = k_u.div_ceil(TERNARY_VALUES_PER_WORD);
        let n_groups = k_u.div_ceil(group_u);

        Self::expect_len(
            "packed_w",
            packed_w.len(),
            m_u.saturating_mul(words_per_row),
        )?;
        Self::expect_len("scales", scales.len(), m_u.saturating_mul(n_groups))?;
        Self::expect_len("x", x.len(), k_u)?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("ternary_gemv")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        let block = 256u32;
        let grid = Self::ceil_div_u32(m as u32, block);
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemv");
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                packed_w.as_device_ptr(),
                scales.as_device_ptr(),
                x.as_device_ptr(),
                y.as_device_ptr(),
                m,
                k,
                group_size,
                skip,
            ))
        };
        range_pop!();
        launch_result.map_err(|e| GpuError::LaunchFailed(format!("ternary_gemv launch: {e:?}")))?;

        Ok(())
    }

    /// Group-scaled ternary GEMM: `C = scale(W) @ B`.
    ///
    /// `B` is `K × N` row-major f32; `C` is `M × N` row-major f32.
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemm(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        self.ternary_gemm_async(packed_w, scales, b, c, m, k, n, group_size, skip_zeros)?;
        self.synchronize()
    }

    /// Async variant of [`Self::ternary_gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn ternary_gemm_async(
        &self,
        packed_w: &GpuBuffer<u32>,
        scales: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &mut GpuBuffer<f32>,
        m: i32,
        k: i32,
        n: i32,
        group_size: i32,
        skip_zeros: bool,
    ) -> GpuResult<()> {
        if m < 0 || k < 0 || n < 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemm: m, k, n must be >= 0, got m={m} k={k} n={n}"
            )));
        }
        if group_size <= 0 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemm: group_size must be > 0, got {group_size}"
            )));
        }

        let m_u = m as usize;
        let k_u = k as usize;
        let n_u = n as usize;
        Self::expect_len("c", c.len(), m_u.saturating_mul(n_u))?;

        // Match host-ref: empty product zeros C (not "leave C untouched").
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // Device memset — no host-sized staging buffer; preserve pooled tail.
            c.zero_prefix(m_u.saturating_mul(n_u))?;
            return Ok(());
        }

        let group_u = group_size as usize;
        let words_per_row = k_u.div_ceil(TERNARY_VALUES_PER_WORD);
        let n_groups = k_u.div_ceil(group_u);

        Self::expect_len(
            "packed_w",
            packed_w.len(),
            m_u.saturating_mul(words_per_row),
        )?;
        Self::expect_len("scales", scales.len(), m_u.saturating_mul(n_groups))?;
        Self::expect_len("b", b.len(), k_u.saturating_mul(n_u))?;

        let kernels = self.kernels()?;
        let func = kernels.get_function("ternary_gemm")?;
        let stream = self.stream.as_ref().ok_or(GpuError::NoGpu)?;
        // m,n are nonnegative i32 after validation; product always fits u64.
        let total = (m as u64) * (n as u64);
        let block = 256u32;
        // Launch uses 1-D grid of u32 block indices over flattened M*N threads.
        let grid_u64 = total.div_ceil(block as u64);
        if grid_u64 > u32::MAX as u64 {
            return Err(GpuError::LaunchFailed(format!(
                "ternary_gemm: grid too large for 1-D launch ({grid_u64} blocks, M*N={total})"
            )));
        }
        let grid = grid_u64 as u32;
        let skip = if skip_zeros { 1i32 } else { 0i32 };

        range_push!("ternary_gemm");
        let launch_result = unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                packed_w.as_device_ptr(),
                scales.as_device_ptr(),
                b.as_device_ptr(),
                c.as_device_ptr(),
                m,
                k,
                n,
                group_size,
                skip,
            ))
        };
        range_pop!();
        launch_result.map_err(|e| GpuError::LaunchFailed(format!("ternary_gemm launch: {e:?}")))?;

        Ok(())
    }

    #[cfg(feature = "saaq")]
    fn require_state_neuron_count(state: &TemporalState, neuron_count: usize) -> GpuResult<()> {
        if state.neuron_count != neuron_count {
            return Err(GpuError::MemoryError(format!(
                "temporal neuron_count mismatch: state has {}, requested {neuron_count}",
                state.neuron_count
            )));
        }
        Ok(())
    }

    fn expect_len(name: &str, actual: usize, minimum: usize) -> GpuResult<()> {
        if actual < minimum {
            return Err(GpuError::MemoryError(format!(
                "{name} too small: need at least {minimum} elements, got {actual}"
            )));
        }
        Ok(())
    }

    fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
        value.div_ceil(divisor)
    }
}

impl Drop for GpuAccelerator {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "saaq"))]
mod tests {
    use super::*;
    use crate::gif::{GIF_ADAPTATION_SCALE, project_snapshot_current, saaq_find_best_walker};

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn corinth_saaq_fixture_selects_global_best() {
        let mut accelerator = GpuAccelerator::new();
        assert!(accelerator.is_ready(), "GPU not ready");

        let neuron_count = (8 * TEMPORAL_BLOCK_SIZE) as usize;
        accelerator
            .ensure_temporal_state(neuron_count)
            .expect("temporal state should allocate");

        let mut membrane = vec![0.0f32; neuron_count];
        let adaptation = vec![0.0f32; neuron_count];
        membrane[17] = 2.0;
        membrane[5 * TEMPORAL_BLOCK_SIZE as usize + 9] = 4.5;
        let expected = saaq_find_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE);

        {
            let state = accelerator
                .temporal_state
                .as_mut()
                .expect("temporal state should exist");
            state.membrane.upload(&membrane).expect("membrane upload");
            state
                .adaptation
                .upload(&adaptation)
                .expect("adaptation upload");
        }

        let best = accelerator
            .saaq_find_best_walker(neuron_count)
            .expect("SAAQ reduction should succeed");
        assert_eq!(best, expected);
        assert_eq!(best, 5 * TEMPORAL_BLOCK_SIZE + 9);
    }

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn corinth_saaq_fixture_tie_breaks_lower_index() {
        let mut accelerator = GpuAccelerator::new();
        assert!(accelerator.is_ready(), "GPU not ready");

        let neuron_count = (8 * TEMPORAL_BLOCK_SIZE) as usize;
        accelerator
            .ensure_temporal_state(neuron_count)
            .expect("temporal state should allocate");

        let mut membrane = vec![0.0f32; neuron_count];
        let adaptation = vec![0.0f32; neuron_count];
        membrane[11] = 3.0;
        membrane[3 * TEMPORAL_BLOCK_SIZE as usize + 4] = 3.0;
        let expected = saaq_find_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE);

        {
            let state = accelerator
                .temporal_state
                .as_mut()
                .expect("temporal state should exist");
            state.membrane.upload(&membrane).expect("membrane upload");
            state
                .adaptation
                .upload(&adaptation)
                .expect("adaptation upload");
        }

        let best = accelerator
            .saaq_find_best_walker(neuron_count)
            .expect("SAAQ reduction should succeed");
        assert_eq!(best, expected);
        assert_eq!(best, 11);
    }

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn project_snapshot_matches_host_ref() {
        let mut accelerator = GpuAccelerator::new();
        assert!(accelerator.is_ready(), "GPU not ready");

        let neuron_count = 64usize;
        let snap = SnapshotChannels {
            gpu_temp_c: 75.0,
            gpu_power_w: 280.0,
            cpu_tctl_c: 62.0,
            cpu_package_power_w: 90.0,
        };
        let expected = project_snapshot_current(snap, neuron_count);
        accelerator
            .ensure_temporal_state(neuron_count)
            .expect("temporal state");
        accelerator
            .project_snapshot_current(snap, neuron_count)
            .expect("project");
        let got = accelerator
            .temporal_state
            .as_ref()
            .expect("state")
            .input_current
            .to_vec()
            .expect("download input_current");
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "input_current[{i}] gpu={g} host={e}");
        }
    }
}
