// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::GpuAccelerator;
use crate::gif::TemporalDynamics;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::memory::GpuBuffer;
use cust::launch;

const LIF_BLOCK_SIZE: u32 = 256;
const DEFAULT_DYNAMIC_SHARED_MEM_BYTES: u32 = 48 * 1024;

impl GpuAccelerator {
    /// One LIF tick with the default dynamics with pre-summed external current.
    ///
    /// `membrane.len()` is the neuron count. `input_current`, `refractory`,
    /// and `spikes_out` must each contain at least that many elements.
    /// Synchronizes before returning.
    pub fn lif_step(
        &self,
        membrane: &mut GpuBuffer<f32>,
        input_current: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
    ) -> GpuResult<()> {
        self.lif_step_with_dynamics(
            membrane,
            input_current,
            refractory,
            spikes_out,
            TemporalDynamics::default(),
        )
    }

    /// One LIF tick with caller-selected validated dynamics.
    pub fn lif_step_with_dynamics(
        &self,
        membrane: &mut GpuBuffer<f32>,
        input_current: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
        dynamics: TemporalDynamics,
    ) -> GpuResult<()> {
        let launch_result = self.lif_step_with_dynamics_async(
            membrane,
            input_current,
            refractory,
            spikes_out,
            dynamics,
        );
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::lif_step`]; synchronize before reading or reusing the buffers.
    pub fn lif_step_async(
        &self,
        membrane: &mut GpuBuffer<f32>,
        input_current: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
    ) -> GpuResult<()> {
        self.lif_step_with_dynamics_async(
            membrane,
            input_current,
            refractory,
            spikes_out,
            TemporalDynamics::default(),
        )
    }

    /// Enqueue [`Self::lif_step_with_dynamics`]; synchronize before reading or
    /// reusing the buffers.
    pub fn lif_step_with_dynamics_async(
        &self,
        membrane: &mut GpuBuffer<f32>,
        input_current: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
        dynamics: TemporalDynamics,
    ) -> GpuResult<()> {
        dynamics.validate().map_err(GpuError::invalid_input)?;
        let n_neurons = membrane.len();
        Self::expect_len("input_current", input_current.len(), n_neurons)?;
        Self::expect_len("refractory", refractory.len(), n_neurons)?;
        Self::expect_len("spikes_out", spikes_out.len(), n_neurons)?;
        let n_neurons_i32 = kernel_dimension("lif_step", "n_neurons", n_neurons)?;
        if n_neurons == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self.kernels()?.get_function("lif_step")?;
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_neurons as u32, LIF_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(float* membrane, const float* I_ext,
        // unsigned int* refract, unsigned int* spikes_out, int n_neurons)`.
        // Types and ordering match. All arrays contain at least `n_neurons`
        // elements, `n_neurons` fits i32, and the grid covers every neuron.
        // The kernel guards `tid >= n_neurons`. Buffers remain alive until
        // the synchronous wrapper or the caller synchronizes the stream.
        unsafe {
            launch!(func<<<grid, LIF_BLOCK_SIZE, 0, stream>>>(
                membrane.as_device_ptr(),
                input_current.as_device_ptr(),
                refractory.as_device_ptr(),
                spikes_out.as_device_ptr(),
                n_neurons_i32,
                dynamics.lif_decay,
                dynamics.lif_threshold,
                dynamics.lif_reset,
                dynamics.lif_refractory_ticks,
            ))
            .map_err(|e| Self::launch_error("lif_step", e))?;
        }
        Ok(())
    }

    /// One LIF tick with the default dynamics with a row-major synaptic weight matrix.
    ///
    /// `membrane.len()` is the neuron count and `input_spikes.len()` is the
    /// input width. `weights` must contain at least `neurons × inputs`
    /// elements. The wrapper passes `inputs × size_of::<f32>()` bytes of
    /// dynamic shared memory to the kernel and rejects dimensions that cannot
    /// satisfy that contract. Synchronizes before returning.
    pub fn lif_step_weighted(
        &self,
        membrane: &mut GpuBuffer<f32>,
        weights: &GpuBuffer<f32>,
        input_spikes: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
    ) -> GpuResult<()> {
        self.lif_step_weighted_with_dynamics(
            membrane,
            weights,
            input_spikes,
            refractory,
            spikes_out,
            TemporalDynamics::default(),
        )
    }

    /// One weighted LIF tick with caller-selected validated dynamics.
    pub fn lif_step_weighted_with_dynamics(
        &self,
        membrane: &mut GpuBuffer<f32>,
        weights: &GpuBuffer<f32>,
        input_spikes: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
        dynamics: TemporalDynamics,
    ) -> GpuResult<()> {
        let launch_result = self.lif_step_weighted_with_dynamics_async(
            membrane,
            weights,
            input_spikes,
            refractory,
            spikes_out,
            dynamics,
        );
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::lif_step_weighted`]; synchronize before reading or
    /// reusing the buffers.
    pub fn lif_step_weighted_async(
        &self,
        membrane: &mut GpuBuffer<f32>,
        weights: &GpuBuffer<f32>,
        input_spikes: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
    ) -> GpuResult<()> {
        self.lif_step_weighted_with_dynamics_async(
            membrane,
            weights,
            input_spikes,
            refractory,
            spikes_out,
            TemporalDynamics::default(),
        )
    }

    /// Enqueue [`Self::lif_step_weighted_with_dynamics`]; synchronize before
    /// reading or reusing the buffers.
    pub fn lif_step_weighted_with_dynamics_async(
        &self,
        membrane: &mut GpuBuffer<f32>,
        weights: &GpuBuffer<f32>,
        input_spikes: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
        dynamics: TemporalDynamics,
    ) -> GpuResult<()> {
        dynamics.validate().map_err(GpuError::invalid_input)?;
        let n_neurons = membrane.len();
        let n_inputs = input_spikes.len();
        Self::expect_len("refractory", refractory.len(), n_neurons)?;
        Self::expect_len("spikes_out", spikes_out.len(), n_neurons)?;
        let n_neurons_i32 = kernel_dimension("lif_step_weighted", "n_neurons", n_neurons)?;
        let n_inputs_i32 = kernel_dimension("lif_step_weighted", "n_inputs", n_inputs)?;
        let weight_elements = matrix_elements("lif_step_weighted weights", n_neurons, n_inputs)?;
        Self::expect_len("weights", weights.len(), weight_elements)?;
        let shared_bytes = lif_weighted_shared_bytes(n_inputs)?;
        let shared_limit = Self::shared_memory_opt_in_limit()?;
        if shared_bytes > shared_limit {
            return Err(GpuError::invalid_input(format!(
                "lif_step_weighted: dynamic shared-memory size {shared_bytes} exceeds device opt-in limit {shared_limit}; reduce n_inputs"
            )));
        }
        if n_neurons == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self.kernels()?.get_function("lif_step_weighted")?;
        if shared_bytes > DEFAULT_DYNAMIC_SHARED_MEM_BYTES {
            Self::opt_in_dynamic_shared_mem(&func, shared_bytes, "lif_step_weighted")?;
        }
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_neurons as u32, LIF_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(float* membrane, const float* weights,
        // const float* input_spikes, unsigned int* refract,
        // unsigned int* spikes_out, int n_neurons, int n_inputs)`. Types and
        // ordering match. `weights` contains at least `n_neurons * n_inputs`
        // elements and all state arrays contain at least `n_neurons` elements.
        // `shared_bytes` is exactly `n_inputs * size_of::<f32>()`, nonzero
        // whenever `n_inputs` is nonzero, within the device opt-in limit, and
        // opted in above when it exceeds the default 48 KiB launch ceiling.
        // The grid covers every neuron and the kernel guards `tid >= n_neurons`.
        // Buffers remain alive until synchronization as documented above.
        unsafe {
            launch!(func<<<grid, LIF_BLOCK_SIZE, shared_bytes, stream>>>(
                membrane.as_device_ptr(),
                weights.as_device_ptr(),
                input_spikes.as_device_ptr(),
                refractory.as_device_ptr(),
                spikes_out.as_device_ptr(),
                n_neurons_i32,
                n_inputs_i32,
                dynamics.lif_decay,
                dynamics.lif_threshold,
                dynamics.lif_reset,
                dynamics.lif_refractory_ticks,
            ))
            .map_err(|e| Self::launch_error("lif_step_weighted", e))?;
        }
        Ok(())
    }

    /// One weighted analog-current LIF tick with per-neuron decay and
    /// threshold vectors plus scalar reset/refractory behavior.
    ///
    /// `membrane.len()` is the neuron count and `input_spikes.len()` is the
    /// input width. Despite the historical buffer name, input values may be
    /// continuous analog currents. `decay` and `threshold` must contain one
    /// finite value per neuron. Synchronizes before returning.
    #[allow(clippy::too_many_arguments)]
    pub fn lif_step_weighted_per_neuron(
        &self,
        membrane: &mut GpuBuffer<f32>,
        weights: &GpuBuffer<f32>,
        input_spikes: &GpuBuffer<f32>,
        decay: &GpuBuffer<f32>,
        threshold: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
        lif_reset: f32,
        lif_refractory_ticks: u32,
    ) -> GpuResult<()> {
        let launch_result = self.lif_step_weighted_per_neuron_async(
            membrane,
            weights,
            input_spikes,
            decay,
            threshold,
            refractory,
            spikes_out,
            lif_reset,
            lif_refractory_ticks,
        );
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::lif_step_weighted_per_neuron`]; synchronize before
    /// reading or reusing the buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn lif_step_weighted_per_neuron_async(
        &self,
        membrane: &mut GpuBuffer<f32>,
        weights: &GpuBuffer<f32>,
        input_spikes: &GpuBuffer<f32>,
        decay: &GpuBuffer<f32>,
        threshold: &GpuBuffer<f32>,
        refractory: &mut GpuBuffer<u32>,
        spikes_out: &mut GpuBuffer<u32>,
        lif_reset: f32,
        lif_refractory_ticks: u32,
    ) -> GpuResult<()> {
        if !lif_reset.is_finite() {
            return Err(GpuError::invalid_input(
                "lif_step_weighted_per_neuron: lif_reset must be finite",
            ));
        }
        let n_neurons = membrane.len();
        let n_inputs = input_spikes.len();
        for (name, len) in [
            ("decay", decay.len()),
            ("threshold", threshold.len()),
            ("refractory", refractory.len()),
            ("spikes_out", spikes_out.len()),
        ] {
            Self::expect_len(name, len, n_neurons)?;
        }
        let n_neurons_i32 =
            kernel_dimension("lif_step_weighted_per_neuron", "n_neurons", n_neurons)?;
        let n_inputs_i32 = kernel_dimension("lif_step_weighted_per_neuron", "n_inputs", n_inputs)?;
        let weight_elements =
            matrix_elements("lif_step_weighted_per_neuron weights", n_neurons, n_inputs)?;
        Self::expect_len("weights", weights.len(), weight_elements)?;
        let shared_bytes = lif_weighted_shared_bytes(n_inputs)?;
        let shared_limit = Self::shared_memory_opt_in_limit()?;
        if shared_bytes > shared_limit {
            return Err(GpuError::invalid_input(format!(
                "lif_step_weighted_per_neuron: dynamic shared-memory size {shared_bytes} exceeds device opt-in limit {shared_limit}; reduce n_inputs"
            )));
        }
        if n_neurons == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self
            .kernels()?
            .get_function("lif_step_weighted_per_neuron")?;
        if shared_bytes > DEFAULT_DYNAMIC_SHARED_MEM_BYTES {
            Self::opt_in_dynamic_shared_mem(&func, shared_bytes, "lif_step_weighted_per_neuron")?;
        }
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_neurons as u32, LIF_BLOCK_SIZE);

        // SAFETY: pointer/scalar ordering matches the CUDA entry. Weight,
        // input, state, decay, and threshold lengths were checked above;
        // shared memory exactly covers the input tile, with device opt-in
        // performed when required. The grid covers every neuron and buffers
        // remain alive until synchronization as documented above.
        unsafe {
            launch!(func<<<grid, LIF_BLOCK_SIZE, shared_bytes, stream>>>(
                membrane.as_device_ptr(),
                weights.as_device_ptr(),
                input_spikes.as_device_ptr(),
                decay.as_device_ptr(),
                threshold.as_device_ptr(),
                refractory.as_device_ptr(),
                spikes_out.as_device_ptr(),
                n_neurons_i32,
                n_inputs_i32,
                lif_reset,
                lif_refractory_ticks,
            ))
            .map_err(|e| Self::launch_error("lif_step_weighted_per_neuron", e))?;
        }
        Ok(())
    }

    /// Update an exponential spike-rate estimate in place.
    ///
    /// `rates.len()` is the neuron count and `spikes` must contain at least
    /// that many elements. Synchronizes before returning.
    pub fn spike_rate(
        &self,
        rates: &mut GpuBuffer<f32>,
        spikes: &GpuBuffer<f32>,
        alpha: f32,
    ) -> GpuResult<()> {
        let launch_result = self.spike_rate_async(rates, spikes, alpha);
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::spike_rate`]; synchronize before reading or reusing the buffers.
    pub fn spike_rate_async(
        &self,
        rates: &mut GpuBuffer<f32>,
        spikes: &GpuBuffer<f32>,
        alpha: f32,
    ) -> GpuResult<()> {
        let n_neurons = rates.len();
        Self::expect_len("spikes", spikes.len(), n_neurons)?;
        let n_neurons_i32 = kernel_dimension("spike_rate", "n_neurons", n_neurons)?;
        if n_neurons == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self.kernels()?.get_function("spike_rate")?;
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_neurons as u32, LIF_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(float* rates, const float* spikes,
        // int n_neurons, float alpha)`. Both arrays contain at least
        // `n_neurons` elements, the grid covers every neuron, and the kernel
        // guards `tid >= n_neurons`. Buffers remain alive until synchronization.
        unsafe {
            launch!(func<<<grid, LIF_BLOCK_SIZE, 0, stream>>>(
                rates.as_device_ptr(),
                spikes.as_device_ptr(),
                n_neurons_i32,
                alpha,
            ))
            .map_err(|e| Self::launch_error("spike_rate", e))?;
        }
        Ok(())
    }

    /// Reset every membrane potential in `membrane` to `v_reset`.
    pub fn reset_membrane(&self, membrane: &mut GpuBuffer<f32>, v_reset: f32) -> GpuResult<()> {
        let launch_result = self.reset_membrane_async(membrane, v_reset);
        let sync_result = self.synchronize();
        launch_result?;
        sync_result
    }

    /// Enqueue [`Self::reset_membrane`]; synchronize before reading or reusing the buffer.
    pub fn reset_membrane_async(
        &self,
        membrane: &mut GpuBuffer<f32>,
        v_reset: f32,
    ) -> GpuResult<()> {
        let n_neurons = membrane.len();
        let n_neurons_i32 = kernel_dimension("reset_membrane", "n_neurons", n_neurons)?;
        if n_neurons == 0 {
            self.require_ready()?;
            return Ok(());
        }

        let func = self.kernels()?.get_function("reset_membrane")?;
        let stream = self.ready_stream()?;
        let grid = Self::ceil_div_u32(n_neurons as u32, LIF_BLOCK_SIZE);

        // SAFETY: the CUDA entry is `(float* membrane, int n_neurons,
        // float v_reset)`. The buffer contains `n_neurons` elements, the grid
        // covers every neuron, and the kernel guards `tid >= n_neurons`. The
        // buffer remains alive until synchronization.
        unsafe {
            launch!(func<<<grid, LIF_BLOCK_SIZE, 0, stream>>>(
                membrane.as_device_ptr(),
                n_neurons_i32,
                v_reset,
            ))
            .map_err(|e| Self::launch_error("reset_membrane", e))?;
        }
        Ok(())
    }
}

fn kernel_dimension(operation: &str, name: &str, value: usize) -> GpuResult<i32> {
    i32::try_from(value).map_err(|_| {
        GpuError::invalid_input(format!(
            "{operation}: {name} {value} exceeds i32::MAX kernel limit"
        ))
    })
}

fn matrix_elements(label: &str, rows: usize, columns: usize) -> GpuResult<usize> {
    rows.checked_mul(columns).ok_or_else(|| {
        GpuError::invalid_input(format!("{label}: dimensions overflow host element count"))
    })
}

fn lif_weighted_shared_bytes(n_inputs: usize) -> GpuResult<u32> {
    let bytes = n_inputs
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            GpuError::invalid_input("lif_step_weighted: dynamic shared-memory size overflows usize")
        })?;
    let bytes = u32::try_from(bytes).map_err(|_| {
        GpuError::invalid_input(format!(
            "lif_step_weighted: dynamic shared-memory size {bytes} exceeds u32 launch limit"
        ))
    })?;
    if n_inputs > 0 && bytes == 0 {
        return Err(GpuError::invalid_input(
            "lif_step_weighted: refusing zero dynamic shared-memory launch with n_inputs > 0",
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::lif_weighted_shared_bytes;

    #[test]
    fn lif_weighted_shared_memory_contract() {
        assert_eq!(lif_weighted_shared_bytes(0).unwrap(), 0);
        assert_eq!(lif_weighted_shared_bytes(1).unwrap(), 4);
        assert_eq!(lif_weighted_shared_bytes(16).unwrap(), 64);
        assert!(lif_weighted_shared_bytes((u32::MAX as usize) / 4 + 1).is_err());
    }
}
