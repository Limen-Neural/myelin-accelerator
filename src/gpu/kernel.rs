// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/kernel.rs — PTX Module Loading and Kernel Management
//
//  PTX files are compiled by myelin-accelerator/build.rs into OUT_DIR
//  and embedded at compile time with include_str!.  There is no runtime
//  file-system lookup — the bytes travel with the binary.
// ════════════════════════════════════════════════════════════════════

use crate::capability::{KernelAvailability, sanitize_diagnostic};
use crate::gpu::error::{GpuError, GpuResult};
use cust::error::CudaError;
use cust::function::Function;
use cust::module::Module;
use std::collections::HashMap;

use nvtx::{range_pop, range_push};

// ── Compile-time PTX embedding ───────────────────────────────────────────────
//
// OUT_DIR is set by Cargo to the directory where build.rs wrote its outputs.
// include_str! expands at compile time, so no file-system access at runtime.
static SPIKING_NETWORK_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/spiking_network_sm_120.ptx"));

static VECTOR_SIMILARITY_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/vector_similarity_sm_120.ptx"));

static SATSOLVER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/satsolver_sm_120.ptx"));

static TERNARY_GEMM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ternary_gemm_sm_120.ptx"));

// ── KernelModule ─────────────────────────────────────────────────────────────

/// Manages compiled PTX modules and kernel function handles.
pub struct KernelModule {
    modules: HashMap<String, Module>,
    func_map: HashMap<String, String>,
}

pub(crate) struct KernelLoadFailure {
    pub(crate) error: GpuError,
    pub(crate) availability: KernelAvailability,
}

impl KernelModule {
    /// Load all PTX modules from their compile-time-embedded byte strings.
    ///
    /// The PTX is JIT-compiled by the CUDA driver on first call.
    /// On sm_120 hardware with an up-to-date driver this takes < 1 s.
    pub fn load() -> GpuResult<Self> {
        Self::load_with_availability().map_err(|failure| failure.error)
    }

    pub(crate) fn load_with_availability() -> Result<Self, KernelLoadFailure> {
        range_push!("KernelModule::load");
        let result = Self::load_inner();
        range_pop!();
        result
    }

    fn load_inner() -> Result<Self, KernelLoadFailure> {
        let mut modules = HashMap::new();
        let mut func_map = HashMap::new();
        let mut availability = KernelAvailability::compiled_unverified();

        if let Err(error) = Self::load_and_map(
            &mut modules,
            &mut func_map,
            SPIKING_NETWORK_PTX,
            "spiking_network",
            &[
                "poisson_encode",
                "lif_step",
                "lif_step_weighted",
                "spike_rate",
                "reset_membrane",
                "stdp_update",
                "neuro_bias_logits",
                "membrane_dv_dt_reduce_pass1",
                "routing_entropy_reduce_pass1",
                "latent_reduce_pass2",
            ],
        ) {
            availability.spiking_network = Some(false);
            return Err(KernelLoadFailure {
                error,
                availability,
            });
        }
        availability.spiking_network = Some(true);
        if let Err(error) = Self::load_and_map(
            &mut modules,
            &mut func_map,
            VECTOR_SIMILARITY_PTX,
            "vector_similarity",
            &["cosine_similarity_batched", "cosine_similarity_top_k"],
        ) {
            availability.vector_similarity = Some(false);
            return Err(KernelLoadFailure {
                error,
                availability,
            });
        }
        availability.vector_similarity = Some(true);
        if let Err(error) = Self::load_and_map(
            &mut modules,
            &mut func_map,
            SATSOLVER_PTX,
            "satsolver",
            &[
                "satsolver_init",
                "satsolver_step",
                "satsolver_aux_update",
                "satsolver_check_solution",
                "satsolver_extract",
                "satsolver_best_reduce_pass1",
                "satsolver_best_reduce_pass2",
            ],
        ) {
            availability.satsolver = Some(false);
            return Err(KernelLoadFailure {
                error,
                availability,
            });
        }
        availability.satsolver = Some(true);
        if let Err(error) = Self::load_and_map(
            &mut modules,
            &mut func_map,
            TERNARY_GEMM_PTX,
            "ternary_gemm",
            &["ternary_gemv", "ternary_gemm"],
        ) {
            availability.ternary_gemm = Some(false);
            return Err(KernelLoadFailure {
                error,
                availability,
            });
        }

        Ok(Self { modules, func_map })
    }

    fn load_and_map(
        modules: &mut HashMap<String, Module>,
        func_map: &mut HashMap<String, String>,
        ptx: &str,
        mod_name: &str,
        funcs: &[&str],
    ) -> GpuResult<()> {
        let module = Self::load_module_from_ptx(ptx, mod_name)?;
        for &func_name in funcs {
            module
                .get_function(func_name)
                .map_err(|error| function_lookup_error(func_name, error))?;
            func_map.insert(func_name.to_string(), mod_name.to_string());
        }
        modules.insert(mod_name.to_string(), module);
        Ok(())
    }

    /// Alias kept for satsolver call-sites.
    pub fn load_satsolver() -> GpuResult<Self> {
        Self::load()
    }

    /// Retrieve a kernel [`Function`] handle by name.
    pub fn get_function<'a>(&'a self, name: &str) -> GpuResult<Function<'a>> {
        let mod_name = self
            .func_map
            .get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;

        let module = self
            .modules
            .get(mod_name)
            .ok_or_else(|| GpuError::KernelNotFound(format!("module {mod_name} missing")))?;

        module
            .get_function(name)
            .map_err(|error| function_lookup_error(name, error))
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// JIT-compile a PTX string into a loaded CUDA module.
    fn load_module_from_ptx(ptx: &str, name: &str) -> GpuResult<Module> {
        Module::from_ptx(ptx, &[]).map_err(|error| {
            let detail = sanitize_diagnostic(&format!("{error:?}"));
            eprintln!("[CUDA JIT] Failed to load module '{name}': {detail}");
            module_load_error(name, error)
        })
    }
}

fn function_lookup_error(name: &str, error: CudaError) -> GpuError {
    let detail = sanitize_diagnostic(&format!("{name}: {error:?}"));
    if error == CudaError::NotFound {
        GpuError::KernelNotFound(detail)
    } else {
        GpuError::CudaError(detail)
    }
}

fn module_load_error(name: &str, error: CudaError) -> GpuError {
    let detail = sanitize_diagnostic(&format!("{error:?}"));
    if matches!(
        error,
        CudaError::InvalidImage
            | CudaError::NoBinaryForGpu
            | CudaError::InvalidPtx
            | CudaError::InvalidSource
    ) {
        GpuError::ModuleLoadFailed(format!(
            "JIT compilation failed for '{name}': {detail} \
             (target: sm_120 — check driver ≥ 570 and CUDA toolkit ≥ 12.8)"
        ))
    } else {
        GpuError::CudaError(format!("loading module '{name}': {detail}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use cust::error::CudaError;

    #[test]
    fn missing_symbol_is_distinct_from_runtime_lookup_failure() {
        assert!(matches!(
            function_lookup_error("missing", CudaError::NotFound),
            GpuError::KernelNotFound(_)
        ));
        let runtime = function_lookup_error("lif_step", CudaError::InvalidContext);
        assert!(matches!(runtime, GpuError::CudaError(_)));
        assert_eq!(
            runtime.fallback_reason(),
            Some(crate::FallbackReason::DriverRuntimeFailure)
        );
    }

    #[test]
    fn invalid_ptx_is_distinct_from_runtime_module_failure() {
        assert!(matches!(
            module_load_error("spiking_network", CudaError::InvalidPtx),
            GpuError::ModuleLoadFailed(_)
        ));
        let runtime = module_load_error("spiking_network", CudaError::InvalidContext);
        assert!(matches!(runtime, GpuError::CudaError(_)));
    }

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn test_load_kernels() {
        let _ctx = GpuContext::init().expect("Failed to initialize GPU context");
        let kernels = KernelModule::load().expect("Failed to load kernels");

        assert!(kernels.get_function("cosine_similarity_batched").is_ok());
        assert!(kernels.get_function("lif_step").is_ok());
        assert!(kernels.get_function("satsolver_step").is_ok());
        assert!(kernels.get_function("ternary_gemv").is_ok());
        assert!(kernels.get_function("ternary_gemm").is_ok());
    }
}
