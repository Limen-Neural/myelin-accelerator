// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

// ════════════════════════════════════════════════════════════════════
//  gpu/kernel.rs — Fatbin / PTX module loading and kernel management
//
//  Kernel binaries are produced by build.rs as nvcc fatbin files
//  (sm_120 SASS + compute_120 PTX as a JIT fallback) plus a matching
//  `.ptx` sidecar. Both are written to OUT_DIR and embedded at compile
//  time. There is no runtime file-system lookup.
//
//  Load order: fatbin (SASS) first; on failure, PTX JIT. When the PTX
//  path also fails we re-run cuModuleLoadDataEx with CU_JIT_LOG_VERBOSE
//  and capture CU_JIT_ERROR_LOG_BUFFER / CU_JIT_INFO_LOG_BUFFER.
//
//  The Blackwell-critical F16 GIF and SAAQ paths (feature `saaq`) launch
//  through the C ABI shim in ffi.rs; their symbols are still registered here
//  so consumers can profile the unfused baseline via get_function.
// ════════════════════════════════════════════════════════════════════

use crate::capability::KernelAvailability;
use crate::gpu::error::{GpuError, GpuResult};
use crate::launch_hook::{LaunchFailure, LaunchType, report_launch_failure};
use cust::error::CudaError;
use cust::function::Function;
use cust::module::Module;
use cust::sys as cuda;
use nvtx::{range_pop, range_push};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_uint;
use std::ptr;

static SPIKING_NETWORK_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/spiking_network_sm_120.fatbin"));
static SPIKING_NETWORK_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/spiking_network_sm_120.ptx"));

static VECTOR_SIMILARITY_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/vector_similarity_sm_120.fatbin"));
static VECTOR_SIMILARITY_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/vector_similarity_sm_120.ptx"));

static SATSOLVER_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/satsolver_sm_120.fatbin"));
static SATSOLVER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/satsolver_sm_120.ptx"));

static TERNARY_GEMM_FATBIN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/ternary_gemm_sm_120.fatbin"));
static TERNARY_GEMM_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ternary_gemm_sm_120.ptx"));

#[cfg(not(feature = "saaq"))]
const SPIKING_NETWORK_SYMBOLS: &[&str] = &[
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
];

#[cfg(feature = "saaq")]
const SPIKING_NETWORK_SYMBOLS: &[&str] = &[
    "poisson_encode",
    "project_snapshot_current",
    "lif_step",
    "lif_step_weighted",
    "gif_step_weighted",
    "gif_step_weighted_f16",
    "spike_rate",
    "reset_membrane",
    "stdp_update",
    "neuro_bias_logits",
    "membrane_dv_dt_reduce_pass1",
    "routing_entropy_reduce_pass1",
    "latent_reduce_pass2",
    "saaq_find_best_walker",
    "saaq_reduce_partials_f16",
];

/// Manages compiled fatbin/PTX modules and kernel function handles.
pub struct KernelModule {
    modules: HashMap<String, Module>,
    func_map: HashMap<String, String>,
}

pub(crate) struct KernelLoadFailure {
    pub(crate) error: GpuError,
    pub(crate) availability: KernelAvailability,
}

impl KernelModule {
    /// Load all modules from their compile-time-embedded images.
    ///
    /// On sm_120 hardware the driver picks precompiled SASS from the fatbin.
    /// Embedded PTX is used only when SASS is missing or incompatible.
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
            SPIKING_NETWORK_FATBIN,
            SPIKING_NETWORK_PTX,
            "spiking_network",
            SPIKING_NETWORK_SYMBOLS,
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
            VECTOR_SIMILARITY_FATBIN,
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
            SATSOLVER_FATBIN,
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
            TERNARY_GEMM_FATBIN,
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
        fatbin: &[u8],
        ptx: &str,
        mod_name: &str,
        funcs: &[&str],
    ) -> GpuResult<()> {
        let module = Self::load_module(fatbin, ptx, mod_name)?;
        for &func_name in funcs {
            if module.get_function(func_name).is_err() {
                return Err(GpuError::KernelNotFound(format!(
                    "{func_name} in {mod_name}"
                )));
            }
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
            .map_err(|e| GpuError::KernelNotFound(format!("{name}: {e}")))
    }

    fn load_module(fatbin: &[u8], ptx: &str, name: &str) -> GpuResult<Module> {
        if !fatbin.is_empty() {
            match Module::from_fatbin(fatbin, &[]) {
                Ok(module) => return Ok(module),
                Err(e) => {
                    eprintln!(
                        "[CUDA] fatbin load failed for '{name}': {e:?}; falling back to PTX JIT"
                    );
                }
            }
        }
        Self::load_module_from_ptx(ptx, name)
    }

    fn load_module_from_ptx(ptx: &str, name: &str) -> GpuResult<Module> {
        Module::from_ptx(ptx, &[]).map_err(|e| {
            let (error_log, info_log) = capture_jit_log_split(ptx.as_bytes());
            let combined_log = format_jit_diagnostics(&error_log, &info_log);
            eprintln!(
                "[CUDA JIT] Failed to load module '{name}': {e:?}\n\
                 --- CUDA JIT log ---\n{combined_log}\n--------------------"
            );
            let gpu_error = GpuError::ModuleLoadFailed(format!(
                "JIT compilation failed for '{name}': {e:?} \
                 (target: sm_120 — check driver ≥ 570 and CUDA toolkit ≥ 12.8)\n\
                 --- CUDA JIT log ---\n{combined_log}\n--------------------"
            ));
            report_launch_failure(LaunchFailure {
                kernel_name: name.to_string(),
                launch_type: LaunchType::PtxFatbin,
                grid: (0, 0, 0),
                block: (0, 0, 0),
                shared_mem: 0,
                neuron_count: None,
                error: gpu_error.to_string(),
                jit_error_log: nonempty_log(error_log),
                jit_info_log: nonempty_log(info_log),
            });
            gpu_error
        })
    }
}

fn nonempty_log(log: String) -> Option<String> {
    if log.is_empty() { None } else { Some(log) }
}

fn format_jit_diagnostics(error_log: &str, info_log: &str) -> String {
    if error_log.is_empty() && info_log.is_empty() {
        return "<driver returned no JIT diagnostics>".to_string();
    }

    let mut log = String::new();
    if !error_log.is_empty() {
        log.push_str("error: ");
        log.push_str(error_log);
    }
    if !info_log.is_empty() {
        if !log.is_empty() {
            log.push('\n');
        }
        log.push_str("info: ");
        log.push_str(info_log);
    }
    log
}

/// Re-run `cuModuleLoadDataEx` with error/info log buffers attached.
fn capture_jit_log_split(bytes: &[u8]) -> (String, String) {
    const LOG_CAP: usize = 16 * 1024;

    let mut image = bytes.to_vec();
    image.push(0);

    let mut error_buf = vec![0u8; LOG_CAP];
    let mut info_buf = vec![0u8; LOG_CAP];

    let mut options: [cuda::CUjit_option; 5] = [
        cuda::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        cuda::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        cuda::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda::CUjit_option::CU_JIT_LOG_VERBOSE,
    ];

    let mut option_values: [*mut c_void; 5] = [
        error_buf.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
        info_buf.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
        1 as *mut c_void,
    ];

    let mut module: cuda::CUmodule = ptr::null_mut();
    let result = unsafe {
        cuda::cuModuleLoadDataEx(
            &mut module as *mut cuda::CUmodule,
            image.as_ptr() as *const c_void,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
    };

    if result == cuda::cudaError_enum::CUDA_SUCCESS && !module.is_null() {
        unsafe {
            let _ = cuda::cuModuleUnload(module);
        }
    }

    (cstr_from_buf(&error_buf), cstr_from_buf(&info_buf))
}

fn cstr_from_buf(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    #[test]
    #[ignore] // requires GPU + driver ≥ 570
    fn test_load_kernels() {
        let _ctx = GpuContext::init().expect("Failed to initialize GPU context");
        let kernels = KernelModule::load().expect("Failed to load kernels");

        assert!(kernels.get_function("cosine_similarity_batched").is_ok());
        assert!(kernels.get_function("lif_step").is_ok());
        #[cfg(feature = "saaq")]
        {
            assert!(kernels.get_function("gif_step_weighted").is_ok());
            assert!(kernels.get_function("saaq_find_best_walker").is_ok());
        }
        assert!(kernels.get_function("satsolver_step").is_ok());
        assert!(kernels.get_function("ternary_gemv").is_ok());
        assert!(kernels.get_function("ternary_gemm").is_ok());
    }
}
