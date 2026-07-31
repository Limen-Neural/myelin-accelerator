cmake_minimum_required(VERSION 3.20)
project(myelin_accelerator CXX)

set(CMAKE_CXX_STANDARD 17)

# ── CUDA toolkit ─────────────────────────────────────────────────────
# nvcc 13.x cannot use CMake's CUDA language with GCC 16 libstdc++
# (first hit on 13.2; still true on 13.3 hosts). Mirror the flags from
# build.rs: compile .cu → .ptx only, using -std=c++17 -D__STRICT_ANSI__
# to dodge C++23 header paths.
# Prefer /usr/local/cuda (symlink) or override: -DCUDA_NVCC=/usr/local/cuda-13.3/bin/nvcc
set(CUDA_NVCC /usr/local/cuda/bin/nvcc CACHE FILEPATH "Path to nvcc")
set(CUDA_ARCH sm_120)
set(CUDA_COMMON_FLAGS
        -ptx
        -arch=${CUDA_ARCH}
        -O3
        --use_fast_math
        --restrict
        -std=c++17
        -D__STRICT_ANSI__
        --allow-unsupported-compiler
        --expt-relaxed-constexpr
        -Xcompiler -fno-builtin)

set(CU_SOURCES
        cu/satsolver.cu
        cu/spiking_network.cu
        cu/vector_similarity.cu
        cu/ternary_gemm.cu)
set(CU_HEADERS
        cu/common.cuh)

# Build each .cu → .ptx via custom command
set(PTX_OUTPUTS "")
foreach(cu_src ${CU_SOURCES})
    get_filename_component(cu_name ${cu_src} NAME_WE)
    set(ptx_out "${CMAKE_CURRENT_BINARY_DIR}/${cu_name}.ptx")
    add_custom_command(
            OUTPUT ${ptx_out}
            COMMAND ${CUDA_NVCC} ${CUDA_COMMON_FLAGS}
                    -I${CMAKE_SOURCE_DIR}/cu
                    -o ${ptx_out}
                    ${CMAKE_SOURCE_DIR}/${cu_src}
            DEPENDS
                    ${CMAKE_SOURCE_DIR}/${cu_src}
                    ${CMAKE_SOURCE_DIR}/cu/common.cuh
            COMMENT "Compiling CUDA kernel → PTX: ${cu_src}")
    list(APPEND PTX_OUTPUTS ${ptx_out})
endforeach()

# SOURCES= attaches .cu/.cuh for CLion/IDE indexing. Do not mark ALL when
# nvcc is missing — default `cmake --build` must stay usable on CPU-only hosts.
# Still not a native CUDA language target — build produces PTX only.
#
# Do not use if(IS_EXECUTABLE) here: that operator needs CMake ≥ 3.29, but
# this project requires only 3.20. EXISTS + NOT IS_DIRECTORY is portable.
set(MYELIN_HAS_NVCC FALSE)
if(EXISTS "${CUDA_NVCC}" AND NOT IS_DIRECTORY "${CUDA_NVCC}")
    set(MYELIN_HAS_NVCC TRUE)
endif()

if(MYELIN_HAS_NVCC)
    add_custom_target(cuda_kernels ALL
            DEPENDS ${PTX_OUTPUTS}
            SOURCES ${CU_SOURCES} ${CU_HEADERS})
else()
    # Indexing-only target (no PTX deps) so CLion still sees .cu/.cuh files.
    add_custom_target(cuda_kernels
            SOURCES ${CU_SOURCES} ${CU_HEADERS})
    message(STATUS "nvcc not found at ${CUDA_NVCC}; cuda_kernels is indexing-only (not in ALL)")
endif()

# ── Testing ──────────────────────────────────────────────────────────
# CPU-safe CTest entries (no GPU required). CUDA feature build + self-hosted
# PTX validation live in .github/workflows/ci.yml; local PTX compile is
# covered by cuda_kernel_build below.
enable_testing()

# Cargo test (CPU-safe, no CUDA feature)
add_custom_target(cargo_test
        COMMAND cargo test --locked
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running Cargo tests (CPU-safe)")

add_test(NAME cargo_tests
        COMMAND cargo test --locked
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Cargo build without default features (compile check)
add_custom_target(cargo_build_no_defaults
        COMMAND cargo build --locked --no-default-features
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Building Cargo with no default features")

add_test(NAME cargo_build_no_default_features
        COMMAND cargo build --locked --no-default-features
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Formatting (must match CI Lint job)
add_custom_target(cargo_fmt
        COMMAND cargo fmt --check
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Checking rustfmt")

add_test(NAME cargo_fmt_check
        COMMAND cargo fmt --check
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Clippy without CUDA (must match CI Lint job)
add_custom_target(cargo_clippy
        COMMAND cargo clippy --locked --no-default-features -- -D warnings
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Clippy (no default features)")

add_test(NAME cargo_clippy_no_default
        COMMAND cargo clippy --locked --no-default-features -- -D warnings
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Bench example compile (covers the --features bench CI blind spot)
add_custom_target(cargo_bench_example
        COMMAND cargo build --locked --features bench --example benchmark
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Building benchmark example (bench feature)")

add_test(NAME cargo_build_bench_example
        COMMAND cargo build --locked --features bench --example benchmark
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CUDA PTX compile check — only when nvcc is present. Building the
# indexing-only cuda_kernels target does not compile PTX.
if(MYELIN_HAS_NVCC)
    add_test(NAME cuda_kernel_build
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_kernels
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    message(STATUS "Skipping CTest cuda_kernel_build (nvcc not found at ${CUDA_NVCC})")
endif()

# Do not add_executable() on .cu files here. CLion "New Target" / native CUDA
# language is intentionally avoided (GCC 16 + nvcc host front-end). Quality
# for cu/*.cu and cu/common.cuh is: target `cuda_kernels` (nvcc -ptx) and/or
# `cargo build --features cuda` (build.rs).
