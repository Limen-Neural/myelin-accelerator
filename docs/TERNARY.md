# Ternary packing and group-scale GEMV / GEMM

Host-side ternary layouts, group scales, reference matmul, and device
kernels for [GH #9](https://github.com/Limen-Neural/myelin-accelerator/issues/9).

| Layer | Location |
|-------|----------|
| Host packing / scales / ref | `src/bitpacking.rs` |
| Device GEMV / GEMM | `cu/ternary_gemm.cu` → PTX `ternary_gemm_sm_120.ptx` |
| Launch wrappers | `GpuAccelerator::ternary_gemv` / `ternary_gemm` |

---

## Encoding (2-bit trits)

Each weight is a signed trit in `{-1, 0, +1}`. Codes are **LSB-first** in both
`u32` words and byte payloads:

| Trit | Code  |
|------|-------|
| `0`  | `0b00` |
| `+1` | `0b01` |
| `-1` | `0b10` |
| reserved | `0b11` (decoded as `0`) |

### Myelin `u32` packing

- 16 trits per `u32` (`TERNARY_VALUES_PER_WORD`).
- Element `i` lives in bits `(2*i, 2*i+1)` of word `i / 16`.
- APIs: `pack_ternary` / `unpack_ternary`, `ternary_word_count`.

### Matrix layout

Packed weight matrices are **row-major**:

- Row length in words: `ternary_word_count(k)`
- Total words: `packed_words_for_matrix(m, k) = m * ternary_word_count(k)`
- Use [`pack_ternary_matrix`] (not flat `pack_ternary` on `m*k`) so rows do not
  share a word when `k` is not a multiple of 16.

---

## Group scales

Default group size along the inner dimension `K`:

```text
DEFAULT_GROUP_SIZE = 128
```

Scales are `f32`, **row-major**, one scale per group of columns:

```text
groups_per_row(k, group_size) = k.div_ceil(group_size)
scales[m * groups_per_row + group],  group = k_index / group_size
scale_count(m, k, group_size) = m * groups_per_row
```

Helpers:

| Function | Role |
|----------|------|
| `uniform_group_scales(m, k, group_size, scale)` | Fill every group with a constant |
| `group_scales_from_abs_max(weights_f32, m, k, group_size)` | Per-group `max(|w|)`, or `1.0` if the group is all zero |

`weights_f32` for abs-max is row-major `m * k` (unpacked f32, not trits).

---

## GOZ1 / byte interop

Byte payloads store **4 trits per byte** with the same codes and LSB-first order
(`pack_ternary_bytes` / `unpack_ternary_bytes`).

**Interop rule:** a GOZ1-style payload of `n` trits is exactly the first

```text
ternary_payload_byte_len(n) = n.div_ceil(4)
```

little-endian bytes of myelin `u32` packing:

```text
pack_ternary(values) → packed_u32_as_le_bytes(words) → prefix of length payload_len
```

Conversely, `packed_u32_from_le_bytes` rebuilds words (zero-padding an incomplete
final word). Use this when comparing or converting external GOZ1 artifacts to
the myelin layout—not for running full experiment recipes (those stay out of
this crate; see `docs/ARCHITECTURE.md`).

---

## Host reference matmul (goldens)

CPU-only references for correctness tests (not GPU):

### `ternary_gemv_ref`

```text
y[m] = Σ_k  unpack(W[m,k]) * scales[m, group(k)] * x[k]
```

- `packed`: row-major ternary words
- `scales`: row-major group scales
- `x`: length `k`
- `skip_zeros`: skip zero trits (same numeric result)

### `ternary_gemm_ref`

```text
C[m,n] = Σ_k  unpack(W[m,k]) * scales[m, group(k)] * B[k,n]
```

- `C`: `m * n` row-major  
- `B`: `k * n` row-major  

---

## Device kernels (`cu/ternary_gemm.cu`)

| Kernel | Shape | Notes |
|--------|-------|--------|
| `ternary_gemv` | `(m×k) · k → m` | Packed W + group scales + dense `x` |
| `ternary_gemm` | `(m×k) · (k×n) → m×n` | Same packing; multi-RHS dense `B` |

Both take `skip_zeros` (non-zero → skip trit `0`). Wrappers:
`GpuAccelerator::ternary_gemv` / `ternary_gemm` (feature `cuda`).

GPU goldens:

```bash
cargo test --locked --features cuda -- --ignored ternary_
```

### Benchmarks (latency harness)

```bash
CUDA_NVCC=/usr/local/cuda/bin/nvcc \
  cargo run --example benchmark --profile bench --features bench,cuda
```

Rows include `ternary_gemv_1024x4096`, `ternary_gemv_1024x4096_skip_zeros`,
`ternary_gemm_256x1024x64`, and a host dense f32 GEMV compare.

### Profiling (optional local gate)

NVTX ranges: `ternary_gemv` / `ternary_gemm` (feature `cuda`).

```bash
# Timeline
nsys profile --trace=cuda,nvtx -o /tmp/ternary_nsys \
  cargo run --example benchmark --profile bench --features bench,cuda -- --iterations 20

# Kernel occupancy / memory (filter by name)
ncu --kernel-name-base function --kernel-name regex:ternary_gem \
  --set full -o /tmp/ternary_ncu \
  cargo run --example benchmark --profile bench --features bench,cuda -- --iterations 5
```

---

## Related

- `docs/ARCHITECTURE.md` — ownership (kernels here; GOZ1 experiment orchestration not here)
- `src/bitpacking.rs` — public host API and unit tests
- GH #9 / [LIM-890](https://linear.app/rpd-34/issue/LIM-890) — packed ternary path (Limen-Neural SoT; not rmems)
