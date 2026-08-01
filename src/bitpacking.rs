// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Binary and ternary bitpacking utilities.
//!
//! # Packing Layouts
//!
//! ## Binary (1-bit)
//! Each `u32` word holds 32 binary values. Bit `i` of word `w` encodes
//! element `w * 32 + i`. A set bit means `true`/`1`, a clear bit means
//! `false`/`0`.
//!
//! ## Ternary (2-bit)
//! Each `u32` word holds 16 ternary values. Bits `(2*i, 2*i+1)` of word
//! `w` encode element `w * 16 + i` (LSB-first within the word):
//! - `0b00` → `0`
//! - `0b01` → `+1`
//! - `0b10` → `-1`
//! - `0b11` → `0` (reserved, decoded as zero)
//!
//! Byte-oriented packing ([`pack_ternary_bytes`]) stores the same trit codes
//! four per byte, also LSB-first. A GOZ1-style payload of `n` trits is exactly
//! the first [`ternary_payload_byte_len`]`(n)` little-endian bytes of the
//! myelin `u32` packing ([`pack_ternary`] → [`packed_u32_as_le_bytes`]).
//!
//! # Group scales (ternary GEMV / GEMM)
//!
//! Per-row groups along the `K` (inner) dimension, default size
//! [`DEFAULT_GROUP_SIZE`] (128). Scale buffers are `f32`, **row-major**:
//!
//! ```text
//! scales[m * groups_per_row(k, group_size) + group]
//!   where group = k_index / group_size
//! ```
//!
//! Length is always [`scale_count`]`(m, k, group_size)`.
//! Packed weight matrices are row-major with
//! [`packed_words_for_matrix`]`(m, k)` words total
//! (`ternary_word_count(k)` words per row).
//!
//! Host reference matmul ([`ternary_gemv_ref`], [`ternary_gemm_ref`]) multiplies
//! unpacked trit codes by the group scale and the activation. Device kernels
//! `ternary_gemv` / `ternary_gemm` live in `cu/ternary_gemm.cu` (see `docs/TERNARY.md`).
//!
//! # Alignment
//! Packed buffers are naturally 4-byte aligned (they contain `u32`).
//! For CUDA vectorized 128-bit loads, ensure 16-byte alignment.

/// Values packed into a single `u32` for binary encoding.
pub const BINARY_VALUES_PER_WORD: usize = 32;

/// Values packed into a single `u32` for ternary encoding.
pub const TERNARY_VALUES_PER_WORD: usize = 16;

/// Default group size along `K` for per-group scales (ternary GEMV/GEMM).
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Trits packed into a single byte (GOZ1 / byte-oriented ternary payload).
const TERNARY_VALUES_PER_BYTE: usize = 4;

// ── Binary packing ──────────────────────────────────────────────────────────

/// Number of `u32` words needed to pack `n` binary values.
pub fn binary_word_count(n: usize) -> usize {
    n.div_ceil(BINARY_VALUES_PER_WORD)
}

/// Pack boolean values into a dense bit vector.
///
/// `true` → bit set (1), `false` → bit clear (0).
///
/// # Panics
/// Never panics; output length is always `binary_word_count(values.len())`.
pub fn pack_binary(values: &[bool]) -> Vec<u32> {
    let word_count = binary_word_count(values.len());
    let mut out = vec![0u32; word_count];
    for (i, &v) in values.iter().enumerate() {
        if v {
            out[i / BINARY_VALUES_PER_WORD] |= 1u32 << (i % BINARY_VALUES_PER_WORD);
        }
    }
    out
}

/// Unpack a dense bit vector back into boolean values.
///
/// Returns `n` values, where `n = values.len() * 32` if `count` is `None`.
/// If `count` is provided, returns exactly that many values (truncating
/// any bits beyond `count` in the last word).
pub fn unpack_binary(packed: &[u32], count: Option<usize>) -> Vec<bool> {
    let max_bits = packed.len() * BINARY_VALUES_PER_WORD;
    let total_bits = count.unwrap_or(max_bits).min(max_bits);
    let mut out = Vec::with_capacity(total_bits);
    for i in 0..total_bits {
        let word = i / BINARY_VALUES_PER_WORD;
        let bit = i % BINARY_VALUES_PER_WORD;
        out.push((packed[word] >> bit) & 1 != 0);
    }
    out
}

// ── Ternary packing ─────────────────────────────────────────────────────────

/// Number of `u32` words needed to pack `n` ternary values.
pub fn ternary_word_count(n: usize) -> usize {
    n.div_ceil(TERNARY_VALUES_PER_WORD)
}

/// Encode a single trit code: `0` → `0b00`, `+1` → `0b01`, `-1` → `0b10`.
/// Values outside `{-1, 0, +1}` are clamped.
#[inline]
fn encode_trit(v: i8) -> u32 {
    match v {
        0 => 0b00,
        1..=i8::MAX => 0b01,  // +1 and above clamped to +1
        i8::MIN..=-1 => 0b10, // -1 and below clamped to -1
    }
}

/// Decode a 2-bit trit code: `0b00`/`0b11` → `0`, `0b01` → `+1`, `0b10` → `-1`.
#[inline]
fn decode_trit(code: u32) -> i8 {
    match code & 0b11 {
        0b01 => 1i8,
        0b10 => -1i8,
        _ => 0i8,
    }
}

/// Pack signed ternary values (`{-1, 0, +1}`) into dense 2-bit encoding.
///
/// Encoding: `0` → `0b00`, `+1` → `0b01`, `-1` → `0b10`.
/// Values outside `{-1, 0, +1}` are clamped.
pub fn pack_ternary(values: &[i8]) -> Vec<u32> {
    let word_count = ternary_word_count(values.len());
    let mut out = vec![0u32; word_count];
    for (i, &v) in values.iter().enumerate() {
        let code = encode_trit(v);
        let word = i / TERNARY_VALUES_PER_WORD;
        let bit = (i % TERNARY_VALUES_PER_WORD) * 2;
        out[word] |= code << bit;
    }
    out
}

/// Unpack a dense 2-bit ternary vector back into signed values.
///
/// Returns `n` values, where `n = values.len() * 16` if `count` is `None`.
/// If `count` is provided, returns exactly that many values.
///
/// Decoding: `0b00` → `0`, `0b01` → `+1`, `0b10` → `-1`, `0b11` → `0`.
pub fn unpack_ternary(packed: &[u32], count: Option<usize>) -> Vec<i8> {
    let max_values = packed.len() * TERNARY_VALUES_PER_WORD;
    let total = count.unwrap_or(max_values).min(max_values);
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let word = i / TERNARY_VALUES_PER_WORD;
        let bit = (i % TERNARY_VALUES_PER_WORD) * 2;
        let code = (packed[word] >> bit) & 0b11;
        out.push(decode_trit(code));
    }
    out
}

// ── Layout helpers (group scales / packed matrices) ─────────────────────────

/// Number of scale groups along a row of length `k`.
///
/// Equal to `k.div_ceil(group_size)`.
///
/// # Panics
/// Panics if `group_size == 0`.
pub fn groups_per_row(k: usize, group_size: usize) -> usize {
    assert!(group_size > 0, "group_size must be > 0");
    k.div_ceil(group_size)
}

/// Total number of `f32` group scales for an `m × k` weight matrix.
///
/// Equal to `m * groups_per_row(k, group_size)`.
///
/// # Panics
/// Panics if `group_size == 0`.
pub fn scale_count(m: usize, k: usize, group_size: usize) -> usize {
    m.saturating_mul(groups_per_row(k, group_size))
}

/// Total number of `u32` words for a row-major packed ternary `m × k` matrix.
///
/// Equal to `m * ternary_word_count(k)`.
pub fn packed_words_for_matrix(m: usize, k: usize) -> usize {
    m.saturating_mul(ternary_word_count(k))
}

/// Pack a row-major `m × k` trit matrix for device kernels.
///
/// Each row is packed independently into [`ternary_word_count`]`(k)` words so
/// row `r` starts at word offset `r * ternary_word_count(k)`. This differs from
/// [`pack_ternary`] on a flat `m*k` slice when `k` is not a multiple of 16
/// (flat packing would share words across row boundaries).
///
/// # Panics
/// Panics if `values.len() < m * k`.
pub fn pack_ternary_matrix(values: &[i8], m: usize, k: usize) -> Vec<u32> {
    let need = m.saturating_mul(k);
    assert!(
        values.len() >= need,
        "values length {} < m*k = {}",
        values.len(),
        need
    );
    let words_per_row = ternary_word_count(k);
    let mut out = vec![0u32; m.saturating_mul(words_per_row)];
    for row in 0..m {
        let row_vals = &values[row * k..(row + 1) * k];
        let row_packed = pack_ternary(row_vals);
        debug_assert_eq!(row_packed.len(), words_per_row);
        let dest = row * words_per_row;
        out[dest..dest + words_per_row].copy_from_slice(&row_packed);
    }
    out
}

// ── GOZ1 / byte interop ─────────────────────────────────────────────────────

/// Number of payload bytes needed for `n` trits at 4 trits per byte.
///
/// A GOZ1-style ternary payload of length `n` is exactly the first
/// `ternary_payload_byte_len(n)` little-endian bytes of myelin `u32` packing
/// ([`pack_ternary`] followed by [`packed_u32_as_le_bytes`]).
pub fn ternary_payload_byte_len(n: usize) -> usize {
    n.div_ceil(TERNARY_VALUES_PER_BYTE)
}

/// Pack signed ternary values into a dense byte payload (4 trits per byte).
///
/// Uses the same trit codes as [`pack_ternary`] (`0b00` / `0b01` / `0b10`),
/// LSB-first within each byte. Output length is
/// [`ternary_payload_byte_len`]`(values.len())`.
pub fn pack_ternary_bytes(values: &[i8]) -> Vec<u8> {
    let byte_count = ternary_payload_byte_len(values.len());
    let mut out = vec![0u8; byte_count];
    for (i, &v) in values.iter().enumerate() {
        let code = encode_trit(v) as u8;
        let byte = i / TERNARY_VALUES_PER_BYTE;
        let bit = (i % TERNARY_VALUES_PER_BYTE) * 2;
        out[byte] |= code << bit;
    }
    out
}

/// Unpack a dense 4-trits-per-byte payload back into signed values.
///
/// Returns `n` values, where `n = packed.len() * 4` if `count` is `None`.
/// If `count` is provided, returns at most that many values (capped by the
/// payload capacity).
pub fn unpack_ternary_bytes(packed: &[u8], count: Option<usize>) -> Vec<i8> {
    let max_values = packed.len() * TERNARY_VALUES_PER_BYTE;
    let total = count.unwrap_or(max_values).min(max_values);
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let byte = i / TERNARY_VALUES_PER_BYTE;
        let bit = (i % TERNARY_VALUES_PER_BYTE) * 2;
        let code = (packed[byte] >> bit) & 0b11;
        out.push(decode_trit(code as u32));
    }
    out
}

/// Serialize packed ternary `u32` words as little-endian bytes.
pub fn packed_u32_as_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Deserialize little-endian bytes into `u32` words.
///
/// An incomplete final word is zero-padded on the high bytes.
pub fn packed_u32_from_le_bytes(bytes: &[u8]) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let word_count = bytes.len().div_ceil(4);
    let mut out = Vec::with_capacity(word_count);
    let mut chunks = bytes.chunks_exact(4);
    for chunk in chunks.by_ref() {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut last = [0u8; 4];
        last[..rem.len()].copy_from_slice(rem);
        out.push(u32::from_le_bytes(last));
    }
    out
}

// ── Group scales ────────────────────────────────────────────────────────────

/// Build a uniform group-scale buffer: every entry equals `scale`.
///
/// Length is [`scale_count`]`(m, k, group_size)`.
///
/// # Panics
/// Panics if `group_size == 0`.
pub fn uniform_group_scales(m: usize, k: usize, group_size: usize, scale: f32) -> Vec<f32> {
    vec![scale; scale_count(m, k, group_size)]
}

/// Derive per-group scales from the absolute max of each K-group.
///
/// `weights_f32` is row-major `m * k`. For each group of the K dimension,
/// `scale = max(|w|)` over the group, or `1.0` if all weights in the group
/// are zero.
///
/// # Panics
/// Panics if `group_size == 0` or if `weights_f32.len() < m * k`.
///
/// Non-finite weights: only finite values update the running abs-max
/// (`NaN` / `±Inf` are ignored). An all-non-finite (or all-zero) group keeps
/// the default scale `1.0`. Callers that need strict rejection should scan
/// inputs before packing.
pub fn group_scales_from_abs_max(
    weights_f32: &[f32],
    m: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    let need = m.saturating_mul(k);
    assert!(
        weights_f32.len() >= need,
        "weights_f32 length {} < m*k = {}",
        weights_f32.len(),
        need
    );
    let gpr = groups_per_row(k, group_size);
    let mut scales = vec![1.0f32; m.saturating_mul(gpr)];
    for row in 0..m {
        let row_base = row * k;
        for g in 0..gpr {
            let start = g * group_size;
            let end = (start + group_size).min(k);
            let mut max_abs = 0.0f32;
            for kk in start..end {
                let a = weights_f32[row_base + kk].abs();
                // Skip NaN and Inf so they cannot poison the scale.
                if a.is_finite() && a > max_abs {
                    max_abs = a;
                }
            }
            scales[row * gpr + g] = if max_abs > 0.0 { max_abs } else { 1.0 };
        }
    }
    scales
}

// ── Host reference matmul (goldens; not GPU) ────────────────────────────────

/// Host reference ternary GEMV: `y[m] = W_row[m] · x` with group scales.
///
/// For each output row `m`:
/// `y[m] = Σ_k unpack(W[m,k]) * scales[m, group(k)] * x[k]`.
///
/// `packed` is row-major with `ternary_word_count(k)` words per row.
/// `scales` is row-major with `groups_per_row(k, group_size)` entries per row.
/// When `skip_zeros` is true, zero-valued trits are skipped (result identical).
///
/// # Panics
/// Panics if dimensions are inconsistent with buffer lengths, or `group_size == 0`.
// Mirrors device kernel ABI (buffers + dims + flags); grouping would not help callers.
#[allow(clippy::too_many_arguments)]
pub fn ternary_gemv_ref(
    packed: &[u32],
    scales: &[f32],
    x: &[f32],
    m: usize,
    k: usize,
    group_size: usize,
    skip_zeros: bool,
) -> Vec<f32> {
    let words_per_row = ternary_word_count(k);
    let gpr = groups_per_row(k, group_size);
    assert!(
        packed.len() >= m.saturating_mul(words_per_row),
        "packed too short for m×k"
    );
    assert!(
        scales.len() >= m.saturating_mul(gpr),
        "scales too short for m×groups"
    );
    assert!(x.len() >= k, "x too short for k");

    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let row_words = &packed[row * words_per_row..(row + 1) * words_per_row];
        let trits = unpack_ternary(row_words, Some(k));
        let mut acc = 0.0f32;
        for kk in 0..k {
            let w = trits[kk];
            if skip_zeros && w == 0 {
                continue;
            }
            let g = kk / group_size;
            let s = scales[row * gpr + g];
            acc += (w as f32) * s * x[kk];
        }
        y[row] = acc;
    }
    y
}

/// Host reference ternary GEMM: `C = W · B` with group scales.
///
/// `C` is `m × n` row-major; `B` is `k × n` row-major;
/// `C[m,n] = Σ_k W[m,k] * scale[m, group(k)] * B[k,n]`.
///
/// `packed` / `scales` layouts match [`ternary_gemv_ref`].
///
/// # Panics
/// Panics if dimensions are inconsistent with buffer lengths, or `group_size == 0`.
// Mirrors device kernel ABI (buffers + dims + flags); grouping would not help callers.
#[allow(clippy::too_many_arguments)]
pub fn ternary_gemm_ref(
    packed: &[u32],
    scales: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    skip_zeros: bool,
) -> Vec<f32> {
    let words_per_row = ternary_word_count(k);
    let gpr = groups_per_row(k, group_size);
    assert!(
        packed.len() >= m.saturating_mul(words_per_row),
        "packed too short for m×k"
    );
    assert!(
        scales.len() >= m.saturating_mul(gpr),
        "scales too short for m×groups"
    );
    assert!(b.len() >= k.saturating_mul(n), "b too short for k×n");

    let mut c = vec![0.0f32; m.saturating_mul(n)];
    for row in 0..m {
        let row_words = &packed[row * words_per_row..(row + 1) * words_per_row];
        let trits = unpack_ternary(row_words, Some(k));
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let w = trits[kk];
                if skip_zeros && w == 0 {
                    continue;
                }
                let g = kk / group_size;
                let s = scales[row * gpr + g];
                acc += (w as f32) * s * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Binary roundtrip ────────────────────────────────────────────────────

    #[test]
    fn binary_empty() {
        let packed = pack_binary(&[]);
        assert!(packed.is_empty());
        let unpacked = unpack_binary(&packed, Some(0));
        assert!(unpacked.is_empty());
    }

    #[test]
    fn binary_single_word() {
        let values: Vec<bool> = (0..32).map(|i| i % 3 == 0).collect();
        let packed = pack_binary(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_binary(&packed, Some(32));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_multi_word() {
        let values: Vec<bool> = (0..100).map(|i| i % 7 < 3).collect();
        let packed = pack_binary(&values);
        assert_eq!(packed.len(), binary_word_count(100));
        let unpacked = unpack_binary(&packed, Some(100));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_all_true() {
        let values = vec![true; 64];
        let packed = pack_binary(&values);
        assert_eq!(packed, vec![0xFFFF_FFFF, 0xFFFF_FFFF]);
        let unpacked = unpack_binary(&packed, Some(64));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_all_false() {
        let values = vec![false; 64];
        let packed = pack_binary(&values);
        assert_eq!(packed, vec![0u32, 0u32]);
        let unpacked = unpack_binary(&packed, Some(64));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn binary_exact_boundary() {
        // Exactly 32 values → 1 word
        let values = vec![true; 32];
        let packed = pack_binary(&values);
        assert_eq!(packed.len(), 1);
        // 33 values → 2 words
        let mut values33 = values;
        values33.push(false);
        let packed = pack_binary(&values33);
        assert_eq!(packed.len(), 2);
    }

    // ── Ternary roundtrip ───────────────────────────────────────────────────

    #[test]
    fn ternary_empty() {
        let packed = pack_ternary(&[]);
        assert!(packed.is_empty());
        let unpacked = unpack_ternary(&packed, Some(0));
        assert!(unpacked.is_empty());
    }

    #[test]
    fn ternary_single_word() {
        let values: Vec<i8> = vec![0, 1, -1, 0, 1, 1, -1, -1, 0, 1, -1, 0, 1, -1, 0, 0];
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_ternary(&packed, Some(16));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn ternary_multi_word() {
        let values: Vec<i8> = (0..50)
            .map(|i| match i % 3 {
                0 => 0i8,
                1 => 1i8,
                _ => -1i8,
            })
            .collect();
        let packed = pack_ternary(&values);
        assert_eq!(packed.len(), ternary_word_count(50));
        let unpacked = unpack_ternary(&packed, Some(50));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn ternary_all_zero() {
        let values = vec![0i8; 32];
        let packed = pack_ternary(&values);
        assert_eq!(packed, vec![0u32, 0u32]);
        let unpacked = unpack_ternary(&packed, Some(32));
        assert_eq!(values, unpacked);
    }

    #[test]
    fn ternary_clamping() {
        // Values outside {-1, 0, +1} should be clamped
        let packed = pack_ternary(&[100i8, -100i8, 0i8]);
        let unpacked = unpack_ternary(&packed, Some(3));
        assert_eq!(unpacked, vec![1i8, -1i8, 0i8]);
    }

    #[test]
    fn ternary_reserved_decodes_to_zero() {
        // Manually craft a word with 0b11 in one slot
        let mut word = 0u32;
        // Put 0b11 in slot 0 (bits 0-1)
        word |= 0b11;
        // Put 0b01 in slot 1 (bits 2-3)
        word |= 0b01 << 2;
        let unpacked = unpack_ternary(&[word], Some(2));
        assert_eq!(unpacked[0], 0i8); // 0b11 → 0
        assert_eq!(unpacked[1], 1i8); // 0b01 → +1
    }

    // ── Word count helpers ──────────────────────────────────────────────────

    #[test]
    fn word_count_helpers() {
        assert_eq!(binary_word_count(0), 0);
        assert_eq!(binary_word_count(1), 1);
        assert_eq!(binary_word_count(32), 1);
        assert_eq!(binary_word_count(33), 2);

        assert_eq!(ternary_word_count(0), 0);
        assert_eq!(ternary_word_count(1), 1);
        assert_eq!(ternary_word_count(16), 1);
        assert_eq!(ternary_word_count(17), 2);
    }

    // ── Layout helpers ──────────────────────────────────────────────────────

    #[test]
    fn groups_per_row_and_scale_count() {
        assert_eq!(DEFAULT_GROUP_SIZE, 128);
        assert_eq!(groups_per_row(0, 128), 0);
        assert_eq!(groups_per_row(1, 128), 1);
        assert_eq!(groups_per_row(128, 128), 1);
        assert_eq!(groups_per_row(129, 128), 2);
        assert_eq!(groups_per_row(256, 64), 4);

        assert_eq!(scale_count(2, 256, 128), 4);
        assert_eq!(scale_count(3, 100, 128), 3);
        assert_eq!(packed_words_for_matrix(2, 20), 2 * ternary_word_count(20));
        assert_eq!(packed_words_for_matrix(0, 16), 0);
    }

    #[test]
    #[should_panic(expected = "group_size must be > 0")]
    fn groups_per_row_zero_panics() {
        let _ = groups_per_row(10, 0);
    }

    // ── GOZ1 / byte interop ─────────────────────────────────────────────────

    #[test]
    fn ternary_payload_byte_len_helpers() {
        assert_eq!(ternary_payload_byte_len(0), 0);
        assert_eq!(ternary_payload_byte_len(1), 1);
        assert_eq!(ternary_payload_byte_len(4), 1);
        assert_eq!(ternary_payload_byte_len(5), 2);
        assert_eq!(ternary_payload_byte_len(16), 4);
    }

    #[test]
    fn pack_ternary_bytes_matches_le_prefix_of_pack_ternary() {
        // Various lengths including partial last byte / last u32 word.
        for n in [0usize, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 50, 128] {
            let values: Vec<i8> = (0..n)
                .map(|i| match i % 3 {
                    0 => 0i8,
                    1 => 1i8,
                    _ => -1i8,
                })
                .collect();
            let bytes = pack_ternary_bytes(&values);
            assert_eq!(bytes.len(), ternary_payload_byte_len(n));

            let words = pack_ternary(&values);
            let le = packed_u32_as_le_bytes(&words);
            let prefix_len = ternary_payload_byte_len(n);
            assert_eq!(
                &bytes[..],
                &le[..prefix_len],
                "GOZ1 payload mismatch for n={n}"
            );

            // Roundtrip bytes
            let unpacked = unpack_ternary_bytes(&bytes, Some(n));
            assert_eq!(unpacked, values);
        }
    }

    #[test]
    fn packed_u32_le_bytes_roundtrip_and_pad() {
        let words = vec![0x0102_0304u32, 0xAABB_CCDDu32];
        let bytes = packed_u32_as_le_bytes(&words);
        assert_eq!(bytes, vec![0x04, 0x03, 0x02, 0x01, 0xDD, 0xCC, 0xBB, 0xAA]);
        assert_eq!(packed_u32_from_le_bytes(&bytes), words);

        // Incomplete last word padded with zeros on high bytes
        let partial = packed_u32_from_le_bytes(&[0x11, 0x22, 0x33]);
        assert_eq!(partial, vec![u32::from_le_bytes([0x11, 0x22, 0x33, 0x00])]);

        assert!(packed_u32_from_le_bytes(&[]).is_empty());
        assert!(packed_u32_as_le_bytes(&[]).is_empty());
    }

    // ── Group scales ────────────────────────────────────────────────────────

    #[test]
    fn pack_ternary_matrix_row_pad_differs_from_flat_when_k_not_multiple_of_16() {
        // m=3, k=20: each row needs 2 words; flat pack of 60 trits needs 4 words.
        let m = 3usize;
        let k = 20usize;
        let values: Vec<i8> = (0..m * k)
            .map(|i| match i % 3 {
                0 => 0i8,
                1 => 1i8,
                _ => -1i8,
            })
            .collect();
        let matrix = pack_ternary_matrix(&values, m, k);
        let flat = pack_ternary(&values);
        assert_eq!(matrix.len(), packed_words_for_matrix(m, k));
        assert_eq!(matrix.len(), m * ternary_word_count(k));
        assert_eq!(ternary_word_count(k), 2);
        assert_eq!(flat.len(), ternary_word_count(m * k));
        assert_ne!(
            matrix.len(),
            flat.len(),
            "row-padded matrix packing must not collapse across rows"
        );
        for row in 0..m {
            let words = ternary_word_count(k);
            let row_words = &matrix[row * words..(row + 1) * words];
            let unpacked = unpack_ternary(row_words, Some(k));
            assert_eq!(&unpacked[..], &values[row * k..(row + 1) * k]);
        }
    }

    #[test]
    fn group_scales_length_correct() {
        let m = 3usize;
        let k = 200usize;
        let gs = 128usize;
        let uniform = uniform_group_scales(m, k, gs, 0.5);
        assert_eq!(uniform.len(), scale_count(m, k, gs));
        assert!(uniform.iter().all(|&s| s == 0.5));

        let weights: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let from_max = group_scales_from_abs_max(&weights, m, k, gs);
        assert_eq!(from_max.len(), scale_count(m, k, gs));
        assert_eq!(from_max.len(), m * groups_per_row(k, gs));
    }

    #[test]
    fn group_scales_from_abs_max_values() {
        // 1×4 matrix, group_size 2
        // row: [1.0, -3.0, 0.0, 0.0] → groups max(|w|)=3.0 and 1.0 (all-zero → 1.0)
        let w = vec![1.0f32, -3.0, 0.0, 0.0];
        let s = group_scales_from_abs_max(&w, 1, 4, 2);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 3.0).abs() < 1e-6);
        assert!((s[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn group_scales_ignore_non_finite() {
        // NaN/Inf must not set infinite scales; finite peers still win.
        let w = vec![f32::NAN, 2.0, f32::INFINITY, f32::NEG_INFINITY];
        let s = group_scales_from_abs_max(&w, 1, 4, 2);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 2.0).abs() < 1e-6, "s0={}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-6, "all-non-finite → 1.0, s1={}", s[1]);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    // ── Host reference matmul ───────────────────────────────────────────────

    #[test]
    fn ternary_gemv_ref_known_2x4() {
        // W = [[+1, 0, -1, +1],
        //      [ 0,+1,  0, -1]]
        // scales all 2.0, group_size 4 (one group per row)
        // x = [1, 2, 3, 4]
        // y0 = 2*(1*1 + 0*2 + (-1)*3 + 1*4) = 2*(1 - 3 + 4) = 4
        // y1 = 2*(0*1 + 1*2 + 0*3 + (-1)*4) = 2*(2 - 4) = -4
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
        let packed = pack_ternary_matrix(&w, 2, 4);
        let scales = uniform_group_scales(2, 4, 4, 2.0);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];

        for skip in [false, true] {
            let y = ternary_gemv_ref(&packed, &scales, &x, 2, 4, 4, skip);
            assert_eq!(y.len(), 2);
            assert!((y[0] - 4.0).abs() < 1e-5, "y0={}", y[0]);
            assert!((y[1] - (-4.0)).abs() < 1e-5, "y1={}", y[1]);
        }
    }

    #[test]
    fn ternary_gemv_ref_multi_group() {
        // k=4, group_size=2 → two groups; scales [1.0, 10.0]
        // W row = [+1, +1, +1, +1], x = [1,1,1,1]
        // y = 1*1*1 + 1*1*1 + 1*10*1 + 1*10*1 = 22
        let packed = pack_ternary(&[1i8, 1, 1, 1]);
        let scales = vec![1.0f32, 10.0];
        let x = vec![1.0f32; 4];
        let y = ternary_gemv_ref(&packed, &scales, &x, 1, 4, 2, false);
        assert!((y[0] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn ternary_gemm_ref_consistent_with_gemv_columns() {
        // M=2, K=4, N=3
        let w: Vec<i8> = vec![1, 0, -1, 1, 0, 1, 0, -1];
        let packed = pack_ternary_matrix(&w, 2, 4);
        let scales = uniform_group_scales(2, 4, 2, 1.5);
        // B is K×N row-major
        let b: Vec<f32> = vec![
            1.0, 0.0, 2.0, // k=0
            2.0, 1.0, 0.0, // k=1
            3.0, 2.0, 1.0, // k=2
            4.0, 3.0, 2.0, // k=3
        ];
        let n = 3usize;
        let c = ternary_gemm_ref(&packed, &scales, &b, 2, 4, n, 2, true);

        for col in 0..n {
            let x: Vec<f32> = (0..4).map(|kk| b[kk * n + col]).collect();
            let y = ternary_gemv_ref(&packed, &scales, &x, 2, 4, 2, false);
            assert!((c[col] - y[0]).abs() < 1e-5);
            assert!((c[n + col] - y[1]).abs() < 1e-5);
        }
    }

    // ── Property-based roundtrip tests ──────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn binary_roundtrip(values in proptest::collection::vec(any::<bool>(), 0..=512)) {
            let packed = pack_binary(&values);
            let unpacked = unpack_binary(&packed, Some(values.len()));
            prop_assert_eq!(&values, &unpacked);
        }

        #[test]
        fn binary_word_count_never_zero_for_nonempty(
            n in 1usize..=10_000
        ) {
            prop_assert!(binary_word_count(n) > 0);
        }

        #[test]
        fn ternary_roundtrip(
            values in proptest::collection::vec(
                prop_oneof![Just(0i8), Just(1i8), Just(-1i8)],
                0..=512
            )
        ) {
            let packed = pack_ternary(&values);
            let unpacked = unpack_ternary(&packed, Some(values.len()));
            prop_assert_eq!(&values, &unpacked);
        }

        #[test]
        fn ternary_clamp_roundtrip(
            values in proptest::collection::vec(any::<i8>(), 0..=512)
        ) {
            // Pack raw values (they get clamped), then unpack and verify
            // each element matches the clamped input.
            let packed = pack_ternary(&values);
            let unpacked = unpack_ternary(&packed, Some(values.len()));
            for (i, (&inp, &out)) in values.iter().zip(unpacked.iter()).enumerate() {
                let expected = match inp {
                    0 => 0i8,
                    1..=i8::MAX => 1i8,
                    i8::MIN..=-1 => -1i8,
                };
                prop_assert_eq!(out, expected, "mismatch at index {}: input={}", i, inp);
            }
        }

        #[test]
        fn ternary_word_count_never_zero_for_nonempty(
            n in 1usize..=10_000
        ) {
            prop_assert!(ternary_word_count(n) > 0);
        }

        #[test]
        fn ternary_bytes_match_u32_le_prefix(
            values in proptest::collection::vec(
                prop_oneof![Just(0i8), Just(1i8), Just(-1i8)],
                0..=256
            )
        ) {
            let bytes = pack_ternary_bytes(&values);
            let le = packed_u32_as_le_bytes(&pack_ternary(&values));
            let n = values.len();
            prop_assert_eq!(bytes.len(), ternary_payload_byte_len(n));
            prop_assert_eq!(&bytes[..], &le[..ternary_payload_byte_len(n)]);
            let unpacked = unpack_ternary_bytes(&bytes, Some(n));
            prop_assert_eq!(&values, &unpacked);
        }
    }
}
