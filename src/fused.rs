// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host reference and VRAM-traffic model for fused routing + SAAQ kernels.
//!
//! Device kernels live in `cu/fused_routing_saaq.cu` and the unfused SAAQ
//! baseline in `cu/spiking_network.cu`. This module is CPU-safe so CI can
//! check numerical parity and traffic reduction without a GPU.

/// Matches `GIF_ADAPTATION_SCALE` in corinth-canal / `spiking_network.cu`.
pub const GIF_ADAPTATION_SCALE: f32 = 0.22;

/// Launch block size used by the fused pass1 / SAAQ kernels.
pub const FUSED_BLOCK_SIZE: usize = 256;

/// Register-resident top-k cap in `routing_saaq_fused_pass1`.
pub const MAX_FUSED_TOP_K: usize = 8;

/// Matches the device SAAQ sentinel used to mask empty lanes.
const SAAQ_SENTINEL: f32 = f32::NEG_INFINITY;

/// Blackwell SM occupancy constants used for the analytical model.
/// Compute capability 12.0 (GeForce Blackwell / sm_120): 48 resident warps
/// = 1536 threads, 65536 32-bit registers, 32 blocks per SM
/// (NVIDIA Blackwell Tuning Guide / CUDA programming guide).
pub const SM120_MAX_THREADS_PER_SM: u32 = 1536;
pub const SM120_REGS_PER_SM: u32 = 65536;
pub const SM120_MAX_BLOCKS_PER_SM: u32 = 32;

/// Output of fused routing + SAAQ selection (host or device).
#[derive(Debug, Clone, PartialEq)]
pub struct FusedRoutingSaaqResult {
    pub top_k_indices: Vec<i32>,
    pub entropy_sum: f32,
    pub entropy_max: f32,
    pub best_walker: u32,
}

/// Estimated device-side traffic for one fused or unfused invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrafficReport {
    pub path: &'static str,
    pub n_nodes: usize,
    pub n_routes: usize,
    pub top_k: usize,
    pub n_blocks: usize,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub launches: u32,
    pub materializes_routing_matrix: bool,
}

impl TrafficReport {
    pub fn bytes_total(&self) -> u64 {
        self.bytes_read.saturating_add(self.bytes_written)
    }
}

/// Occupancy estimate from a register-pressure model (not Nsight).
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyEstimate {
    pub path: &'static str,
    pub threads_per_block: u32,
    pub blocks: u32,
    pub estimated_registers_per_thread: u32,
    pub theoretical_occupancy: f32,
    pub notes: &'static str,
}

/// Numerically stable softmax of a single routing row.
///
/// NaN logits are treated as `-inf` (probability 0 when any finite/`+inf`
/// value exists). `+inf` logits share remaining mass uniformly. Rows with
/// no finite maximum (all `-inf` / NaN) are uniform.
pub fn softmax_row(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let sanitized: Vec<f32> = logits
        .iter()
        .map(|&x| if x.is_nan() { f32::NEG_INFINITY } else { x })
        .collect();
    let logits = sanitized.as_slice();
    let n = logits.len();
    let n_pos_inf = logits
        .iter()
        .filter(|&&x| x.is_infinite() && x > 0.0)
        .count();
    if n_pos_inf > 0 {
        let p = 1.0 / n_pos_inf as f32;
        return logits
            .iter()
            .map(|&x| if x.is_infinite() && x > 0.0 { p } else { 0.0 })
            .collect();
    }
    let row_max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !row_max.is_finite() {
        return vec![1.0 / n as f32; n];
    }
    let mut out: Vec<f32> = logits.iter().map(|&x| (x - row_max).exp()).collect();
    let sum: f32 = out.iter().sum();
    let inv = 1.0 / sum.max(1e-8);
    for p in &mut out {
        *p *= inv;
    }
    out
}

/// Shannon entropy in bits: `H = -sum p log2(p)` for p > 1e-8.
pub fn entropy_row(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        let p = p.max(0.0);
        if p > 1e-8 {
            h += -p * p.log2();
        }
    }
    h
}

/// Per-node entropy plus the sum/max reduction used by the device kernels.
pub fn routing_entropy(probs: &[f32], n_nodes: usize, n_routes: usize) -> (f32, f32) {
    assert_eq!(probs.len(), n_nodes.saturating_mul(n_routes));
    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    for node in 0..n_nodes {
        let row = &probs[node * n_routes..node * n_routes + n_routes];
        let h = entropy_row(row);
        sum += h;
        if h > max {
            max = h;
        }
    }
    (sum, max)
}

/// Top-k route indices for one row (higher score, then lower index).
pub fn top_k_indices(scores: &[f32], top_k: usize) -> Vec<i32> {
    if top_k == 0 {
        return Vec::new();
    }
    let mut pairs: Vec<(f32, usize)> = scores.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    pairs.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut out = vec![-1i32; top_k];
    let n = pairs.len().min(top_k);
    for (slot, &(_, idx)) in pairs.iter().take(n).enumerate() {
        out[slot] = idx as i32;
    }
    out
}

/// SAAQ argmax: `score = membrane - scale * adaptation`, tie-break lower index.
pub fn saaq_best_walker(membrane: &[f32], adaptation: &[f32], scale: f32) -> u32 {
    assert_eq!(membrane.len(), adaptation.len());
    let mut best_score = SAAQ_SENTINEL;
    let mut best = 0u32;
    for (i, (&m, &a)) in membrane.iter().zip(adaptation.iter()).enumerate() {
        let mut score = m - scale * a;
        if score.is_nan() {
            score = SAAQ_SENTINEL;
        }
        if score > best_score || (score == best_score && (i as u32) < best) {
            best_score = score;
            best = i as u32;
        }
    }
    best
}

/// Inputs for the host fused / staged routing + SAAQ reference.
pub struct RoutingSaaqInput<'a> {
    pub scores: &'a [f32],
    pub membrane: &'a [f32],
    pub adaptation: &'a [f32],
    pub n_nodes: usize,
    pub n_routes: usize,
    pub top_k: usize,
    pub adaptation_scale: f32,
    pub scores_are_logits: bool,
}

/// Fused host path — same numerical result as staging softmax → entropy → SAAQ.
pub fn fused_routing_saaq(input: &RoutingSaaqInput<'_>) -> FusedRoutingSaaqResult {
    let n_nodes = input.n_nodes;
    let n_routes = input.n_routes;
    let top_k = input.top_k;
    assert_eq!(input.scores.len(), n_nodes.saturating_mul(n_routes));
    assert_eq!(input.membrane.len(), n_nodes);
    assert_eq!(input.adaptation.len(), n_nodes);
    assert!(
        top_k > 0 && top_k <= MAX_FUSED_TOP_K,
        "top_k must be in 1..=MAX_FUSED_TOP_K ({MAX_FUSED_TOP_K}), got {top_k}"
    );
    let k = top_k;
    let mut top_k_indices_out = vec![-1i32; n_nodes.saturating_mul(top_k)];
    let mut entropy_sum = 0.0f32;
    let mut entropy_max = 0.0f32;

    for node in 0..n_nodes {
        let row = &input.scores[node * n_routes..node * n_routes + n_routes];
        let probs = if input.scores_are_logits {
            softmax_row(row)
        } else {
            row.iter().map(|&p| p.max(0.0)).collect()
        };
        let h = entropy_row(&probs);
        entropy_sum += h;
        if h > entropy_max {
            entropy_max = h;
        }
        let idx = top_k_indices(&probs, k);
        for (t, &v) in idx.iter().enumerate() {
            if t < top_k {
                top_k_indices_out[node * top_k + t] = v;
            }
        }
    }

    FusedRoutingSaaqResult {
        top_k_indices: top_k_indices_out,
        entropy_sum,
        entropy_max,
        best_walker: saaq_best_walker(input.membrane, input.adaptation, input.adaptation_scale),
    }
}

fn n_blocks(n_nodes: usize) -> usize {
    n_nodes.div_ceil(FUSED_BLOCK_SIZE)
}

/// Unfused traffic: softmax write + entropy re-read + SAAQ two-pass.
///
/// Stages (all on-device, excluding the initial score/membrane upload):
/// 1. `routing_softmax` reads scores, writes the full probability matrix
/// 2. `routing_entropy_reduce_pass1` re-reads the matrix, writes partials
/// 3. `latent_reduce_pass2` reduces entropy partials
/// 4. `saaq_find_best_walker` reads membrane+adaptation, writes partials
/// 5. `saaq_reduce_partials_f16` writes the 4-byte walker
/// 6. host top-k re-reads the matrix (counted as a sixth pipeline stage, not a CUDA launch)
pub fn traffic_unfused(n_nodes: usize, n_routes: usize, top_k: usize) -> TrafficReport {
    if n_nodes == 0 {
        return TrafficReport {
            path: "unfused",
            n_nodes,
            n_routes,
            top_k,
            n_blocks: 0,
            bytes_read: 0,
            bytes_written: 0,
            launches: 0,
            materializes_routing_matrix: true,
        };
    }
    let n_blocks = n_blocks(n_nodes);
    let matrix = (n_nodes as u64)
        .saturating_mul(n_routes as u64)
        .saturating_mul(4);
    let activity = (n_nodes as u64).saturating_mul(8); // membrane + adaptation
    let entropy_partials = (n_blocks as u64).saturating_mul(8); // sum + max
    let saaq_partials = (n_blocks as u64).saturating_mul(8); // score + walker
    let topk_out = (n_nodes as u64)
        .saturating_mul(top_k as u64)
        .saturating_mul(4);

    let bytes_read = matrix // softmax read scores
        + matrix // entropy re-read probs
        + entropy_partials // pass2
        + activity // saaq pass1
        + saaq_partials // saaq pass2
        + matrix; // top-k re-read (unfused host/kernel)

    let bytes_written = matrix // softmax write probs
        + entropy_partials
        + 8 // entropy sum/max
        + saaq_partials
        + 4 // best walker
        + topk_out;

    TrafficReport {
        path: "unfused",
        n_nodes,
        n_routes,
        top_k,
        n_blocks,
        bytes_read,
        bytes_written,
        launches: 6,
        materializes_routing_matrix: true,
    }
}

/// Fused traffic: one logical pass over unique score + activity buffers.
/// Softmax still walks each row three times in registers/L1; those reloads
/// are not counted as extra DRAM traffic.
pub fn traffic_fused(n_nodes: usize, n_routes: usize, top_k: usize) -> TrafficReport {
    if n_nodes == 0 {
        return TrafficReport {
            path: "fused",
            n_nodes,
            n_routes,
            top_k,
            n_blocks: 0,
            bytes_read: 0,
            bytes_written: 0,
            launches: 0,
            materializes_routing_matrix: false,
        };
    }
    let n_blocks = n_blocks(n_nodes);
    let scores = (n_nodes as u64)
        .saturating_mul(n_routes as u64)
        .saturating_mul(4);
    let activity = (n_nodes as u64).saturating_mul(8);
    let partials = (n_blocks as u64).saturating_mul(16); // entropy sum/max + saaq score/walker
    let topk_out = (n_nodes as u64)
        .saturating_mul(top_k as u64)
        .saturating_mul(4);
    let telemetry = 12u64; // entropy sum + max + walker

    TrafficReport {
        path: "fused",
        n_nodes,
        n_routes,
        top_k,
        n_blocks,
        bytes_read: scores + activity + partials,
        bytes_written: topk_out + partials + telemetry,
        launches: 2,
        materializes_routing_matrix: false,
    }
}

/// Fraction of peak SM occupancy from a simple register/thread/block model.
pub fn theoretical_occupancy(threads_per_block: u32, regs_per_thread: u32, blocks: u32) -> f32 {
    if threads_per_block == 0 {
        return 0.0;
    }
    let thread_lim = SM120_MAX_THREADS_PER_SM / threads_per_block;
    let regs_per_block = regs_per_thread.saturating_mul(threads_per_block).max(1);
    let reg_lim = SM120_REGS_PER_SM / regs_per_block;
    let resident = thread_lim
        .min(reg_lim)
        .min(SM120_MAX_BLOCKS_PER_SM)
        .min(blocks.max(1));
    (resident.saturating_mul(threads_per_block)) as f32 / SM120_MAX_THREADS_PER_SM as f32
}

/// Occupancy estimates for the kernels compared in the GH #14 write-up.
///
/// Register counts are **estimates** from kernel complexity, not `ncu`
/// measurements. Local RTX 5080 Nsight is the occupancy quality gate.
pub fn occupancy_estimates(n_nodes: usize) -> Vec<OccupancyEstimate> {
    let blocks = n_blocks(n_nodes) as u32;
    vec![
        OccupancyEstimate {
            path: "routing_entropy_reduce_pass1",
            threads_per_block: FUSED_BLOCK_SIZE as u32,
            blocks,
            estimated_registers_per_thread: 32,
            theoretical_occupancy: theoretical_occupancy(256, 32, blocks),
            notes: "light reduce; high occupancy when enough blocks cover the SM",
        },
        OccupancyEstimate {
            path: "routing_saaq_fused_pass1",
            threads_per_block: FUSED_BLOCK_SIZE as u32,
            blocks,
            estimated_registers_per_thread: 56,
            theoretical_occupancy: theoretical_occupancy(256, 56, blocks),
            notes: "softmax + top-k array raises register pressure vs entropy-only",
        },
        OccupancyEstimate {
            path: "saaq_find_best_walker",
            threads_per_block: FUSED_BLOCK_SIZE as u32,
            blocks,
            estimated_registers_per_thread: 24,
            theoretical_occupancy: theoretical_occupancy(256, 24, blocks),
            notes: "8 blocks for 2048 neurons; occupancy limited by grid, not regs",
        },
        OccupancyEstimate {
            path: "saaq_select_fused",
            threads_per_block: FUSED_BLOCK_SIZE as u32,
            blocks: 1,
            estimated_registers_per_thread: 24,
            theoretical_occupancy: theoretical_occupancy(256, 24, 1),
            notes: "one block: occupancy 256/1536 ≈ 16.7% — fusion here does not help occupancy",
        },
    ]
}

/// Shapes used in the committed traffic artifact (SNN-ish and MoE-ish).
pub fn benchmark_shapes() -> &'static [(usize, usize, usize)] {
    &[
        (2048, 16, 4),  // 16-channel SNN / GIF hidden
        (256, 4096, 8), // MoE-style routing
        (1024, 128, 4),
        (64, 8, 2),
    ]
}

/// JSON (no serde) for the analytical traffic table.
pub fn traffic_model_json() -> String {
    let mut out = String::from(
        "{\n  \"model\": \"analytical-vram-bytes\",\n  \"target\": \"sm_120\",\n  \"rows\": [\n",
    );
    let shapes = benchmark_shapes();
    for (i, &(n, r, k)) in shapes.iter().enumerate() {
        let u = traffic_unfused(n, r, k);
        let f = traffic_fused(n, r, k);
        let saved = u.bytes_total().saturating_sub(f.bytes_total());
        let pct = if u.bytes_total() == 0 {
            0.0
        } else {
            100.0 * saved as f64 / u.bytes_total() as f64
        };
        out.push_str(&format!(
            "    {{\"n_nodes\":{n},\"n_routes\":{r},\"top_k\":{k},\
\"unfused_bytes\":{},\"fused_bytes\":{},\"bytes_saved\":{saved},\
\"pct_saved\":{pct:.2},\"unfused_launches\":{},\"fused_launches\":{}}}",
            u.bytes_total(),
            f.bytes_total(),
            u.launches,
            f.launches
        ));
        if i + 1 != shapes.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

/// CSV for the analytical traffic table.
pub fn traffic_model_csv() -> String {
    let mut csv = String::from(
        "path,n_nodes,n_routes,top_k,n_blocks,bytes_read,bytes_written,bytes_total,launches,materializes_matrix\n",
    );
    for &(n, r, k) in benchmark_shapes() {
        for report in [traffic_unfused(n, r, k), traffic_fused(n, r, k)] {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                report.path,
                report.n_nodes,
                report.n_routes,
                report.top_k,
                report.n_blocks,
                report.bytes_read,
                report.bytes_written,
                report.bytes_total(),
                report.launches,
                report.materializes_routing_matrix,
            ));
        }
    }
    csv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_pos_inf_takes_all_mass() {
        let p = softmax_row(&[1.0, f32::INFINITY, 2.0]);
        assert!((p[0] - 0.0).abs() < 1e-6);
        assert!((p[1] - 1.0).abs() < 1e-6);
        assert!((p[2] - 0.0).abs() < 1e-6);
        assert_eq!(top_k_indices(&p, 2), vec![1, 0]);
    }

    #[test]
    fn softmax_nan_logit_is_neg_inf() {
        let p = softmax_row(&[f32::NAN, 0.0]);
        assert!((p[0] - 0.0).abs() < 1e-6);
        assert!((p[1] - 1.0).abs() < 1e-6);
        assert_eq!(top_k_indices(&p, 2), vec![1, 0]);
    }

    #[test]
    fn softmax_two_pos_inf_share_mass() {
        let p = softmax_row(&[1.0, f32::INFINITY, f32::INFINITY]);
        assert!((p[0] - 0.0).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
        assert!((p[2] - 0.5).abs() < 1e-6);
        assert_eq!(top_k_indices(&p, 2), vec![1, 2]);
        assert!((entropy_row(&p) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_all_neg_inf_is_uniform() {
        let p = softmax_row(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn saaq_all_nan_scores_return_walker_zero() {
        let membrane = vec![1.0, 2.0, 3.0];
        let adaptation = vec![0.0, 0.0, 0.0];
        assert_eq!(saaq_best_walker(&membrane, &adaptation, f32::NAN), 0);
    }

    #[test]
    fn saaq_mixed_nan_adaptation_skips_invalid_walker() {
        let membrane = [1.0f32, 4.0, 2.0];
        let adaptation = [0.0, f32::NAN, 0.0];
        assert_eq!(saaq_best_walker(&membrane, &adaptation, 1.0), 2);
    }

    #[test]
    fn softmax_uniform_when_equal_logits() {
        let p = softmax_row(&[1.0, 1.0, 1.0, 1.0]);
        for v in &p {
            assert!((v - 0.25).abs() < 1e-6);
        }
        let h = entropy_row(&p);
        assert!((h - 2.0).abs() < 1e-5);
    }

    #[test]
    fn empty_traffic_is_zero_blocks_and_launches() {
        let u = traffic_unfused(0, 16, 4);
        let f = traffic_fused(0, 16, 4);
        assert_eq!(u.n_blocks, 0);
        assert_eq!(f.n_blocks, 0);
        assert_eq!(u.bytes_total(), 0);
        assert_eq!(f.bytes_total(), 0);
        assert_eq!(u.launches, 0);
        assert_eq!(f.launches, 0);
    }

    #[test]
    #[should_panic(expected = "top_k must be in 1..=MAX_FUSED_TOP_K")]
    fn fused_rejects_top_k_above_cap() {
        let _ = fused_routing_saaq(&RoutingSaaqInput {
            scores: &[0.0, 1.0],
            membrane: &[0.0],
            adaptation: &[0.0],
            n_nodes: 1,
            n_routes: 2,
            top_k: MAX_FUSED_TOP_K + 1,
            adaptation_scale: GIF_ADAPTATION_SCALE,
            scores_are_logits: true,
        });
    }

    #[test]
    fn saaq_empty_returns_walker_zero() {
        assert_eq!(saaq_best_walker(&[], &[], 0.22), 0);
    }

    #[test]
    fn saaq_selects_true_max_below_old_finite_sentinel() {
        let membrane = vec![-1e32, -1e31, -1e32];
        let adaptation = vec![0.0, 0.0, 0.0];
        assert_eq!(saaq_best_walker(&membrane, &adaptation, 0.0), 1);
    }

    #[test]
    fn fused_zero_routes_still_selects_saaq() {
        let membrane = [1.0f32, 4.0, 2.0];
        let adaptation = [0.0f32, 0.0, 0.0];
        let fused = fused_routing_saaq(&RoutingSaaqInput {
            scores: &[],
            membrane: &membrane,
            adaptation: &adaptation,
            n_nodes: 3,
            n_routes: 0,
            top_k: 2,
            adaptation_scale: GIF_ADAPTATION_SCALE,
            scores_are_logits: true,
        });
        assert_eq!(fused.best_walker, 1);
        assert_eq!(fused.entropy_sum, 0.0);
        assert_eq!(fused.entropy_max, 0.0);
        assert_eq!(fused.top_k_indices, vec![-1, -1, -1, -1, -1, -1]);
    }

    #[test]
    fn saaq_tie_breaks_to_lower_index() {
        let membrane = vec![3.0, 1.0, 3.0];
        let adaptation = vec![0.0, 0.0, 0.0];
        assert_eq!(saaq_best_walker(&membrane, &adaptation, 0.22), 0);
    }

    #[test]
    fn saaq_prefers_higher_score() {
        let membrane = vec![1.0, 4.0, 2.0];
        let adaptation = vec![0.0, 0.0, 0.0];
        assert_eq!(saaq_best_walker(&membrane, &adaptation, 0.22), 1);
    }

    #[test]
    fn saaq_adaptation_penalizes() {
        let membrane = vec![2.0, 2.0];
        let adaptation = vec![0.0, 10.0];
        assert_eq!(saaq_best_walker(&membrane, &adaptation, 0.22), 0);
    }

    #[test]
    fn fused_matches_staged_unfused() {
        let n_nodes = 8usize;
        let n_routes = 4usize;
        let top_k = 2usize;
        let mut scores = Vec::new();
        let mut membrane = Vec::new();
        let mut adaptation = Vec::new();
        for i in 0..n_nodes {
            membrane.push(i as f32 * 0.1);
            adaptation.push((i % 3) as f32);
            for r in 0..n_routes {
                scores.push((i as f32) * 0.01 - (r as f32) * 0.2);
            }
        }
        let fused = fused_routing_saaq(&RoutingSaaqInput {
            scores: &scores,
            membrane: &membrane,
            adaptation: &adaptation,
            n_nodes,
            n_routes,
            top_k,
            adaptation_scale: GIF_ADAPTATION_SCALE,
            scores_are_logits: true,
        });
        let mut probs = Vec::new();
        for node in 0..n_nodes {
            probs.extend(softmax_row(
                &scores[node * n_routes..node * n_routes + n_routes],
            ));
        }
        let (h_sum, h_max) = routing_entropy(&probs, n_nodes, n_routes);
        assert!((fused.entropy_sum - h_sum).abs() < 1e-5);
        assert!((fused.entropy_max - h_max).abs() < 1e-5);
        assert_eq!(
            fused.best_walker,
            saaq_best_walker(&membrane, &adaptation, GIF_ADAPTATION_SCALE)
        );
        for node in 0..n_nodes {
            let row = &probs[node * n_routes..node * n_routes + n_routes];
            let expect = top_k_indices(row, top_k);
            assert_eq!(
                &fused.top_k_indices[node * top_k..node * top_k + top_k],
                expect.as_slice()
            );
        }
    }

    #[test]
    fn fused_traffic_drops_the_routing_matrix() {
        let u = traffic_unfused(256, 4096, 8);
        let f = traffic_fused(256, 4096, 8);
        assert!(u.materializes_routing_matrix);
        assert!(!f.materializes_routing_matrix);
        assert!(f.bytes_total() < u.bytes_total());
        assert!(f.launches < u.launches);
        // The full Q×K matrix is 4 MiB; fused must not write it.
        assert!(f.bytes_written < 256 * 4096 * 4);
    }

    #[test]
    fn fused_traffic_small_grid_still_saves_re_read() {
        let u = traffic_unfused(2048, 16, 4);
        let f = traffic_fused(2048, 16, 4);
        assert!(f.bytes_total() < u.bytes_total());
    }

    #[test]
    fn saaq_fused_occupancy_is_one_block() {
        let occ = occupancy_estimates(2048);
        let fused = occ.iter().find(|o| o.path == "saaq_select_fused").unwrap();
        assert!((fused.theoretical_occupancy - (256.0 / 1536.0)).abs() < 1e-6);
        let unfused = occ
            .iter()
            .find(|o| o.path == "saaq_find_best_walker")
            .unwrap();
        assert!(unfused.theoretical_occupancy > fused.theoretical_occupancy);
    }

    #[test]
    fn traffic_json_and_csv_are_nonempty() {
        let json = traffic_model_json();
        assert!(json.contains("\"pct_saved\""));
        let csv = traffic_model_csv();
        assert!(csv.lines().count() > 4);
    }

    #[test]
    fn committed_traffic_artifacts_match_model() {
        let json = include_str!("../docs/fused_routing_saaq/traffic_model.json");
        assert_eq!(json, traffic_model_json());
        let csv = include_str!("../docs/fused_routing_saaq/traffic_model.csv");
        assert_eq!(csv, traffic_model_csv());
    }
}
