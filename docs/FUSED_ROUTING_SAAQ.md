# Fused routing and SAAQ kernels (GH #14)

Reusable device primitives that combine **routing softmax**, **entropy
reduction**, **top-k selection**, and **SAAQ walker selection** so the full
probability matrix never has to hit VRAM.

Higher-level experiment loops stay in `corinth-canal`. This crate only ships
the kernels, host references, and the traffic model.

Related: [GH #14](https://github.com/Limen-Neural/myelin-accelerator/issues/14) /
[LIM-891](https://linear.app/rpd-34/issue/LIM-891). GIF/`TemporalState` / myelin_shim
porting remains [GH #27](https://github.com/Limen-Neural/myelin-accelerator/issues/27).

| Layer | Location |
|-------|----------|
| Unfused SAAQ pass 1 / 2 | `cu/spiking_network.cu` (`saaq_find_best_walker`, `saaq_reduce_partials_f16`) |
| Fused kernels | `cu/fused_routing_saaq.cu` |
| Host reference + traffic model | `src/fused.rs` |
| Launch wrappers | `GpuAccelerator::{routing_softmax, routing_entropy_reduce, saaq_select, saaq_select_fused, routing_saaq_fused}` |
| Artifacts | [`docs/fused_routing_saaq/traffic_model.json`](fused_routing_saaq/traffic_model.json), [`.csv`](fused_routing_saaq/traffic_model.csv) |

---

## Research questions

| Question | Result |
|----------|--------|
| Can routing and SAAQ selection be fused into a single pass? | **Yes.** `routing_saaq_fused_pass1` softmaxes each node row, inserts top-k, and scores SAAQ in the same loop. |
| Can entropy be accumulated during routing? | **Yes.** Shannon entropy is reduced in-register per node, then warp/block-reduced to on-device partials. |
| Can telemetry stay on-device until final reduction? | **Yes.** Pass 2 writes 12 bytes (`entropy_sum`, `entropy_max`, `best_walker`). The probability matrix is never stored. |
| Does fusion help latency / bandwidth / occupancy on Blackwell? | **Bandwidth: yes** (see table). **SAAQ-only occupancy: no** (one-block fused argmax is 12.5% theoretical occupancy). **Latency:** GPU numbers need an RTX 5080; host and analytical bytes are below. |

---

## Unfused vs fused pipelines

Unfused (matches the corinth-canal tick tail, minus GIF itself):

1. `routing_softmax` — write `[n_nodes × n_routes]` probabilities
2. `routing_entropy_reduce_pass1` — **re-read** that matrix
3. `latent_reduce_pass2` — 8-byte entropy telemetry
4. `saaq_find_best_walker` — read membrane + adaptation
5. `saaq_reduce_partials_f16` — 4-byte walker
6. A separate top-k pass would re-read the matrix again

Fused:

1. `routing_saaq_fused_pass1` — read scores + activity once; write top-k + small partials
2. `fused_telemetry_reduce_pass2` — 12-byte telemetry

VRAM traffic is counted in `src/fused.rs` (`traffic_unfused` / `traffic_fused`).
Those functions are the source of the committed JSON/CSV.

### Where fusion helps

- **Skipping the routing matrix.** For a 256×4096 MoE-style grid the unfused path
  writes and re-reads a 4 MiB probability matrix. Fusion drops that entirely.
- **Launch count.** 6 launches → 2 for the combined routing+SAAQ path.
- **Entropy during softmax.** No second pass over `p log p`.

### Where fusion does not help

- **SAAQ pass1 + pass2 alone.** Partials are 8 floats/uints per block (64 bytes
  at 8 blocks). Fusing them into `saaq_select_fused` (<<<1, 256>>>) saves a
  launch but **cuts occupancy to one block** (~12.5% of an SM). Prefer the
  two-pass SAAQ kernels when the grid is already 8×256 for 2048 neurons.
- **GIF `gif_step_weighted` itself.** That kernel’s traffic is the weight
  matrix, not routing logits. Fusing GIF into this pass is a different
  prototype and belongs with the GH #27 port (`TemporalState`).
- **Register pressure.** Fused pass1 keeps a top-k array (`MAX_FUSED_TOP_K=8`)
  plus softmax state. Estimated occupancy is slightly lower than
  entropy-only pass1 when the grid is large enough to fill the SM.

---

## Analytical traffic (committed)

Source: `myelin_accelerator::fused::{traffic_unfused, traffic_fused}`.
Regenerate with `cargo test --locked fused::tests::committed_traffic_artifacts_match_model`.

| n_nodes | n_routes | top_k | Unfused bytes | Fused bytes | Saved |
|---------|----------|-------|---------------|-------------|-------|
| 2048 | 16 | 4 | 573708 | 180492 | 68.54% |
| 256 | 4096 | 8 | 16787500 | 4204588 | 74.95% |
| 1024 | 128 | 4 | 2121868 | 549004 | 74.13% |
| 64 | 8 | 2 | 9260 | 3116 | 66.35% (launch overhead dominates) |

Exact figures live in the JSON/CSV so they cannot drift from the model.

Host wall-clock in `examples/benchmark.rs` is **not** the fusion quality
gate (the CPU reference allocates per-row softmax buffers). The device
question is VRAM bytes and kernel launches; GPU latency needs
`--features bench,cuda` on an RTX 5080.

### Occupancy model

`occupancy_estimates` uses sm_120-class SM limits (2048 threads, 65536 registers,
32 blocks) and **estimated** register counts, not Nsight Compute. On an RTX
5080, re-measure with:

```bash
ncu --set full -o fused_saaq \
  cargo run --example benchmark --profile bench --features bench,cuda
```

Compare `routing_entropy_reduce_pass1` vs `routing_saaq_fused_pass1`, and
`saaq_find_best_walker` vs `saaq_select_fused`.

---

## API

Host (always available):

```rust
use myelin_accelerator::fused::{
    RoutingSaaqInput, fused_routing_saaq, saaq_best_walker, traffic_fused, traffic_unfused,
};
```

Device (`cuda` feature), telemetry stays on the GPU until `to_vec` of the
three scalar buffers:

```rust
acc.routing_saaq_fused(
    &scores, &membrane, &adaptation,
    &mut top_k, &mut entropy_sum, &mut entropy_max, &mut best_walker,
    n_nodes, n_routes, top_k, 0.22, true,
)?;
```

`top_k` must be in `1..=MAX_FUSED_TOP_K` (8). `scores_are_logits = true` runs
a stable softmax; `false` treats rows as already-normalized probabilities.

---

## What this is not

- Not GIF/`TemporalState` (GH #27).
- Not SAAQ experiment recipes (corinth-canal).
- Not a replacement for `cosine_similarity_top_k` on high-dimensional keys;
  that kernel still materializes Q×K similarities. The fused kernel is for
  **already-scored routing rows** (logits or probabilities).
