# AGENTS.md

## Project
This is a Rust + CUDA kernel crate for Blackwell / RTX 5080-class GPU work.

## Build and test
Run CPU-safe checks first:

```bash
cargo test --locked
cargo test --locked --features saaq
cargo build --locked --no-default-features
```

Local GPU path (after CPU checks): toolkit **13.2+** OK; ShipOfTheseus
default is **13.3** via `/usr/local/cuda`. Full local quality gate is in
`REVIEW.md` §6–§7 and `CLAUDE.md` (not a cloud-only check).
