use super::super::*;

#[test]
fn traffic_counts_composable_path() {
    use analytical::composable;
    // 256 nodes, 4096 routes, k=8: M = 4 MiB, one block.
    let m = 4_194_304u64;
    let c = composable(256, 4096, 8);
    assert_eq!(c.bytes_read, 5 * m + 2048 + 16);
    assert_eq!(c.bytes_written, m + 8192 + 16 + 16);
    assert_eq!(c.launches, 6);
    assert_eq!(composable(0, 16, 4).bytes_total(), 0);
}

#[test]
fn traffic_counts_fused_path() {
    use analytical::fused;
    let m = 4_194_304u64;
    let f = fused(256, 4096, 8);
    assert_eq!(f.bytes_read, 2 * m + 2048 + 16);
    assert_eq!(f.bytes_written, 8192 + 16 + 16);
    assert_eq!(f.launches, 2);
    assert_eq!(fused(0, 16, 4).launches, 0);
}
