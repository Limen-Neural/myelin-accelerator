#!/usr/bin/env python3
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Verify the pinned SNN fixtures under tests/fixtures/snn/ (LIM-1462 / #44).

Default (offline): re-check the checked-in bytes and derived integrity values
against each fixture.json. No network access.

--network: additionally re-download the pinned upstream artifacts and confirm
that checksums, Q8.8 decoding, weight orientation, Synfire NIR parameters and
the NIR-paper stimulus still match what is recorded. This is a manual
provenance audit; ordinary `cargo test` never runs it.

Stdlib only. `h5py` is optional and only used with --network to re-extract the
NIR graph parameters from model.nir; without it that single check is skipped
(the run then exits 2 as INCOMPLETE, never as passed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNN = ROOT / "tests" / "fixtures" / "snn"
SPIKENAUT = SNN / "spikenaut"
SYNFIRE = SNN / "synfire_lifneuron"
TIMEOUT = 30

failures: list[str] = []
skipped: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    """GET an https URL. Any other scheme (file:, ftp:, custom) is refused."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"refusing non-https URL: {url!r}")
    # Scheme validated above, so urlopen cannot be steered to file:/custom handlers.
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310  # nosec B310
        return resp.read()


def f32_bits(x: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', x))[0]:08x}"


Q88_WORD = re.compile(r"[0-9A-Fa-f]{4}")


def decode_q88(text: str) -> list[int]:
    """Strict decode, same rules as the Rust loader: LF-terminated lines of
    exactly four hex digits, no blank lines or other whitespace."""
    if not text.endswith("\n"):
        raise ValueError("Q8.8 .mem must end with a newline")
    codes = []
    for lineno, line in enumerate(text[:-1].split("\n"), start=1):
        if not Q88_WORD.fullmatch(line):
            raise ValueError(f"line {lineno}: expected 4 hex digits, got {line!r}")
        word = int(line, 16)
        codes.append(word - 0x10000 if word & 0x8000 else word)
    return codes


# ── offline ────────────────────────────────────────────────────────────────


def verify_spikenaut_offline() -> tuple[dict, bytes]:
    card = json.loads((SPIKENAUT / "fixture.json").read_text())
    raw = (SPIKENAUT / card["weights"]["file"]).read_bytes()
    src, w = card["source"], card["weights"]
    check(sha256(raw) == src["sha256"], "spikenaut: checked-in .mem sha256")
    check(len(raw) == src["size_bytes"], "spikenaut: checked-in .mem size")
    codes = decode_q88(raw.decode("ascii"))
    check(len(codes) == w["word_count"] == w["n_neurons"] * w["n_inputs"], "spikenaut: word count")
    check(sum(codes) == w["raw_code_sum"], "spikenaut: raw Q8.8 code sum")
    check(sum(1 for c in codes if c) == w["nonzero_count"], "spikenaut: nonzero count")
    check(min(codes) == w["min_code"] and max(codes) == w["max_code"], "spikenaut: code range")
    return card, raw


def verify_synfire_offline() -> dict:
    card = json.loads((SYNFIRE / "fixture.json").read_text())
    g = card["graph"]
    for key in ("affine_weight", "affine_bias", "lif_tau", "lif_v_threshold"):
        check(f32_bits(g[key]) == g[f"{key}_f32_bits"], f"synfire: {key} decimal matches recorded f32 bits")
    d0 = card["stimuli"][0]["d0"]
    check(len(d0) == card["stimuli"][0]["steps"] == 100 and set(d0) <= {0, 1}, "synfire: d0 shape")
    return card


# ── network ────────────────────────────────────────────────────────────────


def verify_spikenaut_network(card: dict, raw: bytes) -> None:
    src = card["source"]
    gh = f"https://raw.githubusercontent.com/rmems/Spikenaut-SNN/{src['commit']}/{src['path']}"
    hf = (
        f"https://huggingface.co/{src['huggingface_repo']}/resolve/"
        f"{src['huggingface_revision']}/{src['path']}"
    )
    for label, url in (("github", gh), ("huggingface", hf)):
        remote = fetch(url)
        check(sha256(remote) == src["sha256"], f"spikenaut: {label} sha256 at pinned revision")
        check(remote == raw, f"spikenaut: {label} bytes identical to checked-in fixture")

    model_url = (
        f"https://raw.githubusercontent.com/rmems/Spikenaut-SNN/{src['commit']}/"
        f"{src['companion_checkpoint']}"
    )
    model_raw = fetch(model_url)
    check(sha256(model_raw) == src["companion_checkpoint_sha256"], "spikenaut: snn_model.json sha256")
    neurons = json.loads(model_raw)["neurons"]
    codes = decode_q88(raw.decode("ascii"))
    n, k = card["weights"]["n_neurons"], card["weights"]["n_inputs"]
    row_major = sum(
        1 for i in range(n) for j in range(k) if round(neurons[i]["weights"][j] * 256) == codes[i * k + j]
    )
    col_major = sum(
        1 for i in range(n) for j in range(k) if round(neurons[i]["weights"][j] * 256) == codes[j * n + i]
    )
    print(f"     orientation: neuron-major {row_major}/{n * k}, input-major {col_major}/{n * k}")
    check(row_major == n * k, "spikenaut: neuron-major orientation (row-major [n_neurons x n_inputs])")


def _verify_synfire_release(src: dict) -> dict[str, bytes]:
    meta = json.loads(fetch(src["release_api"]))
    for key in ("id", "version", "license"):
        want = src[{"id": "release_id", "version": "release_version"}.get(key, key)]
        check(meta[key] == want, f"synfire: release {key}")
    remote_files = {f["name"]: f for f in meta["files"]}
    blobs: dict[str, bytes] = {}
    for rec in src["files"]:
        remote = remote_files.get(rec["name"], {})
        check(remote.get("sha256") == rec["sha256"], f"synfire: registry sha256 {rec['name']}")
        # A release file that cannot be downloaded is an audit failure, never a skip.
        check("downloadUrl" in remote, f"synfire: registry offers download for {rec['name']}")
        if "downloadUrl" not in remote:
            continue
        blob = fetch(remote["downloadUrl"])
        blobs[rec["name"]] = blob
        ok = sha256(blob) == rec["sha256"] and len(blob) == rec["size_bytes"]
        check(ok, f"synfire: downloaded {rec['name']}")
    return blobs


def _verify_synfire_upstream(card: dict) -> None:
    src = card["source"]
    base = f"https://raw.githubusercontent.com/neuromorphs/NIR/{src['upstream_commit']}"
    upstream = fetch(f"{base}/{src['upstream_path']}")
    check(sha256(upstream) == src["upstream_sha256"], "synfire: upstream NIR lif_norse.nir sha256")

    stim = card["stimuli"][0]
    m = re.search(r"commit ([0-9a-f]{40}), sha256 ([0-9a-f]{64})", stim["source"])
    check(m is not None, "synfire: d0 source records notebook commit + sha256")
    if m is None:
        return
    nb_raw = fetch(f"https://raw.githubusercontent.com/neuromorphs/NIR/{m.group(1)}/paper/01_lif/lif_norse.ipynb")
    check(sha256(nb_raw) == m.group(2), "synfire: lif_norse.ipynb sha256")
    cells = ["".join(c["source"]) for c in json.loads(nb_raw)["cells"] if c["cell_type"] == "code"]
    found = [json.loads(x.group(1)) for c in cells if (x := re.search(r"d0 = (\[[^\]]*\])", c))]
    check(found == [stim["d0"]], "synfire: d0 stimulus matches notebook")


def _verify_synfire_nir_params(g: dict, model_nir: bytes) -> None:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        what = "synfire: NIR parameter re-extraction (h5py not installed)"
        print(f"skip {what}")
        skipped.append(what)
        return
    datasets = {
        "affine_weight": "0/weight",
        "affine_bias": "0/bias",
        "lif_tau": "1/tau",
        "lif_r": "1/r",
        "lif_v_leak": "1/v_leak",
        "lif_v_threshold": "1/v_threshold",
    }
    with tempfile.NamedTemporaryFile(suffix=".nir") as tmp:
        tmp.write(model_nir)
        tmp.flush()
        with h5py.File(tmp.name, "r") as f:
            nodes = f["node/nodes"]
            check(f["version"][()].decode() == g["nir_version"], "synfire: NIR version")
            edges = [[a.decode(), b.decode()] for a, b in f["node/edges"][()]]
            check(edges == g["edges"], "synfire: graph edges")
            types = (nodes["0/type"][()].decode(), nodes["1/type"][()].decode())
            check(types == ("Affine", "LIF"), "synfire: node types")
            for key, path in datasets.items():
                val = float(nodes[path][()].reshape(-1)[0])
                check(f32_bits(val) == f32_bits(g[key]), f"synfire: {key} f32 bits from model.nir")
            check(("v_reset" in nodes["1"]) == (g["lif_v_reset"] is not None), "synfire: v_reset presence")


def verify_synfire_network(card: dict) -> None:
    blobs = _verify_synfire_release(card["source"])
    _verify_synfire_upstream(card)
    # Parameters are re-extracted from the *release* bytes only; no fallback
    # to the upstream copy, so a passing audit covers the pinned release.
    if "model.nir" in blobs:
        _verify_synfire_nir_params(card["graph"], blobs["model.nir"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", action="store_true", help="also re-download and audit pinned upstream artifacts")
    args = ap.parse_args()

    sp_card, sp_raw = verify_spikenaut_offline()
    sy_card = verify_synfire_offline()
    if args.network:
        verify_spikenaut_network(sp_card, sp_raw)
        verify_synfire_network(sy_card)

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    if skipped:
        # Never report a partial audit as a pass.
        print(f"\nINCOMPLETE: {len(skipped)} check(s) skipped: {skipped}", file=sys.stderr)
        return 2
    print("\nall SNN fixture checks passed" + (" (offline only)" if not args.network else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
