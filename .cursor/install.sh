#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Cursor Cloud Agent install script for myelin-accelerator (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Derived from .github/workflows (ci.yml, coverage.yml); installs only what CI needs:
#   - apt: build-essential, pkg-config, libssl-dev, python3 (release-prep unittest), curl, ca-certificates
#   - Rust stable (+rustfmt, clippy) [default] and 1.98.1 (coverage.yml)
#   - cargo fetch --locked
# The cuda feature and the self-hosted GPU job are skipped; CPU CI builds
# with --no-default-features (default = []).
#
# It ends with a dependency fetch, not a build or test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Intentionally unquoted wherever it is used: an empty $SUDO must expand to no
# word at all, so the command runs directly instead of failing on "".
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (C toolchain, libssl-dev as in ci.yml, python3 for the release-prep tests; curl for rustup) ---
apt_install build-essential pkg-config libssl-dev python3 curl ca-certificates

# --- Rust (rustup) ---
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v rustup >/dev/null 2>&1; then
  # Download the installer first: piping curl into sh can execute a truncated
  # installer when the download fails partway through.
  rustup_installer="$(mktemp)"
  trap 'rm -f "$rustup_installer"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --output "$rustup_installer" https://sh.rustup.rs
  sh "$rustup_installer" -y --default-toolchain none --profile minimal --no-modify-path
  rm -f "$rustup_installer"
  trap - EXIT
fi
# No rust-toolchain.toml here, so the default is the latest stable (ci.yml CPU jobs
# use stable too). coverage.yml pins 1.98.1.
rustup toolchain install stable --profile minimal --component rustfmt --component clippy
rustup toolchain install 1.98.1 --profile minimal
rustup default stable

# --- Prefetch crates (no build, no tests) ---
cargo fetch --locked

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get this directory from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
tool_dirs=("$HOME/.cargo/bin")
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH=%q:$PATH\n' "$(IFS=:; echo "${tool_dirs[*]}")" |
  $SUDO tee /etc/profile.d/cursor-env-myelin-accelerator.sh >/dev/null
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    name="${tool##*/}"
    dest="/usr/local/bin/$name"
    # Don't shadow a base-image command with the same name.
    if [ -f "$tool" ] && [ -x "$tool" ] && [ ! -e "$dest" ] && [ ! -L "$dest" ]; then
      $SUDO ln -s "$tool" "$dest"
    fi
  done
done

echo "Cursor install for myelin-accelerator finished."
