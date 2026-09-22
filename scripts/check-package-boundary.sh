#!/usr/bin/env bash
# Copyright 2026 Raul Montoya Cardenas
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

if grep -Ein '(^|/)(experiments|research)(/|$)|^examples/saaq/|^tests/fixtures/saaq/|^docs/saaq/|^(src|cu)/.*\.(json|csv|toml|ya?ml)$'; then
  echo 'experimental research must not be shipped in the crate archive' >&2
  exit 1
fi
