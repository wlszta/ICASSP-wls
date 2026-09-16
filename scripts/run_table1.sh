#!/usr/bin/env bash
set -euo pipefail
config="${1:-configs/strict.yaml}"
shift || true
trainingless run-table1 --config "$config" --device "${DEVICE:-cuda}" "$@"
