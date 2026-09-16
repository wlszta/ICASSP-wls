#!/usr/bin/env bash
set -euo pipefail
trainingless run-sensitivity --config "${1:-configs/strict.yaml}" --device "${DEVICE:-cuda}" "$@"
