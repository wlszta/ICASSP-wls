#!/usr/bin/env bash
set -euo pipefail
trainingless run-figure7 --config "${1:-configs/strict.yaml}" --device "${DEVICE:-cuda}" "$@"
