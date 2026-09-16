#!/usr/bin/env bash
set -euo pipefail
trainingless run-table1 --config "${1:-configs/strict.yaml}" --device "${DEVICE:-cuda}" "$@"
