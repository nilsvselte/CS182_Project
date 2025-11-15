#!/usr/bin/env bash
set -e
echo "running sweep"
echo "model mixing running"
uv run python train.py --config conf/dual_sweep/dual_mixed.yaml
echo "model sequential running"
uv run python train.py --config conf/dual_sweep/dual_sequential.yaml
echo "model mixed running"
uv run python train.py --config conf/dual_sweep/dual_mixed.yaml