#!/usr/bin/env bash
set -e
echo "running sweep"
echo "model 1 running"
python train.py --config conf/linear_sweep/linear1.yaml
echo "model 2 running"
python train.py --config conf/linear_sweep/linear2.yaml
echo "model 3 running"
python train.py --config conf/linear_sweep/linear3.yaml
echo "model 4 running"
python train.py --config conf/linear_sweep/linear4.yaml
