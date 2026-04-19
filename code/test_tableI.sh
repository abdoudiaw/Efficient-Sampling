#!/usr/bin/env bash
set -euo pipefail

# Select the benchmark function and sampler for the Table I workflow.
export BenchmarkFunction='easom'  # options: easom, hartmann6, michal2, rast, rosen, rosen8
export Sampler_DIR='looseopt'     # options: looseopt, loosernd, tightopt, tightrnd

PYTHONPATH=../src python3 -m efficient_sampling.cli run benchmark tableI "${BenchmarkFunction}" "${Sampler_DIR}" "$@"
