#!/usr/bin/env bash
set -euo pipefail

# Select the benchmark function and solver for the Table II workflow.
export BenchmarkFunction='ackley'  # options: ackley, branins, hartmann6, michal5, rosen3, rosen8
export Sampler_DIR='fmin'          # options: fmin, powell

PYTHONPATH=../src python3 -m efficient_sampling.cli run benchmark tableII "${BenchmarkFunction}" "${Sampler_DIR}" "$@"
