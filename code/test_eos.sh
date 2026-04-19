#!/usr/bin/env bash
set -euo pipefail

# Select the equation-of-state example.
export EosModel='Skyrme'   # options: Skyrme

PYTHONPATH=../src python3 -m efficient_sampling.cli run eos "${EosModel}" "$@"
