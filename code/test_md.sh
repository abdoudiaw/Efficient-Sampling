#!/usr/bin/env bash
set -euo pipefail

# Select the molecular dynamics example.
export PlasmaModel='ocp'   # options: ocp

PYTHONPATH=../src python3 -m efficient_sampling.cli run md "${PlasmaModel}" "$@"
