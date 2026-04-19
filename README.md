[![CI](https://github.com/abdoudiaw/Efficient-Sampling/actions/workflows/ci.yml/badge.svg)](https://github.com/abdoudiaw/Efficient-Sampling/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/github/license/abdoudiaw/Efficient-Sampling)](https://github.com/abdoudiaw/Efficient-Sampling/blob/main/LICENSE)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1038%2Fs42256--024--00839--1-blue)](https://doi.org/10.1038/s42256-024-00839-1)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10908462.svg)](https://doi.org/10.5281/zenodo.10908462)

# Efficient Learning of Accurate Surrogates for Simulations of Complex Systems

This repository contains the research code accompanying the Nature Machine Intelligence paper "Efficient learning of accurate surrogates for simulations of complex systems" by A. Diaw, M. McKerns, I. Sagert, L. G. Stanton, and M. S. Murillo, published on May 17, 2024.

Paper: https://doi.org/10.1038/s42256-024-00839-1

The code implements an online learning workflow for surrogate construction using optimizer-driven sampling. It includes the benchmark studies reported in Tables I and II, the dense nuclear matter equation-of-state example, and the molecular dynamics example used in the paper.

## Repository layout

- `src/efficient_sampling/`: modern CLI and repo-local staging utilities
- `code/common/benchmark/`: shared runtime modules for the benchmark workflows
- `code/tableI/` and `code/tableII/`: case-specific benchmark configurations
- `code/eos/`: dense nuclear matter equation-of-state workflow
- `code/md/`: molecular dynamics workflow
- `environment/`: archived Docker environment from the Code Ocean capsule export
- `metadata/`: capsule metadata

## Install

For the lightweight CLI and staging tools:

```bash
pip install -e .
```

For local workflow execution without Docker:

```bash
pip install -e .[runtime]
```

For the EOS neural-network workflow as well:

```bash
pip install -e .[runtime,eos]
```

The optional dependency split keeps CI and lightweight repo operations fast while still supporting full local execution when needed.

## Quick start

List the available workflows:

```bash
efficient-sampling list
```

Run a benchmark case:

```bash
efficient-sampling run benchmark tableII ackley fmin
```

Run the EOS example:

```bash
efficient-sampling run eos Skyrme
```

The existing shell launchers in `code/` are still available and now call the same CLI under the hood:

- `bash code/test_tableI.sh`
- `bash code/test_tableII.sh`
- `bash code/test_eos.sh`
- `bash code/test_md.sh`

The CLI stages each case into `results/` and executes `main_workflow.py` there, preserving the original working-directory assumptions from the paper code.

## Workflow notes

- The benchmark runtime was consolidated into shared modules under `code/common/benchmark/`; the benchmark case directories now contain only their case-specific configuration.
- The Table I, Table II, and EOS examples are self-contained in this repository.
- The MD example requires an external LAMMPS build with the `KSPACE` package enabled, since `code/md/ocp/in.ocp` uses `pair_style coul/long` and `kspace_style ewald`.
- Configure the MD launcher with either `LAMMPS_COMMAND="mpirun -np 4 /path/to/lmp"` or the split variables `LAMMPS_MPIEXEC`, `LAMMPS_NP`, and `LAMMPS_EXECUTABLE`.
- Results are not strictly bitwise reproducible because the sampling and optimization workflows contain randomized components, but they should follow the same qualitative behavior reported in the paper.

## Published artifacts

- Paper DOI: https://doi.org/10.1038/s42256-024-00839-1
- Data on Zenodo: https://doi.org/10.5281/zenodo.10908462
- Original Code Ocean capsule: https://doi.org/10.24433/CO.1152070.v1

## Development

Basic validation:

```bash
python3 -m unittest discover -s tests
bash -n code/run code/test_tableI.sh code/test_tableII.sh code/test_eos.sh code/test_md.sh test_md.sh
```

CI is defined in [.github/workflows/ci.yml](/Users/42d/Efficient-Sampling/.github/workflows/ci.yml). Release notes and the public-release checklist are in [docs/RELEASE.md](/Users/42d/Efficient-Sampling/docs/RELEASE.md).

## Local LAMMPS Install

For the MD workflow, this repo includes a helper installer:

```bash
bash scripts/install_lammps.sh
```

By default it clones LAMMPS into `external/lammps`, builds it with CMake, enables `PKG_KSPACE`, and installs it under `.local/lammps`. You can override locations and build settings with environment variables such as `LAMMPS_GIT_REF`, `LAMMPS_INSTALL_DIR`, and `LAMMPS_BUILD_PARALLEL`.

## License

The project is distributed under the BSD 3-Clause License. See [LICENSE](/Users/42d/Efficient-Sampling/LICENSE).
