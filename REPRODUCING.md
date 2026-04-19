# Reproducing the Published Results

This repository was originally exported from the published Code Ocean capsule and then cleaned up for direct use as a normal Git repository.

## Option 1: run in a modern local Python environment

Install the repo and runtime dependencies:

```bash
pip install -e .[runtime]
```

List the available workflows:

```bash
efficient-sampling list
```

Run the default Table II example used by the archived capsule:

```bash
efficient-sampling run benchmark tableII ackley fmin
```

Run the EOS example:

```bash
efficient-sampling run eos Skyrme
```

Outputs are written under `results/`.

## Option 2: run with Docker

Build the archived environment from the repository root:

```bash
docker build -t efficient-sampling environment
```

Create an output directory:

```bash
mkdir -p results
```

Run the default capsule entrypoint:

```bash
docker run --rm \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  efficient-sampling ./run
```

The default `code/run` script launches the Table II benchmark workflow.

## Runtime dependencies

The modern local runtime is split into optional dependency groups in [pyproject.toml](/Users/42d/Efficient-Sampling/pyproject.toml):

- `.[runtime]`: benchmark and general scientific stack
- `.[runtime,eos]`: adds `torch` for the EOS workflow

The archived Dockerfile still reflects the old Code Ocean environment and Python 3.8-era packaging choices.

## Local LAMMPS build for the MD workflow

If you want to test the MD workflow locally, the repository provides:

```bash
bash scripts/install_lammps.sh
```

This performs an out-of-source CMake build with `PKG_KSPACE=on`, which is required by the `coul/long` plus `ewald` input deck in `code/md/ocp/in.ocp`.

After the build:

```bash
export LAMMPS_COMMAND="mpirun -np 4 $PWD/.local/lammps/bin/lmp"
efficient-sampling run md ocp
```

## Notes

- The workflows can take from minutes to days depending on the selected case and tolerance.
- The optimization and sampling procedures are stochastic, so exact trajectories may differ between runs.
- The MD workflow requires an external LAMMPS executable with the `KSPACE` package enabled.
- You can provide it with `LAMMPS_COMMAND="mpirun -np 4 /path/to/lmp"` or with `LAMMPS_MPIEXEC`, `LAMMPS_NP`, and `LAMMPS_EXECUTABLE`.
- The paper's Code availability statement points to the original Code Ocean capsule DOI: https://doi.org/10.24433/CO.1152070.v1
- The paper's Data availability statement points to Zenodo: https://doi.org/10.5281/zenodo.10908462
