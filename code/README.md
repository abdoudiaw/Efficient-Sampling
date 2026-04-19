# Code Overview

This directory contains the research workflows used in the paper.

## Structure

- `common/benchmark/`: shared benchmark runtime modules
- `tableI/` and `tableII/`: case-specific benchmark configurations and convergence plots
- `eos/`: dense nuclear matter equation-of-state workflow
- `md/`: molecular dynamics workflow
- `run`: default archived capsule entrypoint

The benchmark case directories now keep only the files that are actually case-specific, primarily `_model.py` and small helper modules such as `hartmann.py`.

## Running workflows

From the repository root:

```bash
efficient-sampling run benchmark tableII ackley fmin
efficient-sampling run eos Skyrme
```

The shell launchers in this directory call the same CLI and are kept for backward compatibility with the original repo layout.

## Common outputs

Generated outputs usually include:

- `cost.pkl`: serialized objective function
- `hist.pkl`: test-misfit history
- `size.pkl`: endpoint-cache history
- `score.txt`: score versus model-evaluation history
- `stop`: optimizer endpoint database
- `func.db`: learned surrogate database
- `eval`: objective-evaluation database

Pickle artifacts can be inspected with `dill`, for example:

```python
import dill

with open("cost.pkl", "rb") as fh:
    cost = dill.load(fh)
```

## MD note

The MD workflow shells out to an external LAMMPS executable. It needs a build with the `KSPACE` package enabled because the input deck uses `pair_style coul/long` with `kspace_style ewald`.

Set one of:

- `LAMMPS_COMMAND="mpirun -np 4 /path/to/lmp"`
- `LAMMPS_MPIEXEC`, `LAMMPS_NP`, and `LAMMPS_EXECUTABLE`
