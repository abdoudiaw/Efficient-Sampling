#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${LAMMPS_EXTERNAL_DIR:-$ROOT_DIR/external}"
SRC_DIR="${LAMMPS_SRC_DIR:-$EXTERNAL_DIR/lammps}"
BUILD_DIR="${LAMMPS_BUILD_DIR:-$SRC_DIR/build}"
INSTALL_DIR="${LAMMPS_INSTALL_DIR:-$ROOT_DIR/.local/lammps}"
REPO_URL="${LAMMPS_REPO_URL:-https://github.com/lammps/lammps.git}"
GIT_REF="${LAMMPS_GIT_REF:-stable}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL="${LAMMPS_BUILD_PARALLEL:-4}"
BUILD_MPI="${LAMMPS_BUILD_MPI:-on}"
BUILD_OMP="${LAMMPS_BUILD_OMP:-off}"

mkdir -p "$EXTERNAL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")"

if [ ! -d "$SRC_DIR/.git" ]; then
  echo "Cloning LAMMPS into $SRC_DIR"
  git clone "$REPO_URL" "$SRC_DIR"
else
  echo "Using existing LAMMPS checkout at $SRC_DIR"
fi

cd "$SRC_DIR"
git fetch --tags origin

if git show-ref --verify --quiet "refs/heads/$GIT_REF"; then
  git checkout "$GIT_REF"
elif git show-ref --verify --quiet "refs/remotes/origin/$GIT_REF"; then
  git checkout -B "$GIT_REF" "origin/$GIT_REF"
else
  git checkout "$GIT_REF"
fi

if git rev-parse --verify --quiet "origin/$GIT_REF" >/dev/null; then
  git pull --ff-only origin "$GIT_REF"
fi

cmake -S cmake -B "$BUILD_DIR" \
  -D CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -D BUILD_MPI="$BUILD_MPI" \
  -D BUILD_OMP="$BUILD_OMP" \
  -D PKG_KSPACE=on

cmake --build "$BUILD_DIR" --parallel "$PARALLEL"
cmake --install "$BUILD_DIR"

LAMMPS_BIN="$INSTALL_DIR/bin/lmp"
if [ ! -x "$LAMMPS_BIN" ]; then
  echo "Expected LAMMPS executable not found at $LAMMPS_BIN" >&2
  exit 1
fi

cat <<EOF

LAMMPS build complete.

Executable:
  $LAMMPS_BIN

To run the MD workflow in this repository:
  export LAMMPS_COMMAND="mpirun -np 4 $LAMMPS_BIN"
  source "$ROOT_DIR/.venv-smoke/bin/activate"  # or your own environment
  efficient-sampling run md ocp

EOF
