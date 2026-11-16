#!/usr/bin/env bash
# The architecture scan, one run after the other on one GPU (scan.py skips a run
# whose JSON exists, so the script resumes where it stopped).  First the
# architectures on the full training pool, then the data scaling.
#   OUT=~/data/emu_pk_architecture_scan bash studies/architecture/run_scan.sh
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/software/miniforge3/envs/jaxgpu/bin/python}
OUT=${OUT:-$HOME/data/emu_pk_architecture_scan}
STEPS=${STEPS:-200000}
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest
mkdir -p "$OUT"

run() {  # n_cos hidden...
  local n=$1; shift
  "$PY" studies/architecture/scan.py --n-cos "$n" --hidden "$@" --steps "$STEPS" --out "$OUT" \
    || echo "FAILED: n=$n hidden=$*"
}

FULL=145475
for h in "512 512 512 512" "512 512" "1024" "256 256 256 256" "1024 1024 1024 1024" \
         "512 512 512 512 512 512" "1024 1024" "512" "256 256" "256" \
         "1024 1024 1024 1024 1024 1024" "256 256 256 256 256 256"; do
  run $FULL $h
done
for n in 36000 9000 2250; do
  for h in "512 512 512 512" "512 512" "1024" "1024 1024 1024 1024" "256 256 256 256" \
           "512 512 512 512 512 512"; do
    run $n $h
  done
done
echo "scan finished $(date)"
