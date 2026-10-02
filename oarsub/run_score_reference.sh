#!/usr/bin/env bash
# =============================================================================
# Score networks against validate's reference truth -- CAMB at its converged
# rung times CLASS's heating -- from a cache prefill_reference.py has filled.
#
#   oarsub ... -l /nodes=1/core=4,walltime=02:00:00 \
#       -S "./oarsub/run_score_reference.sh <cache_dir> <weights.npz|shipped> [...]"
#
# Shape and tails only: those are the bands the prefill solves (validate's
# headline band and one tail per network, 10 h/Mpc to its k_max).  The floor,
# low-k, flat-slice, derivative and convergence scores solve truth at points
# the cache does not hold -- minutes each at the reference precision -- and stay
# on the training truth run_train.sh scores against.  With the cache full this
# job solves nothing; it reads.
#
# `shipped` scores the weights the package ships (2.0), which is the footing on
# which 2.0 and 2.1 are compared.  Each network writes
# `<weights>.reference.validation.json` beside itself; the shipped one writes
# `emu_pk_mlp_shipped.reference.validation.json` in EMU_PK_WORK.
# =============================================================================
set -euo pipefail
CACHE="${1:?usage: run_score_reference.sh <cache_dir> <weights.npz|shipped> [...]}"
shift 1
[ $# -gt 0 ] || { echo "no network to score" >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
cd "${REPO}"
mkdir -p oarsub/logs
# The reference truth is CAMB >= 1.6: the generation environment, not training's.
campaign_activate_env gen
NCORES="$(campaign_threads)"
export OMP_NUM_THREADS="${NCORES}"
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  cache=${CACHE}" \
     " networks=$*  start=$(date -Is)"
status=0
for W in "$@"; do
    if [ "${W}" = "shipped" ]; then
        WARG=(); OUT="${EMU_PK_WORK}/emu_pk_mlp_shipped.reference.validation.json"
    else
        WARG=(--weights "${W}"); OUT="${W%.npz}.reference.validation.json"
    fi
    echo "-- ${W} -> ${OUT}"
    python -u -m emu_pk.validate "${WARG[@]}" --truth reference --cache "${CACHE}" \
        --no-floor --no-lowk --no-flat-slice --no-deriv --no-convergence \
        --json "${OUT}" || { echo "!! scoring ${W} failed"; status=1; }
done
echo "done=$(date -Is)"
exit "${status}"
