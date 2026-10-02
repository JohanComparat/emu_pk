#!/usr/bin/env bash
# =============================================================================
# Score trained weights against the truth they learnt, on a machine with CAMB.
#
#   oarsub ... -l /nodes=1/core=16,walltime=12:00:00 \
#       -S "./oarsub/run_validate.sh <weights.npz> [validate args ...]"
#
# What run_train.sh does after training when the node has the solvers.  A GPU
# cluster's training environment does not (Bigfoot, Kraken): run_train.sh then
# prints this command, and the weights -- on /bettik, shared with Dahu -- are
# scored here.  Writes `<weights>.validation.json` beside them, the file every
# other number about the network is read from.  The full default suite took 7 h
# for the v2.1 pilot on 16 cores, almost all of it truth solves.
# =============================================================================
set -euo pipefail
W="${1:?usage: run_validate.sh <weights.npz> [validate args ...]}"
shift 1
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
cd "${REPO}"
mkdir -p oarsub/logs
[ -f "${W}" ] || { echo "no weights at ${W}" >&2; exit 2; }
campaign_activate_env gen
NCORES="$(campaign_threads)"
export OMP_NUM_THREADS="${NCORES}"
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  weights=${W}" \
     " args='$*'  start=$(date -Is)"
python -u -m emu_pk.validate --weights "${W}" --json "${W%.npz}.validation.json" "$@"
echo "done=$(date -Is)"
