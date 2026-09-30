#!/usr/bin/env bash
# =============================================================================
# One slice of validate's reference truth, solved into its cache.
#
#   oarsub ... -l /nodes=1/core=4,walltime=04:00:00 \
#       -S "./oarsub/run_prefill_reference.sh <cache_dir> <point> [point ...]"
#
# See prefill_reference.py.  Generation environment (CAMB >= 1.6, CLASS).
# =============================================================================
set -euo pipefail
CACHE="${1:?cache dir}"; shift 1
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
cd "${REPO}"
mkdir -p oarsub/logs "${CACHE}"
campaign_activate_env
NCORES="$(campaign_threads)"
export OMP_NUM_THREADS="${NCORES}"
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  points=$*  start=$(date -Is)"
python -u oarsub/prefill_reference.py --cache "${CACHE}" --points "$@"
echo "done=$(date -Is)"
