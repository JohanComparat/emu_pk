#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR array worker: one shard of CLASS solves.
#
#   $1  mode   "ratio" (Phase 1 correction grid) or "emu" (Phase 2 training set)
#
# The shard index comes from $OAR_ARRAY_INDEX, which OAR sets per array element
# and numbers from 1; the generator numbers shards from 0, so it is decremented
# here.  Running outside OAR uses shard 0, which is what makes this script
# testable on a login node.
#
# Arguments, not environment: OAR does not propagate the submitting shell's
# environment to the node, so `MODE=emu oarsub -S ./run_generate.sh` would
# silently run the default.
#
# Submit through submit_campaign.sh; by hand it is
#   oarsub --project "${EMU_PK_PROJECT}" -l /nodes=1/core=2,walltime=06:00:00 \
#          --array 94 -S "./oarsub/run_generate.sh emu"
# =============================================================================

# No `#OAR --project` directive: the allocation is site configuration, not
# repository content, and `submit_campaign.sh` passes it on the command line
# where it belongs.  See oarsub/site.sh.example.
#OAR --name emupk_gen
#OAR -l /nodes=1/core=2,walltime=06:00:00
#OAR --stdout oarsub/logs/%jobid%.emupk_gen.out
#OAR --stderr oarsub/logs/%jobid%.emupk_gen.err

set -euo pipefail

MODE="${1:?usage: run_generate.sh <ratio|emu>}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"

cd "${REPO}"
mkdir -p oarsub/logs
campaign_activate_env
NCORES="$(campaign_threads)"

# OAR_ARRAY_INDEX is 1-based; the generator's shards are 0-based.
SHARD=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))

echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  array=${OAR_ARRAY_INDEX:-1}" \
     " shard=${SHARD}  cores=${NCORES}  mode=${MODE}  start=$(date -Is)"

case "${MODE}" in
  ratio)
    mkdir -p "${EMU_PK_SHARDS_RATIO}"
    python -u -m emu_pk.generate --mode ratio \
        --shard "${SHARD}" --n-per-shard "${RATIO_PER_SHARD}" \
        --out "${EMU_PK_SHARDS_RATIO}"
    ;;
  emu)
    mkdir -p "${EMU_PK_SHARDS_EMU}"
    python -u -m emu_pk.generate --mode emu \
        --shard "${SHARD}" --n-per-shard "${EMU_PER_SHARD}" \
        --n-total "${EMU_N_TOTAL}" --out "${EMU_PK_SHARDS_EMU}"
    ;;
  *) echo "unknown mode '${MODE}' (ratio|emu)" >&2; exit 2 ;;
esac

echo "done=$(date -Is)"
