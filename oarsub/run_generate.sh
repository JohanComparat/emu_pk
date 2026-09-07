#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR array worker: one shard of CLASS solves.
#
#   $1  mode   "ratio" (Phase 1 correction grid) or "emu" (Phase 2 training set)
#   $2  arm    "c" (the nine-parameter design) or "f" (its flat control, with
#              Omega_k pinned to zero).  Ignored by "ratio", which is a fiducial
#              sweep and is not curved.
#   $3  n_total     size of the emu design
#   $4  per_shard   cosmologies per array element
#
# $3 and $4 are arguments and not environment variables for the same reason $1
# and $2 are, and the consequence here is the worst of the three.  The submitter
# reads EMU_N_TOTAL on the frontend to size the array; the node re-reads it from
# `_campaign_env.sh` and gets the default.  So `EMU_N_TOTAL=16000
# submit_campaign.sh emu` submitted sixteen elements that each generated from
# the 150000-point design -- a design nobody asked for, in files named as though
# they were the one that was.  Nothing downstream could see it: the parameter
# names match, the grids match, and only the *values* are from another design.
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

MODE="${1:?usage: run_generate.sh <ratio|emu> [arm] [n_total] [per_shard]}"
ARM="${2:-c}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
# Fall back to the campaign defaults only when the caller named nothing.
EMU_N_TOTAL="${3:-${EMU_N_TOTAL}}"
EMU_PER_SHARD="${4:-${EMU_PER_SHARD}}"
# An argument, not an environment variable: OAR does not carry the submitting
# shell's environment to the node, so an arm passed that way would run the
# default one -- into the default one's directory, on top of its shards.
campaign_arm "${ARM}"

cd "${REPO}"
mkdir -p oarsub/logs
campaign_activate_env
NCORES="$(campaign_threads)"

# OAR_ARRAY_INDEX is 1-based; the generator's shards are 0-based.
SHARD=$(( ${OAR_ARRAY_INDEX:-1} - 1 ))

echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  array=${OAR_ARRAY_INDEX:-1}" \
     " shard=${SHARD}  cores=${NCORES}  mode=${MODE}  arm=${EMU_PK_ARM}" \
     " n_total=${EMU_N_TOTAL}  per_shard=${EMU_PER_SHARD}" \
     " pin='${EMU_PIN}'  start=$(date -Is)"

case "${MODE}" in
  ratio)
    mkdir -p "${EMU_PK_SHARDS_RATIO}"
    python -u -m emu_pk.generate --mode ratio \
        --shard "${SHARD}" --n-per-shard "${RATIO_PER_SHARD}" \
        --out "${EMU_PK_SHARDS_RATIO}"
    ;;
  emu)
    mkdir -p "${EMU_PK_SHARDS_EMU}"
    # ${EMU_PIN} is deliberately unquoted: it is either empty or the two words
    # `--pin Omega_k=0`, and quoting it would pass an empty argument.
    # shellcheck disable=SC2086
    python -u -m emu_pk.generate --mode emu \
        --shard "${SHARD}" --n-per-shard "${EMU_PER_SHARD}" \
        --n-total "${EMU_N_TOTAL}" --out "${EMU_PK_SHARDS_EMU}" ${EMU_PIN}
    ;;
  *) echo "unknown mode '${MODE}' (ratio|emu)" >&2; exit 2 ;;
esac

echo "done=$(date -Is)"
