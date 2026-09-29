#!/usr/bin/env bash
# =============================================================================
# Many generation shards inside ONE OAR job, several at a time.
#
#   oarsub ... -l /nodes=1/core=16 -S "./oarsub/run_generate_pool.sh <arm> <n_total> <per_shard> <first_shard> <n_shards>"
#
# The same work as an --array of run_generate.sh, packed for when the array
# cannot be submitted: GRICAD refuses a submission that would leave more than
# 50 jobs waiting, and an array of N is N jobs.  Each shard is the unchanged
# `python -m emu_pk.generate --mode emu --shard i`, at two threads, with
# (cores / 2) of them running at once; chunks skip if their output exists, so
# a besteffort kill or a walltime kill resumes exactly as an array element
# does.  One log per shard, beside the job's own.
# =============================================================================
set -euo pipefail
ARM="${1:?usage: run_generate_pool.sh <arm> <n_total> <per_shard> <first_shard> <n_shards>}"
N_TOTAL="${2:?n_total}"; PER="${3:?per_shard}"; FIRST="${4:?first_shard}"; N="${5:?n_shards}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
campaign_arm "${ARM}"
cd "${REPO}"
mkdir -p oarsub/logs "${EMU_PK_SHARDS_EMU}"
campaign_activate_env
NCORES="$(campaign_threads)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PAR=$(( NCORES / 2 > 0 ? NCORES / 2 : 1 ))
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  parallel=${PAR}" \
     " arm=${ARM} n_total=${N_TOTAL} per_shard=${PER} shards=${FIRST}..$(( FIRST + N - 1 ))" \
     " out=${EMU_PK_SHARDS_EMU}  start=$(date -Is)"
# shellcheck disable=SC2086
seq "${FIRST}" $(( FIRST + N - 1 )) | xargs -P "${PAR}" -I{} sh -c \
  "python -u -m emu_pk.generate --mode emu --shard {} --n-per-shard ${PER} \
     --n-total ${N_TOTAL} --out ${EMU_PK_SHARDS_EMU} ${EMU_PIN} \
     > oarsub/logs/pool_${OAR_JOB_ID:-local}_{}.log 2>&1 && echo 'shard {} done' || echo 'shard {} FAILED'"
echo "done=$(date -Is)"
