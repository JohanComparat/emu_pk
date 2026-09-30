#!/usr/bin/env bash
# =============================================================================
# One generation chunk -- 50 solves -- as a short OAR job.
#
#   oarsub ... -l /nodes=1/core=2,walltime=03:00:00 \
#       -S "./oarsub/run_generate_chunk.sh <arm> <n_total> <per_shard> <chunk_start>"
#
# The unit a busy cluster schedules fastest: small and short jobs backfill
# around the large ones, where a whole node waits.  The chunk is named exactly
# as the whole shard would name it (`generate --chunk-start`), so short jobs,
# pooled node jobs and array elements that meet on the same points write the
# same file and skip each other's work.  Fed by `feed_chunks.py`.
# =============================================================================
set -euo pipefail
ARM="${1:?usage: run_generate_chunk.sh <arm> <n_total> <per_shard> <chunk_start>}"
N_TOTAL="${2:?n_total}"; PER="${3:?per_shard}"; C0="${4:?chunk_start}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
campaign_arm "${ARM}"
cd "${REPO}"
mkdir -p oarsub/logs "${EMU_PK_SHARDS_EMU}"
campaign_activate_env
NCORES="$(campaign_threads)"
export OMP_NUM_THREADS="${NCORES}" OPENBLAS_NUM_THREADS="${NCORES}" MKL_NUM_THREADS="${NCORES}"
SHARD=$(( C0 / PER ))
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  arm=${ARM}" \
     " shard=${SHARD} chunk_start=${C0}  out=${EMU_PK_SHARDS_EMU}  start=$(date -Is)"
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
# shellcheck disable=SC2086
python -u -m emu_pk.generate --mode emu --shard "${SHARD}" --n-per-shard "${PER}" \
    --n-total "${N_TOTAL}" --chunk-start "${C0}" --out "${EMU_PK_SHARDS_EMU}" ${EMU_PIN}
echo "done=$(date -Is)"
