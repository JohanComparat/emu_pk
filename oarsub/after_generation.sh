#!/usr/bin/env bash
# =============================================================================
# What follows generation, submitted when its input is on disk.
#
# Run on the frontend, in the background, from the repository root:
#
#   nohup setsid ./oarsub/after_generation.sh p150k 3000 > oarsub/logs/after_gen.log 2>&1 &
#
#   $1  training tag (names the weights: emu_pk_mlp_<campaign>_<arm>_<tag>.npz)
#   $2  number of generation chunks the design has (150 000 / 50 = 3000)
#   $3  epochs (default 240, the pilot's)
#
# 1. When every chunk is on disk, submit `train` for the design arm (`c`): it
#    assembles the training set -- the one on disk was assembled from the pilot
#    shards and is older than them, so it is rebuilt -- trains, and scores the
#    network against the truth it learnt.  Submitted earlier, the assembly would
#    take whatever chunks had landed and the network would never see the rest.
# 2. When that scoring has written its JSON, submit run_score_reference.sh for
#    the new network and the shipped one, against the prefilled reference cache.
#
# It looks at the disk every `EVERY` seconds (default 900), never at the queue.
# =============================================================================
set -euo pipefail
TAG="${1:?usage: after_generation.sh <tag> <n_chunks> [epochs]}"
N_CHUNKS="${2:?n_chunks}"
EPOCHS="${3:-240}"
EVERY="${EVERY:-900}"
ARM="${EMU_ARM:-c}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source ./oarsub/_campaign_env.sh >/dev/null 2>&1
campaign_arm "${ARM}" >/dev/null
PROJECT="$(campaign_project)"
SHARDS="${EMU_PK_SHARDS_EMU}"
WEIGHTS="${EMU_PK_WEIGHTS%.npz}_${TAG}.npz"
CACHE="${EMU_PK_WORK}/truth_cache_ref"
say() { echo "$(date '+%F %T')  $*"; }
chunks() { find "${SHARDS}" -maxdepth 1 -regextype posix-extended \
               -regex '.*/emu_[0-9]{5}_[0-9]{7}\.npz' | wc -l; }

say "arm ${ARM}, tag ${TAG}, ${EPOCHS} epochs; shards ${SHARDS}; weights ${WEIGHTS}"
[ -e "${WEIGHTS}" ] && { say "!! ${WEIGHTS} exists already: pick another tag"; exit 1; }
while [ "$(chunks)" -lt "${N_CHUNKS}" ]; do
    say "chunks $(chunks)/${N_CHUNKS}"
    sleep "${EVERY}"
done
say "every chunk on disk: submitting train"
EPOCHS="${EPOCHS}" TAG="${TAG}" EMU_ARM="${ARM}" ./oarsub/submit_campaign.sh train

until [ -f "${WEIGHTS%.npz}.validation.json" ]; do sleep "${EVERY}"; done
say "trained and scored on the training truth: submitting the reference score"
oarsub --project "${PROJECT}" --name "emupk_refscore_${TAG}" \
    -l /nodes=1/core=4,walltime=02:00:00 \
    --stdout "oarsub/logs/%jobid%.refscore.out" \
    --stderr "oarsub/logs/%jobid%.refscore.err" \
    -S "./oarsub/run_score_reference.sh ${CACHE} ${WEIGHTS} shipped"
say "done"
