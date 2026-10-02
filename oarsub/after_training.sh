#!/usr/bin/env bash
# =============================================================================
# Score networks trained elsewhere, on Dahu, when each has finished training.
#
# Run on the Dahu frontend, in the background, from the repository root:
#
#   nohup setsid ./oarsub/after_training.sh p150kbf p150kkr > oarsub/logs/after_train.log 2>&1 &
#
# For each tag, waits for `<weights>.trained` (written by run_train.sh when the
# fit ends; the weights file itself exists from the first epoch), then submits
# run_validate.sh -- the training truth, the full suite -- and
# run_score_reference.sh -- the converged reference, shape and tails -- for it.
# A run that already scored itself (`<weights>.validation.json`, a Dahu run)
# gets the reference score only.
# A GPU cluster's environment has no CAMB, so this is where those scores come
# from.  Looks at /bettik every `EVERY` seconds (default 900), never at the queue.
# =============================================================================
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: after_training.sh <tag> [<tag> ...]" >&2; exit 2; }
EVERY="${EVERY:-900}"
ARM="${EMU_ARM:-c}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=/dev/null
source ./oarsub/_campaign_env.sh >/dev/null 2>&1
campaign_arm "${ARM}" >/dev/null
PROJECT="$(campaign_project)"
CACHE="${EMU_PK_WORK}/truth_cache_ref"
say() { echo "$(date '+%F %T')  $*"; }

pending=("$@")
say "arm ${ARM}; waiting for ${pending[*]}"
while [ ${#pending[@]} -gt 0 ]; do
    left=()
    for TAG in "${pending[@]}"; do
        W="${EMU_PK_WEIGHTS%.npz}_${TAG}.npz"
        # A run on a node with the solvers scored itself (run_train.sh): only
        # the reference score is left.  One trained elsewhere needs both.
        if [ -f "${W%.npz}.validation.json" ] || [ -f "${W%.npz}.trained" ]; then
            if [ -f "${W%.npz}.validation.json" ]; then
                say "${TAG} trained and scored: submitting its reference score"
            else
                say "${TAG} trained: submitting its scores"
                oarsub --project "${PROJECT}" --name "emupk_val_${TAG}" \
                    -l /nodes=1/core=16,walltime=12:00:00 \
                    --stdout "oarsub/logs/%jobid%.val.out" --stderr "oarsub/logs/%jobid%.val.err" \
                    -S "./oarsub/run_validate.sh ${W}"
            fi
            oarsub --project "${PROJECT}" --name "emupk_refscore_${TAG}" \
                -l /nodes=1/core=4,walltime=02:00:00 \
                --stdout "oarsub/logs/%jobid%.refscore.out" --stderr "oarsub/logs/%jobid%.refscore.err" \
                -S "./oarsub/run_score_reference.sh ${CACHE} ${W}"
        else
            left+=("${TAG}")
        fi
    done
    pending=("${left[@]+"${left[@]}"}")
    [ ${#pending[@]} -gt 0 ] && sleep "${EVERY}"
done
say "done"
