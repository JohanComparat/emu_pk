#!/usr/bin/env bash
# =============================================================================
# Submit one emu_pk campaign family on Dahu.
#
#   ./oarsub/submit_campaign.sh <family> [--devel]
#     family in { calibrate | ratio | emu | train | train-gpu }
#
#   ./oarsub/submit_campaign.sh calibrate     # gate 1: measure the rate
#   ./oarsub/submit_campaign.sh ratio         # Phase 1, ~300 solves
#   ./oarsub/submit_campaign.sh emu --devel   # one shard, smoke
#   ./oarsub/submit_campaign.sh emu           # Phase 2, the big one
#   ./oarsub/submit_campaign.sh train
#
# The OAR project comes from `EMU_PK_PROJECT`, set in the git-ignored
# `oarsub/site.sh` -- see `oarsub/site.sh.example`.  It is not an argument and
# not a default: no allocation name is committed to this repository.
#
# The training families read three settings from the *submitting* shell -- which
# is a login node, so this is safe where passing them to the node would not be;
# `run_train.sh` receives them as arguments (epochs, tag, arm, then flags):
#
#   EPOCHS       epochs to train                              (default 240)
#   TAG          names the weights, so ablation arms coexist   (default base)
#   TRAIN_FLAGS  passed to `emu_pk.train` verbatim             (default none)
#
# An ablation is that triple, once per arm:
#
#   EPOCHS=240 ./oarsub/submit_campaign.sh train
#   EPOCHS=240 TAG=noreduce TRAIN_FLAGS=--no-reduced   ./oarsub/submit_campaign.sh train
#   EPOCHS=240 TAG=noweight TRAIN_FLAGS=--no-weighted  ./oarsub/submit_campaign.sh train
#   EPOCHS=240 TAG=nosched  TRAIN_FLAGS=--no-schedule  ./oarsub/submit_campaign.sh train
#
# Run the first one alone first.  If the three changes together do not move the
# number there is nothing to ablate, and seven runs at 240 epochs is not a thing
# to launch on the strength of an argument.
#
# Prereqs on dahu: repo pulled at the intended commit, env built, /bettik work
# directory writable.  Smoke-test with --devel before the real submission --
# every family supports it, and `emu` in particular is 94 array elements that
# all fail identically if the environment is wrong.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./oarsub/_campaign_env.sh

FAMILY="${1:?usage: submit_campaign.sh <calibrate|ratio|emu|train|train-gpu> [--devel]}"
DEVEL="${2:-}"
PROJECT="$(campaign_project)"

# Which arm.  Read from the environment *here*, on the frontend where that
# works, and passed to the job as an argument, where it survives.
#
# `campaign_arm` is the *only* place that knows which arms exist, and it
# already refuses an unknown one by name.  This used to repeat the list as
# `case "${ARM}" in c|f)`, which then went stale the moment a third arm was
# added -- `EMU_ARM=d` was rejected here with "must be c or f" by a submitter
# whose own campaign environment had known about `d` for an hour.
ARM="${EMU_ARM:-c}"
campaign_arm "${ARM}" || exit 2

EPOCHS="${EPOCHS:-240}"
# Names the run, and therefore the weights file.  NOT "base": that maps to the
# unsuffixed `emu_pk_mlp.npz`, where the shipped weights sit on /bettik, so a
# default of `base` would have every run overwrite them.  A tag claims its own
# filename; only an explicit `TAG=base` takes the canonical one.
TAG="${TAG:-c2}"
TRAIN_FLAGS="${TRAIN_FLAGS:-}"
# Quoted as one word so `run_train.sh` sees "240 base --no-reduced ..." and
# splits it itself; TRAIN_FLAGS is deliberately unquoted inside so multi-flag
# strings expand.
# The arm sits between the tag and the flags because `run_train.sh` reads
# positionally and then shifts.  Leaving it out did not fail -- it defaulted to
# `c`, so `EMU_ARM=f ... train` assembled the *curved* shards into the curved
# dataset and wrote the curved weights, under a submission that said `f`.
TRAIN_ARGS="${EPOCHS} ${TAG} ${ARM} ${TRAIN_FLAGS}"

# The generator's shard count follows from the design, so it is computed rather
# than written down twice.  A submitter and an audit that disagree about how
# many shards there are report gaps that do not exist.
N_RATIO_SHARDS=$(( (300 + RATIO_PER_SHARD - 1) / RATIO_PER_SHARD ))
N_EMU_SHARDS=$(( (EMU_N_TOTAL + EMU_PER_SHARD - 1) / EMU_PER_SHARD ))

# GRICAD refuses a submission that would leave more than 100 jobs waiting, and
# an OAR array of N is N jobs.  Nothing checked this: raising EMU_N_TOTAL
# without raising EMU_PER_SHARD produced a rejection from the scheduler with no
# hint about which knob to turn.  Element *length* is not what a besteffort kill
# costs -- chunk length is -- so the fix is always a longer element.
# The cap is on jobs **already waiting** plus the ones being submitted, not on
# the array alone -- which is how a 94-element array that passed this check
# still bounced off `[MAX_JOBS] you cannot have more than 100 jobs waiting`,
# because the queue was not empty.  So ask the queue rather than assume it is.
# `grep -c` prints 0 and exits 1 when it matches nothing, and this file runs
# under `set -e`.  Both obvious repairs are wrong, and both fail *only* when the
# queue is empty -- the one case this guard exists to wave through:
#
#   n=$(... | grep -c X || echo 0)   ->  n is "0\n0"; the arithmetic below
#                                        dies with a syntax error
#   n=$(... | grep -c X); n=${n:-0}  ->  set -e kills the script at the
#                                        assignment, silently, before the
#                                        default is ever applied
#
# The `||` has to be on the *assignment*, where it is what suppresses `set -e`.
_waiting=$(oarstat -u "${USER}" 2>/dev/null | grep -c Waiting) || _waiting=0
_room=$(( 100 - _waiting ))
if [ "${FAMILY}" = "emu" ] && [ -z "${DEVEL}" ]; then
  if [ "${N_EMU_SHARDS}" -gt "${_room}" ]; then
    echo "!! ${N_EMU_SHARDS} array elements, but only ${_room} of GRICAD's" >&2
    echo "   100-waiting-job budget is free (${_waiting} already queued)." >&2
    echo "   Either wait for the queue to drain, or raise EMU_PER_SHARD to" >&2
    echo "   $(( (EMU_N_TOTAL + _room - 1) / _room )) so the array is ${_room} elements." >&2
    echo "   Element *length* is not what a besteffort kill costs -- chunk" >&2
    echo "   length is -- so a longer element trades nothing." >&2
    exit 2
  fi
  echo "[submit] queue:  ${_waiting} waiting, ${_room} free of 100; this array is ${N_EMU_SHARDS}"
fi

log_flags () {
  printf -- '--name emupk_%s --stdout oarsub/logs/%%jobid%%.%s.out --stderr oarsub/logs/%%jobid%%.%s.err' \
    "$1" "$1" "$1"
}

echo "[submit] family=${FAMILY} project=${PROJECT} ${DEVEL:+(devel)}"
echo "[submit] design: ratio ${N_RATIO_SHARDS} shards x ${RATIO_PER_SHARD}"
echo "[submit]         emu   ${N_EMU_SHARDS} shards x ${EMU_PER_SHARD} = ${EMU_N_TOTAL} cosmologies"
echo "[submit] arm:    ${ARM} (${EMU_PIN:-no pin, curved})"
echo "[submit] shards: ${EMU_PK_SHARDS_EMU}"
echo "[submit] work:   ${WORK}"
case "${FAMILY}" in
  train|train-gpu)
    echo "[submit] train:  ${EPOCHS} epochs, tag '${TAG}', flags '${TRAIN_FLAGS:-none}'" ;;
esac

case "${FAMILY}" in
  calibrate)
    # shellcheck disable=SC2046
    oarsub --project "${PROJECT}" -t devel \
      -l "/nodes=1/core=2,walltime=00:20:00" \
      $(log_flags cal) -S "./oarsub/run_calibrate.sh 8"
    ;;

  ratio)
    if [ -n "${DEVEL}" ]; then
      # shellcheck disable=SC2046
      oarsub --project "${PROJECT}" -t devel \
        -l "/nodes=1/core=2,walltime=00:30:00" \
        $(log_flags ratio_devel) -S "./oarsub/run_generate.sh ratio"
    else
      # 300 solves at a few seconds each: one array, short walltime, and no
      # besteffort -- it is small enough that queueing for a normal slot is
      # faster than being restarted out of a cheap one.
      # shellcheck disable=SC2046
      oarsub --project "${PROJECT}" \
        -l "/nodes=1/core=2,walltime=02:00:00" \
        --array "${N_RATIO_SHARDS}" \
        $(log_flags ratio) -S "./oarsub/run_generate.sh ratio"
    fi
    ;;

  emu)
    if [ -n "${DEVEL}" ]; then
      # shellcheck disable=SC2046
      oarsub --project "${PROJECT}" -t devel \
        -l "/nodes=1/core=2,walltime=00:30:00" \
        $(log_flags emu_devel) -S "./oarsub/run_generate.sh emu ${ARM} ${EMU_N_TOTAL} ${EMU_PER_SHARD}"
    else
      # besteffort + idempotent: shards skip if their output exists, so a
      # killed element re-runs and costs only what it had not finished.  That
      # is the property that makes the cheap queue the right queue here.
      # shellcheck disable=SC2046
      # 24 h, not 6.  The nodes are heterogeneous by a factor approaching two
      # and OAR says so on every submission, so this is sized on the *slow*
      # one.  Measured the same day on the eleven-parameter box: 14.0 s/solve
      # where `calibrate` landed, and 26.0 s/solve on dahu115 in the devel
      # smoke -- 6.2 h and 11.6 h respectively for an element of 1600.
      #
      # A besteffort kill is cheap: chunks skip, so a restart costs only the
      # partial chunk.  A *walltime* kill is not, because it lands on the last
      # chunk of every one of 94 elements at once.  The asymmetry is why this
      # is sized with a factor of two rather than a margin.  Dahu's ceiling is
      # 48 h.
      oarsub --project "${PROJECT}" \
        -l "/nodes=1/core=2,walltime=24:00:00" \
        -t besteffort -t idempotent \
        --array "${N_EMU_SHARDS}" \
        $(log_flags emu) -S "./oarsub/run_generate.sh emu ${ARM} ${EMU_N_TOTAL} ${EMU_PER_SHARD}"
    fi
    ;;

  train)
    # besteffort + idempotent, like the generation shards and for the same
    # reason: the admission rule makes every job here besteffort whether or not
    # it is asked for, so preemption is a certainty rather than a risk, and
    # without `idempotent` OAR does not put the job back.  Training checkpoints
    # every epoch and resumes from the checkpoint, so a preempted run loses one
    # epoch rather than the whole sweep.  Without `idempotent` a job killed
    # nine minutes in simply stays dead.
    #
    # That resume restores the optimiser as well as the weights: with a
    # scheduled learning rate, a restart that reinitialises Adam also rewinds
    # the schedule to its peak, which on this queue means a run that never
    # decays.
    #
    # Spelled as an if/else like every other family here, and not as a
    # `${DEVEL:+...}` / `${DEVEL:-...}` pair.  That pair looks symmetric and is
    # not: `${DEVEL:-X}` means "DEVEL if it is set, else X", so with DEVEL set
    # it expanded to the literal string `--devel` and handed it to oarsub,
    # which rejected it.  The devel path of this family had never run.
    # shellcheck disable=SC2046
    if [ -n "${DEVEL}" ]; then
      oarsub --project "${PROJECT}" -t devel \
        -l "/nodes=1/core=8,walltime=00:30:00" \
        $(log_flags train_devel) -S "./oarsub/run_train.sh ${TRAIN_ARGS}"
    else
      oarsub --project "${PROJECT}" \
        -t besteffort -t idempotent \
        -l "/nodes=1/core=32,walltime=24:00:00" \
        $(log_flags train) -S "./oarsub/run_train.sh ${TRAIN_ARGS}"
    fi
    ;;

  train-gpu)
    # The same script on Bigfoot.  `run_train.sh` is unchanged: JAX finds the
    # accelerator itself, `campaign_threads` now pins JAX_PLATFORMS=cpu only
    # where there is no `nvidia-smi` to find, and the job logs `jax.devices()`
    # on the first line so a GPU job that quietly ran on the CPU says so.
    #
    # Bigfoot needs a token before the reservation is accepted:
    #     gridtoken -i 9        # then re-run this
    # and `-t` on oarsub is that token's flag, not a job type.  This resource
    # line follows gricad-doc; it has not been exercised from this repository,
    # so smoke it with --devel before committing an ablation to it.
    if [ -z "${OAR_JOB_TOKEN:-}" ]; then
      echo "[submit] note: Bigfoot wants a token -- 'gridtoken -i 9' if this is refused" >&2
    fi
    # shellcheck disable=SC2046
    if [ -n "${DEVEL}" ]; then
      oarsub --project "${PROJECT}" -t devel \
        -l "/nodes=1/gpu=1,walltime=00:30:00" \
        $(log_flags train_gpu_devel) -S "./oarsub/run_train.sh ${TRAIN_ARGS}"
    else
      oarsub --project "${PROJECT}" \
        -t besteffort -t idempotent \
        -l "/nodes=1/gpu=1,walltime=12:00:00" \
        $(log_flags train_gpu) -S "./oarsub/run_train.sh ${TRAIN_ARGS}"
    fi
    ;;

  *) echo "unknown family: ${FAMILY}" >&2; exit 2 ;;
esac

echo "monitor:  oarstat -u \$USER   |   ./oarsub/campaign_status.sh"
