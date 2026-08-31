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
# `run_train.sh` receives them as arguments:
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
TRAIN_ARGS="${EPOCHS} ${TAG} ${TRAIN_FLAGS}"

# The generator's shard count follows from the design, so it is computed rather
# than written down twice.  A submitter and an audit that disagree about how
# many shards there are report gaps that do not exist.
N_RATIO_SHARDS=$(( (300 + RATIO_PER_SHARD - 1) / RATIO_PER_SHARD ))
N_EMU_SHARDS=$(( (EMU_N_TOTAL + EMU_PER_SHARD - 1) / EMU_PER_SHARD ))

log_flags () {
  printf -- '--name emupk_%s --stdout oarsub/logs/%%jobid%%.%s.out --stderr oarsub/logs/%%jobid%%.%s.err' \
    "$1" "$1" "$1"
}

echo "[submit] family=${FAMILY} project=${PROJECT} ${DEVEL:+(devel)}"
echo "[submit] design: ratio ${N_RATIO_SHARDS} shards x ${RATIO_PER_SHARD}"
echo "[submit]         emu   ${N_EMU_SHARDS} shards x ${EMU_PER_SHARD} = ${EMU_N_TOTAL} cosmologies"
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
        $(log_flags emu_devel) -S "./oarsub/run_generate.sh emu"
    else
      # besteffort + idempotent: shards skip if their output exists, so a
      # killed element re-runs and costs only what it had not finished.  That
      # is the property that makes the cheap queue the right queue here.
      # shellcheck disable=SC2046
      oarsub --project "${PROJECT}" \
        -l "/nodes=1/core=2,walltime=06:00:00" \
        -t besteffort -t idempotent \
        --array "${N_EMU_SHARDS}" \
        $(log_flags emu) -S "./oarsub/run_generate.sh emu"
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
