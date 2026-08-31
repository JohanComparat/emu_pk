#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR job: assemble the training set, then fit the network.
#
#   $1  epochs (default 60)
#   $2  tag, naming the weights so ablation runs coexist (default "c2").
#       `base` is the one tag that takes the unsuffixed `emu_pk_mlp.npz`, the
#       canonical filename, so pass it deliberately or not at all.
#   $3+ passed through to `emu_pk.train` verbatim -- this is how an ablation
#       turns individual changes off (--no-reduced, --no-weighted,
#       --no-schedule, --hidden 1024 1024 1024 1024)
#
# Arguments and not environment, for the reason the generator's are: OAR does
# not propagate the submitting shell's environment to the node, so a setting
# passed as `FOO=x oarsub -S ...` silently runs the default.  `submit_campaign.sh`
# bakes them into the command string.
#
# Dahu (CPU) by default.  The network is a 4x512 MLP over ~64 PCA components,
# small enough that a 32-core Dahu node trains it in hours, and that avoids the
# Bigfoot token dance (`oarsub -T`, `gridtoken -i 9`) for no gain while a single
# run is the whole job.  For an ablation -- several runs of the same length --
# `submit_campaign.sh train-gpu` runs this same script on Bigfoot, with only the
# resource line differing.
#
# besteffort + idempotent: the trainer checkpoints every epoch and resumes from
# the checkpoint, so a kill costs at most one epoch.  That is what makes it
# safe to run in the cheap queue.
# =============================================================================

# No `#OAR --project` directive: the allocation is site configuration, not
# repository content, and `submit_campaign.sh` passes it on the command line
# where it belongs.  See oarsub/site.sh.example.
#OAR --name emupk_train
#OAR -l /nodes=1/core=32,walltime=24:00:00
#OAR -t besteffort
#OAR -t idempotent
#OAR --stdout oarsub/logs/%jobid%.emupk_train.out
#OAR --stderr oarsub/logs/%jobid%.emupk_train.err

set -euo pipefail
EPOCHS="${1:-60}"
TAG="${2:-base}"
# `shift 2` fails under `set -e` when fewer than two arguments were given, and
# a job script that exits 1 before doing anything looks exactly like a job that
# was never scheduled.
if [ $# -gt 2 ]; then shift 2; else set --; fi
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"

# One weights file per ablation arm.  Without this every arm overwrites the
# last, and `--no-resume` is not the answer either: a besteffort arm has to be
# able to resume *itself* without resuming its neighbour.
OUT="${EMU_PK_WEIGHTS}"
[ "${TAG}" = "base" ] || OUT="${EMU_PK_WEIGHTS%.npz}_${TAG}.npz"
if [ "${TAG}" = "base" ] && [ -f "${OUT}" ]; then
    echo "!! TAG=base writes ${OUT}, which already exists -- that is the"
    echo "   canonical weights file.  Use a run tag unless you mean to replace it."
fi

cd "${REPO}"
mkdir -p oarsub/logs
campaign_activate_env train
NCORES="$(campaign_threads)"
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  tag=${TAG}" \
     " epochs=${EPOCHS}  flags='$*'  out=${OUT}  start=$(date -Is)"
# Say which device JAX actually took.  A GPU job that quietly ran on the CPU is
# indistinguishable from a slow GPU in every other line of this log.
python -c "import jax; print('jax devices:', jax.devices())" || true

# Assemble only if the dataset is missing or older than the newest shard.
# Rebuild also when the dataset has no part layout: a manifest without `parts`
# still loads, but it loads at a few MB/s off /bettik, which costs the better
# part of an hour on every job restart.
parts_ok=0
if [ -f "${EMU_PK_DATASET}" ] && ls "${EMU_PK_DATASET%.npz}".part*.npz >/dev/null 2>&1; then
    parts_ok=1
fi
newest="$(find "${EMU_PK_SHARDS_EMU}" -name 'emu_*.npz' -newer "${EMU_PK_DATASET}" -print -quit 2>/dev/null || true)"
if [ ! -f "${EMU_PK_DATASET}" ] || [ -n "${newest}" ] || [ "${parts_ok}" -eq 0 ]; then
    echo "-- assembling training set"
    # Threads, not cores: the shard read is filesystem latency, not CPU.
    python -u -m emu_pk.assemble --mode emu --shards "${EMU_PK_SHARDS_EMU}" \
           --out "${EMU_PK_DATASET}" --workers "$(( NCORES > 16 ? NCORES : 16 ))"
else
    echo "-- training set is current, skipping assembly"
fi

python -u -m emu_pk.train --dataset "${EMU_PK_DATASET}" \
       --out "${OUT}" --epochs "${EPOCHS}" "$@"

# Score it here, while the environment that trained it is still loaded, and
# write the numbers to a file beside the weights.  A validation figure retyped
# by hand outlives the weights it describes; one that is not produced at all
# gets quoted from the previous campaign, which is what happened last time.
#
# Needs CLASS, which the Bigfoot environment does not have.  Say which of the
# two it is: "no solver here, score it on dahu" and "the scoring ran and threw"
# are different problems and only one of them is a bug.
#
# Sized from the run.  A production validation is ~300 CLASS solves and the
# better part of an hour; the devel queue caps at thirty minutes total, so a
# smoke that trains for three epochs and then asks for the full score gets
# killed in the scoring and reports nothing at all.  A smoke wants to know the
# scoring *runs*, not what it says.
if [ "${EPOCHS}" -lt 10 ]; then
    VAL_ARGS="--n-shape 4 --n-deriv 1 --z 0.0 1.0 --no-convergence"
    echo "-- ${EPOCHS} epochs is a smoke; scoring it with: ${VAL_ARGS}"
else
    VAL_ARGS=""
fi
if python -c "import classy" 2>/dev/null; then
    # shellcheck disable=SC2086
    python -u -m emu_pk.validate --weights "${OUT}" \
           --json "${OUT%.npz}.validation.json" ${VAL_ARGS} || \
        echo "!! validation ran and failed; the weights are still at ${OUT}"
else
    echo "-- no classy here (expected on bigfoot); score it on dahu with"
    echo "   python -m emu_pk.validate --weights ${OUT} --json ${OUT%.npz}.validation.json"
fi
echo "done=$(date -Is)"
