#!/usr/bin/env bash
# =============================================================================
# Shared environment for every emu_pk job on GRICAD/Dahu.
#
# Sourced by run_*.sh; not executable on its own.  Everything an account has to
# change lives here rather than being repeated in each script.
# =============================================================================

# --- site configuration ------------------------------------------------------
# Everything that identifies an account, an allocation or a host lives in
# `oarsub/site.sh`, which is git-ignored.  Copy `site.sh.example` and fill it
# in.  Nothing below carries a default that would work for only one person, so
# a fresh checkout fails with an instruction rather than submitting a job
# against somebody else's allocation.
if [ -r "$(dirname "${BASH_SOURCE[0]}")/site.sh" ]; then
    # shellcheck disable=SC1091
    source "$(dirname "${BASH_SOURCE[0]}")/site.sh"
fi

# The checkout this file belongs to, not a hardcoded path.  A campaign that
# wants to launch a new arm while earlier ones are still running cannot touch
# their working tree -- bash reads a job script incrementally, so rewriting
# `run_train.sh` under a running job makes it resume reading a different file at
# a stale byte offset.  A second checkout is the way to do that, and it only
# works if the scripts operate on the tree they live in.
_CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${EMU_PK_REPO:-$(dirname "${_CAMPAIGN_DIR}")}"

# Which cluster this is.  Not from `/applis/cluster_name` -- that file says
# "luke" on both dahu and bigfoot, which is worse than having no file at all.
# The hostname is `dahu`/`bigfoot` on the frontends and `dahu103`/`bigfoot7` on
# the compute nodes, so a prefix match is what works in a job.
campaign_cluster () {
    case "$(hostname)" in
        bigfoot*) echo bigfoot ;;
        dahu*)    echo dahu ;;
        *)        echo unknown ;;
    esac
}

# Environment name, per cluster.  The homes are NOT shared between dahu and
# bigfoot, so these are two separate installations that happen to serve one
# repository -- see "Building the environment" in oarsub/README.md.  Only
# /bettik crosses, which is why the training set does not have to.
if [ -z "${EMU_PK_ENV:-}" ]; then
    case "$(campaign_cluster)" in
        bigfoot) CONDA_ENV="emu_pk" ;;      # site conda, jax[cuda12], no classy
        *)       CONDA_ENV="hod_mod" ;;     # personal miniforge, everything
    esac
else
    CONDA_ENV="${EMU_PK_ENV}"
fi
# The OAR allocation this work is charged to.  No default: a wrong project is
# either a submission failure or somebody else's compute, and both deserve to
# stop here rather than at `oarsub`.
campaign_project () {
    if [ -z "${EMU_PK_PROJECT:-}" ]; then
        echo "!! EMU_PK_PROJECT is not set." >&2
        echo "   cp oarsub/site.sh.example oarsub/site.sh and fill it in," >&2
        echo "   or export EMU_PK_PROJECT=<your-oar-project>." >&2
        return 1
    fi
    echo "${EMU_PK_PROJECT}"
}

# /bettik is the parallel BeeGFS scratch on GRICAD, provisioned per project --
# a user cannot create /bettik/$USER, only write under their project's tree.
# This is where the shards belong: a production run writes tens of GB and $HOME
# is a shared NAS with no room for them.
WORK="${EMU_PK_WORK:-/bettik/PROJECTS/${EMU_PK_PROJECT:-UNSET}/${USER}/emu_pk}"

export EMU_PK_SHARDS_RATIO="${WORK}/shards_ratio"
export EMU_PK_SHARDS_EMU="${WORK}/shards_emu"
export EMU_PK_DATASET="${WORK}/training_set.npz"
export EMU_PK_WEIGHTS="${WORK}/emu_pk_mlp.npz"

# --- conda / mamba -----------------------------------------------------------
# Which modules a job actually needs.  Generation and validation need the
# Boltzmann solver; training does not, and on Bigfoot it is not there -- CLASS
# does not build on that login node (see oarsub/README.md).  Checking for
# `classy` unconditionally would abort every GPU training job before it
# started, over a dependency it was never going to import.
#
#   campaign_activate_env              # numpy, jax, classy  (the old behaviour)
#   campaign_activate_env train        # numpy, jax, optax
campaign_activate_env () {
    # Two clusters, two package managers, and neither is a migration of the
    # other -- both are live.  Dahu has a personal miniforge in $HOME; bigfoot
    # has no mamba, no conda and no nix on PATH at all, and offers the site
    # miniconda under /applis instead.  Prefer whichever is actually there.
    local mamba_exe="${MAMBA_EXE:-${HOME}/miniforge3/bin/mamba}"
    if [ -x "${mamba_exe}" ]; then
        export MAMBA_EXE="${mamba_exe}"
        export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/miniforge3}"
        local hook
        if hook="$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}" 2>/dev/null)"; then
            eval "${hook}"
        else
            alias mamba="${MAMBA_EXE}"
        fi
        mamba activate "${CONDA_ENV}"
    elif [ -r /applis/environments/conda.sh ]; then
        # shellcheck disable=SC1091
        source /applis/environments/conda.sh
        conda activate "${CONDA_ENV}"
    else
        echo "!! no mamba at ${mamba_exe} and no /applis/environments/conda.sh." >&2
        echo "   Set MAMBA_EXE, or build the env per oarsub/README.md." >&2
        exit 1
    fi
    echo "-- cluster=$(campaign_cluster) env=${CONDA_ENV} python=$(command -v python)"
    # Fail here, loudly, rather than every array element failing identically
    # twenty minutes later with an ImportError nobody reads.
    EMU_PK_NEED="${1:-gen}" python - <<'PYEOF' || exit 1
import os
import sys

need = {"gen": ("numpy", "jax", "classy"),
        "train": ("numpy", "jax", "optax")}[os.environ["EMU_PK_NEED"]]
missing = []
for m in need:
    try:
        __import__(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__})")
if missing:
    sys.exit("!! environment is missing: " + ", ".join(missing))
print("environment ok: " + ", ".join(need))
PYEOF
}

# --- cores actually allocated ------------------------------------------------
# OAR_RES_NB_CORES does not exist on this OAR build -- the env carries
# OAR_NODEFILE and OAR_CPUSET but no such variable, so a `${OAR_RES_NB_CORES:-8}`
# always falls through to its default and every job threads to the wrong width.
# OAR_NODEFILE has one line per allocated core, and OAR uses cpusets so nproc
# agrees; keep both, plus a floor.
campaign_ncores () {
    local n="${OAR_RES_NB_CORES:-}"
    if [ -z "${n}" ] && [ -r "${OAR_NODEFILE:-/nonexistent}" ]; then
        n="$(wc -l < "${OAR_NODEFILE}")"
    fi
    echo "${n:-$(nproc 2>/dev/null || echo 4)}"
}

campaign_threads () {
    local n; n="$(campaign_ncores)"
    export OMP_NUM_THREADS="${n}"
    export OPENBLAS_NUM_THREADS="${n}"
    export MKL_NUM_THREADS="${n}"
    # Pin JAX to CPU only where there is nothing else to find.  Dahu has no
    # accelerator, so pinning there saves a startup probe.  Bigfoot does, and
    # pinning there would run the `train-gpu` family on the CPU of a GPU node --
    # which does not fail, it just takes as long as Dahu, and reads as "the GPU
    # did not help" rather than as "the GPU was never used".
    if [ -z "${JAX_PLATFORMS:-}" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
        export JAX_PLATFORMS=cpu
    fi
    echo "${n}"
}

# --- the campaign design -----------------------------------------------------
# One place, read by the submitter, the workers and the status audit.  A shard
# count that disagrees between submission and audit reports phantom gaps.
EMU_N_TOTAL="${EMU_N_TOTAL:-150000}"     # cosmologies in the emu design
# GRICAD refuses a submission that would leave more than 100 jobs waiting, and
# an OAR array of N is N jobs.  So the design is cut into 94 elements rather
# than 300, and each is correspondingly longer.  Element length is not what a
# besteffort kill costs -- chunk length is -- so this trades nothing.
EMU_PER_SHARD="${EMU_PER_SHARD:-1600}"      # cosmologies per array element
RATIO_PER_SHARD="${RATIO_PER_SHARD:-25}"    # nodes per array element (300 total)
export EMU_N_TOTAL EMU_PER_SHARD RATIO_PER_SHARD
