#!/usr/bin/env bash
# =============================================================================
# GRICAD / OAR devel job: time CLASS on a Dahu node at production settings.
#
# This exists because the run is sized from a measured rate, not a guess.
# It runs a handful of solves drawn from the *actual* design -- so the number
# covers the expensive corners (k_max = 200 h/Mpc, massive neutrinos, CPL) and
# not the fiducial alone -- and prints seconds/solve, core-hours per 1e5 solves
# and peak RSS.
#
# The devel partition caps at 30 minutes, which is ample: 8 solves is a minute.
#
#   oarsub --project "${EMU_PK_PROJECT}" -t devel \
#          -l /nodes=1/core=2,walltime=00:20:00 -S ./oarsub/run_calibrate.sh
#
# Wall-clock IS the measurement here, so pin the CPU model when comparing two
# calibrations: Dahu's default queue is heterogeneous and a fat node is a
# different machine.  Add /cpumodel=1/ to the resource string to pin it.
# =============================================================================

# No `#OAR --project` directive -- see run_generate.sh.
#OAR --name emupk_cal
#OAR -l /nodes=1/core=2,walltime=00:20:00
#OAR --stdout oarsub/logs/%jobid%.emupk_cal.out
#OAR --stderr oarsub/logs/%jobid%.emupk_cal.err

set -euo pipefail
N="${1:-8}"
source "$(dirname "${BASH_SOURCE[0]}")/_campaign_env.sh"
cd "${REPO}"
mkdir -p oarsub/logs
campaign_activate_env
NCORES="$(campaign_threads)"
echo "host=$(hostname)  job=${OAR_JOB_ID:-local}  cores=${NCORES}  start=$(date -Is)"
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
python -u -m emu_pk.generate --mode time --n-per-shard "${N}"
echo "done=$(date -Is)"
