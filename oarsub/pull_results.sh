#!/usr/bin/env bash
# =============================================================================
# Bring the assembled products back from Dahu.  Run LOCALLY.
#
#   ./oarsub/pull_results.sh            # table + weights
#   ./oarsub/pull_results.sh --shards   # ...and the raw shards (tens of GB)
#
# The default deliberately leaves the shards on the cluster.  What the paper and
# the package need are the assembled table and the trained weights, a few MB
# between them; the shards are reproducible from a seed and an index.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Site configuration -- login, allocation, remote paths.  Git-ignored; copy
# `oarsub/site.sh.example` and fill it in.  No login name is committed here.
if [ -r ./oarsub/site.sh ]; then
    # shellcheck disable=SC1091
    source ./oarsub/site.sh
fi
if [ -z "${EMU_PK_SSH_HOST:-}" ]; then
    echo "!! EMU_PK_SSH_HOST is not set (e.g. a ~/.ssh/config Host alias)." >&2
    echo "   cp oarsub/site.sh.example oarsub/site.sh and fill it in." >&2
    exit 1
fi

DAHU_HOST="${EMU_PK_SSH_HOST}"
# Required, not derived: this script runs on your own machine, so `$USER` here
# is the local login and the remote path is built from the *cluster* one.  The
# GRICAD layout is /bettik/PROJECTS/<project>/<cluster-user>/emu_pk.
if [ -z "${EMU_PK_WORK:-}" ]; then
    echo "!! EMU_PK_WORK is not set -- the remote work directory." >&2
    echo "   Set it in oarsub/site.sh; it cannot be guessed from a local" >&2
    echo "   \$USER, which is not necessarily your cluster login." >&2
    exit 1
fi
DAHU_WORK="${EMU_PK_WORK}"

mkdir -p emu_pk/data
echo "== pulling assembled products from ${DAHU_HOST}:${DAHU_WORK}"
rsync -avz --ignore-missing-args \
  "${DAHU_HOST}:${DAHU_WORK}/class_pk_ratio.npz" \
  "${DAHU_HOST}:${DAHU_WORK}/emu_pk_mlp.npz" \
  emu_pk/data/ || echo "  (nothing to pull yet)"

if [ "${1:-}" = "--shards" ]; then
  echo "== pulling raw shards (this is large)"
  rsync -avz --mkpath "${DAHU_HOST}:${DAHU_WORK}/shards_ratio" ./work/
  rsync -avz --mkpath "${DAHU_HOST}:${DAHU_WORK}/shards_emu" ./work/
fi
ls -la emu_pk/data/
