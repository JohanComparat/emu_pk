#!/usr/bin/env bash
# =============================================================================
# Push the repo to Dahu.  Run LOCALLY -- it uses your ssh access to GRICAD.
#
#   ./oarsub/rsync_to_dahu.sh --dry-run
#   ./oarsub/rsync_to_dahu.sh
#
# The code travels by git in normal use; this exists for the case a run has to
# start from a working tree that is not pushed yet.
#
# Shards are NOT synced in either direction: they live on /bettik and are tens
# of GB.  Only the assembled products come back -- see pull_results.sh.
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
DAHU_REPO="${EMU_PK_REPO_REMOTE:-software/emu_pk}"  # home-relative: rsync
                                               # resolves it against the remote
                                               # login home, which avoids
                                               # relying on the remote shell to
                                               # expand $HOME
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="-n"

echo "== emu_pk -> ${DAHU_HOST}:${DAHU_REPO} ${DRY:+(dry-run)}"
rsync -avz --mkpath ${DRY} \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'oarsub/logs/*' --exclude 'shards*' --exclude '*.egg-info' \
  ./ "${DAHU_HOST}:${DAHU_REPO}/"

echo
echo "== next, on dahu:"
echo "   mamba activate \${EMU_PK_ENV:-hod_mod}   # must carry classy, jax, numpy, optax"
echo "   cd ~/${DAHU_REPO} && ./oarsub/submit_campaign.sh calibrate"
