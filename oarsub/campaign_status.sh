#!/usr/bin/env bash
# =============================================================================
# Which shards landed, and which did not.
#
# Answers "can I assemble yet?".  `assemble.py` refuses a partial ratio grid by
# design -- a table built from a sweep with holes is wrong at exactly the nodes
# that are missing -- so this is what you run first.
#
#   ./oarsub/campaign_status.sh            # both families
#   ./oarsub/campaign_status.sh ratio      # one
#
# Exit 0 = complete; 1 = something is missing, and the missing shard indices are
# named rather than counted.  A count is what lets a campaign look 97% done
# forever; the indices are what you resubmit.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./oarsub/_campaign_env.sh

WHICH="${1:-all}"
miss=0

ok()  { printf "  \033[32m OK \033[0m %s\n" "$1"; }
bad() { printf "  \033[31mMISS\033[0m %s\n" "$1"; miss=$((miss+1)); }

# A shard "exists" only if it is non-empty and loadable.  A zero-byte or
# truncated .npz left by a job killed mid-write is worse than an absent one:
# assemble would read it and fail somewhere far from here.
# audit <dir> <label> -- expected filenames arrive on stdin, one per line.
#
# A shard "exists" only if it is non-empty and closed cleanly.  A truncated
# .npz left by a job killed mid-write is worse than an absent one: the audit
# would count it and the assembler would fail somewhere far from here.  Writes
# go to a .part name and are renamed, so this should never fire -- which is
# exactly why it is checked.
audit () {
    local dir="$1" label="$2"
    local expected=() missing=() broken=() f
    mapfile -t expected
    echo "-- ${label}: ${#expected[@]} files expected in ${dir}"
    if [ ! -d "${dir}" ]; then bad "${dir} does not exist"; return; fi
    for f in "${expected[@]}"; do
        if [ ! -s "${dir}/${f}" ]; then missing+=("${f}"); continue; fi
        if command -v unzip >/dev/null 2>&1; then
            unzip -tq "${dir}/${f}" >/dev/null 2>&1 || broken+=("${f}")
        fi
    done
    local nm=${#missing[@]} nb=${#broken[@]} more=""
    local partial; partial=$(find "${dir}" -name '*.npz.part' 2>/dev/null | wc -l)
    if [ "${nm}" -eq 0 ] && [ "${nb}" -eq 0 ]; then
        ok "all ${#expected[@]} files present"
        [ "${partial}" -gt 0 ] && echo "     (${partial} .part files: chunks in flight)"
        return
    fi
    if [ "${nm}" -gt 0 ]; then
        [ "${nm}" -gt 12 ] && more="... (${nm} total)"
        bad "${nm} missing: $(printf '%s ' "${missing[@]:0:12}")${more}"
    fi
    [ "${nb}" -gt 0 ] && bad "${nb} unreadable: $(printf '%s ' "${broken[@]:0:12}")"
    [ "${partial}" -gt 0 ] && echo "     ${partial} .part files: chunks still in flight"
    echo "     resubmit with:  ./oarsub/submit_campaign.sh <PROJECT> ${label}"
    echo "     (a chunk that landed is skipped, so this only redoes the gaps)"
}

N_RATIO=$(( (300 + RATIO_PER_SHARD - 1) / RATIO_PER_SHARD ))
N_EMU=$(( (EMU_N_TOTAL + EMU_PER_SHARD - 1) / EMU_PER_SHARD ))

echo "== emu_pk campaign status   work=${WORK}"
CHUNK="${EMU_CHUNK:-50}"

if [ "${WHICH}" = "all" ] || [ "${WHICH}" = "ratio" ]; then
    # Process substitution, not a pipe: the right-hand side of a pipe runs in a
    # subshell, so `miss` would be incremented there and lost, and the summary
    # would print "complete" over its own MISS lines.
    audit "${EMU_PK_SHARDS_RATIO}" ratio < <(
        for (( i=0; i<N_RATIO; i++ )); do printf 'ratio_%05d.npz\n' "${i}"; done)
fi
if [ "${WHICH}" = "all" ] || [ "${WHICH}" = "emu" ]; then
    # One file per chunk, named for the shard that owns it and the design index
    # it starts at -- so the name says which cosmologies are inside without
    # opening it, and a resubmitted shard writes the same names.
    audit "${EMU_PK_SHARDS_EMU}" emu < <(
        for (( c=0; c<EMU_N_TOTAL; c+=CHUNK )); do
            printf 'emu_%05d_%07d.npz\n' "$(( c / EMU_PER_SHARD ))" "${c}"
        done)
fi

echo
if [ "${miss}" -eq 0 ]; then
    echo "== complete.  Assemble with:"
    echo "   python -m emu_pk.assemble --mode ratio --shards ${EMU_PK_SHARDS_RATIO}"
    echo "   python -m emu_pk.assemble --mode emu   --shards ${EMU_PK_SHARDS_EMU} --out ${EMU_PK_DATASET}"
    exit 0
else
    echo "== ${miss} gap(s).  Assembly would be built on a sweep with holes; resubmit first."
    exit 1
fi
