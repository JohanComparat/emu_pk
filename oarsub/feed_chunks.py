#!/usr/bin/env python3
r"""Keep short one-chunk generation jobs queued until a design range is done.

Run on the cluster frontend, in the background, from the repository root::

    nohup setsid python3 oarsub/feed_chunks.py --lo 0 --hi 16000 > oarsub/logs/feed.log 2>&1 &

Every ``--every`` seconds it lists the chunks already on disk and the
``emupk_ck_<start>`` jobs already queued or running, and submits one
``run_generate_chunk.sh`` job per missing chunk until this account has
``--max-waiting`` jobs waiting -- GRICAD refuses a submission past 50, and the
account runs other work too.  It stops when every chunk in ``[lo, hi)`` is on
disk.

**Order.**  Each shard's chunks from its *end* backwards, interleaved across
shards: a pooled node job works through its shards from the start, so the two
meet as late as possible, and a chunk the node job reaches that is already on
disk is skipped.  ``--avoid-shards`` names shards a running node job is inside;
their first missing chunk is left to it.

Pure standard library and ``oarstat -J``/``oarsub``: it runs in the frontend's
own Python, not in a job environment.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def site_env() -> dict:
    """The campaign's site.sh and arm, as the jobs will see them."""
    out = sh(["bash", "-c", f"source {HERE}/_campaign_env.sh >/dev/null 2>&1; "
                            "campaign_arm \"$ARM\" >/dev/null; "
                            "echo \"$EMU_PK_SHARDS_EMU\"; campaign_project"])
    shards, project = out.strip().splitlines()[-2:]
    return {"shards": pathlib.Path(shards), "project": project}


def done_chunks(shards: pathlib.Path) -> set[int]:
    out = set()
    for f in shards.glob("emu_*.npz"):
        m = re.fullmatch(r"emu_\d{5}_(\d{7})\.npz", f.name)
        if m:
            out.add(int(m.group(1)))
    return out


def jobs(user: str) -> tuple[set[int], int]:
    """Chunk starts with a queued or running job, and this account's waiting count."""
    d = json.loads(sh(["oarstat", "-u", user, "-J"]) or "{}")
    mine, waiting = set(), 0
    for j in d.values():
        state = j.get("state", "")
        if state in ("Waiting", "toLaunch", "Launching", "Hold"):
            waiting += 1
        m = re.fullmatch(r"emupk_ck_(\d+)", j.get("name") or "")
        if m and state not in ("Terminated", "Error"):
            mine.add(int(m.group(1)))
    return mine, waiting


def order(lo, hi, per, chunk, avoid, done):
    shards = range(lo // per, (hi - 1) // per + 1)
    per_shard = {}
    for s in shards:
        starts = [c for c in range(s * per, min((s + 1) * per, hi), chunk) if c >= lo]
        todo = [c for c in starts if c not in done]
        if s in avoid and todo:
            todo = todo[1:]            # the node job's current chunk
        per_shard[s] = list(reversed(todo))
    out = []
    while any(per_shard.values()):
        for s in shards:
            if per_shard[s]:
                out.append(per_shard[s].pop(0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="c")
    ap.add_argument("--n-total", type=int, default=150000)
    ap.add_argument("--per-shard", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--avoid-shards", type=int, nargs="*", default=[])
    ap.add_argument("--max-waiting", type=int, default=40)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--walltime", default="03:00:00")
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    os.environ["ARM"] = a.arm
    env = site_env()
    user = os.environ.get("USER", "")
    while True:
        done = done_chunks(env["shards"])
        queued, waiting = jobs(user)
        todo = [c for c in order(a.lo, a.hi, a.per_shard, a.chunk,
                                 set(a.avoid_shards), done) if c not in queued]
        n_all = len(range(a.lo, a.hi, a.chunk))
        room = max(0, a.max_waiting - waiting)
        print(f"{time.strftime('%F %T')}  done {len(done & set(range(a.lo, a.hi, a.chunk)))}/{n_all}"
              f"  queued {len(queued)}  waiting {waiting}  submitting {min(room, len(todo))}",
              flush=True)
        for c0 in todo[:room]:
            r = subprocess.run(
                ["oarsub", "--project", env["project"],
                 "-l", f"/nodes=1/core={a.cores},walltime={a.walltime}",
                 "-t", "besteffort", "-t", "idempotent",
                 "--name", f"emupk_ck_{c0}",
                 "--stdout", "oarsub/logs/%jobid%.ck.out",
                 "--stderr", "oarsub/logs/%jobid%.ck.err",
                 "-S", f"./oarsub/run_generate_chunk.sh {a.arm} {a.n_total} "
                       f"{a.per_shard} {c0}"],
                cwd=REPO, capture_output=True, text=True)
            if "OAR_JOB_ID" not in r.stdout:
                print(f"  !! chunk {c0}: {(r.stdout + r.stderr).strip()[-200:]}", flush=True)
                break
        if a.once or (not todo and not queued):
            if not todo and not queued:
                print("every chunk in range is on disk", flush=True)
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
