#!/usr/bin/env python
r"""Solve ``validate``'s truth for some design points ahead of time, into its cache.

The reference truth (``validate --truth reference``) is CAMB at its converged
rung times the heating pair: minutes per point on many cores, and ``validate``
solves its points one after another, once per scored band.  This fills
``validate``'s own cache for chosen points through ``validate._class_pk``
itself -- same design, same k grids, same key -- so the points can be solved
as parallel short jobs and the ``validate`` runs that follow only read.

    python oarsub/prefill_reference.py --cache DIR --points 0 1 2 --tails 200 300

The bands are ``validate``'s: the headline band, and each network's tail from
10 h/Mpc to its k_max (``--tails``), which ``validate`` rounds to six figures
so that both name the same grid -- the k values are part of the cache key.
The design is ``validate``'s ``box.sample(n, seed)``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

# Run as a script from the repository, not installed: the package is the
# directory above this one, which a script's own sys.path does not include.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--points", type=int, nargs="+", required=True)
    ap.add_argument("--n-shape", type=int, default=32)
    ap.add_argument("--seed", type=int, default=991)
    ap.add_argument("--tails", type=float, nargs="+", default=[200.0, 300.0],
                    help="the k_max of each network to be scored: one tail "
                         "band, 10 to k_max, each")
    a = ap.parse_args(argv)

    from emu_pk import box, generate
    from emu_pk import validate as V
    V.TRUTH, V.CACHE_DIR = "reference", a.cache
    bands = [V.K_TRUSTED] + [(V.K_TAIL_MIN, float(t)) for t in a.tails]
    grids = [np.logspace(np.log10(lo), np.log10(hi), 300) for lo, hi in bands]
    z = np.asarray(V.Z_NODES, dtype=float)
    design = box.sample(a.n_shape, seed=a.seed)
    for i in a.points:
        paths = [V._cache_path(design[i], z, k) for k in grids]
        if all(p.exists() for p in paths):
            print(f"point {i:3d} cached", flush=True)
            continue
        # One reference solve for every band: the solve is the expensive
        # part, and each band only evaluates it -- pointwise, so a band's
        # columns of the union are exactly what solving on the band alone
        # returns.  The same call ``validate._class_pk`` makes.
        k_all = np.concatenate(grids)
        d = dict(zip(box.PARAMS, np.asarray(design[i], dtype=float)))
        t0 = time.perf_counter()
        pm, pcb = generate.solve_camb({p: float(d[p]) for p in box.PARAMS}, z,
                                      k_all, precision=V.REFERENCE_PRECISION)
        at = 0
        for k, path in zip(grids, paths):
            sl = slice(at, at + len(k))
            at += len(k)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".part.npz")
            np.savez(tmp, pm=pm[:, sl], pcb=pcb[:, sl])
            tmp.rename(path)
        print(f"point {i:3d} {time.perf_counter() - t0:7.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
