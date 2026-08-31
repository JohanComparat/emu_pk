r"""Shards in, one table or one training set out.

Two products, and the first of them makes a decision that has to be checked
rather than taken on trust.

The correction table ships **factorised**,

.. math::  r(k,z;\,f_\nu,w_0,w_a) \simeq r^{\nu}(k,z;f_\nu)\;r^{\rm DE}(k,z;w_0,w_a)

because a full five-axis tensor-product Hermite needs :math:`2^4` derivative
arrays over a 3.6-million-element cube -- hundreds of megabytes resident -- while
the two factors need :math:`2^2` and :math:`2^3` over cubes a few hundred
kilobytes each.  The factors are the LambdaCDM slice and the massless slice of
the same grid, so each is exact on its own axis and the approximation lives
entirely in the cross term.

:func:`build_ratio` therefore solves the **whole** grid, not just the two
slices, and measures the cross term it is discarding.  The number is printed and
stored in the table as ``resid_max``.  If it is not comfortably below the
emulator's own shape error the factorisation is the wrong call, and the number
is there to say so rather than to be assumed away.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import time

import numpy as np

from . import cosmo, grid

__all__ = ["build_ratio", "build_training_set", "load_training_set",
           "main"]


def _load_ratio_shards(shard_dir):
    """Every ``ratio_*.npz`` in one place, keyed by ``(mnu, w0, wa)``."""
    files = sorted(pathlib.Path(shard_dir).glob("ratio_*.npz"))
    if not files:
        raise FileNotFoundError(f"no ratio_*.npz under {shard_dir}")
    spec, z, lnk, meta = {}, None, None, {}
    for f in files:
        with np.load(f) as d:
            if z is None:
                z, lnk = d["z"], d["lnk"]
                meta = {"h_fid": float(d["h_fid"]),
                        "Omega_m_fid": float(d["Omega_m_fid"])}
            elif not (np.array_equal(z, d["z"]) and np.array_equal(lnk, d["lnk"])):
                raise ValueError(
                    f"{f.name} was generated on a different grid from "
                    f"{files[0].name}.  Shards from two different sweeps "
                    "cannot be merged; regenerate, or keep them in separate "
                    "directories.")
            for row, pm, pcb in zip(d["theta"], d["pm"], d["pcb"]):
                spec[(round(float(row[0]), 6), round(float(row[1]), 6),
                      round(float(row[2]), 6))] = (pm, pcb)
    return spec, z, lnk, meta


def build_ratio(shard_dir, out=None, verbose=True):
    """Assemble the correction table, and measure what a factorised one would cost.

    Ships the **full** four-axis cube.  The factorised form -- a neutrino factor
    times a dark-energy factor -- is far cheaper and not accurate enough: its
    cross term reaches 1.6 percent at high neutrino mass with strongly
    non-LambdaCDM dark energy, an order of magnitude above the emulator's own
    0.16 percent shape error.  The two effects couple physically -- more
    late-time growth is more time for free streaming to suppress -- so the
    discrepancy grows with the product of the two, exactly as it should.

    That number is computed and stored as ``resid_max``, because it is what
    justifies the memory the full cube costs.
    """
    spec, z, lnk, meta = _load_ratio_shards(shard_dir)
    mnu_n, w0_n, wa_n = grid.MNU_NODES, grid.W0_NODES, grid.WA_NODES
    n_z, n_k = len(z), len(lnk)

    def get(mnu, w0, wa):
        key = (round(float(mnu), 6), round(float(w0), 6), round(float(wa), 6))
        if key not in spec:
            raise KeyError(
                f"grid node {key} is missing from the shards.  The table cannot "
                "be assembled from a partial sweep -- run "
                "`oarsub/campaign_status.sh` to see which shards did not land.")
        return spec[key]

    pm_ref, _ = get(0.0, -1.0, 0.0)                 # the LambdaCDM massless corner

    shape = (len(mnu_n), len(w0_n), len(wa_n), n_z, n_k)
    lnr_m = np.zeros(shape)
    lnr_cb = np.zeros(shape)
    f_nu = np.array([cosmo.f_nu(float(m), meta["h_fid"], meta["Omega_m_fid"])
                     for m in mnu_n])
    for i, mnu in enumerate(mnu_n):
        for a, w0 in enumerate(w0_n):
            for b, wa in enumerate(wa_n):
                pm, pcb = get(mnu, w0, wa)
                lnr_m[i, a, b] = np.log(pm / pm_ref)
                lnr_cb[i, a, b] = np.log(pcb / pm_ref)

    # -- what a factorised table would discard -------------------------------
    i_w0 = int(np.argmin(np.abs(w0_n + 1.0)))
    i_wa = int(np.argmin(np.abs(wa_n)))
    worst, where = 0.0, None
    for i, mnu in enumerate(mnu_n):
        for a in range(len(w0_n)):
            for b in range(len(wa_n)):
                for cube in (lnr_m, lnr_cb):
                    fac = cube[i, i_w0, i_wa] + cube[0, a, b]
                    e = float(np.max(np.abs(np.expm1(cube[i, a, b] - fac))))
                    if e > worst:
                        worst, where = e, (float(mnu), float(w0_n[a]),
                                           float(wa_n[b]))
    if verbose:
        m, w, a = where
        print("=== the factorisation this table does not use ===")
        print(f"  max |r_full/(r_nu . r_de) - 1| = {worst:.3%}"
              f"   at mnu={m:.2f}, w0={w:+.2f}, wa={a:+.2f}")
        print(f"  emulator shape error for comparison: ~0.16%")
        print(f"  -> shipping the full cube ({np.prod(shape) * 4 / 1e6:.1f} MB "
              f"per spectrum, float32)")

    out = (pathlib.Path(__file__).resolve().parent / "data" / "class_pk_ratio.npz"
           if out is None else pathlib.Path(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    # float32: the stored quantity is a log-ratio of order 0.01 to 0.2, and
    # float32 resolves it to ~1e-7 relative -- three orders below the accuracy
    # anything downstream asks of it -- while halving what a likelihood holds
    # resident.  The zero at the LambdaCDM massless corner is exact in either.
    np.savez_compressed(
        out, lnk=lnk, z=z, f_nu=f_nu, mnu=mnu_n, w0=w0_n, wa=wa_n,
        lnr_m=lnr_m.astype(np.float32), lnr_cb=lnr_cb.astype(np.float32),
        resid_max=worst, h_fid=meta["h_fid"], Omega_m_fid=meta["Omega_m_fid"])
    if verbose:
        print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB on disk)")
    return out


def build_training_set(shard_dir, out, dtype=np.float32, workers=16,
                       parts=32):
    """Concatenate ``emu_*.npz`` shards into one design matrix and one target.

    Rows are ``(cosmology, redshift)`` pairs: one CLASS solve contributes every
    redshift in :data:`emu_pk.grid.Z_NODES_EMU`, so the design is the outer
    product of the sampled box with that axis.

    Missing design indices are reported rather than filled.  A training set with
    silent gaps trains perfectly well and is wrong exactly where CLASS refused,
    which is the part of the box a forecast is most likely to wander into.

    ``workers`` threads read the shards; see the comment below for why threads
    and not processes, and why the default is not 1.
    """
    files = sorted(pathlib.Path(shard_dir).glob("emu_*.npz"))
    if not files:
        raise FileNotFoundError(f"no emu_*.npz under {shard_dir}")
    n_files = len(files)

    # **Read the shards in parallel.**  Measured on Dahu against /bettik: one
    # 3.0 MB compressed shard takes 3.8 s to open and decompress, which is
    # 0.8 MB/s and puts a serial pass over 3000 of them at three hours.  The
    # cost is the shared filesystem rather than zlib, so it is latency and not
    # CPU, and threads recover it even under the GIL -- numpy's decompression
    # and the read itself both release it.  The first run of this was serial and
    # spent an hour without reporting anything, which is how the number above
    # came to be measured at all.
    def _read(f):
        with np.load(f) as d:
            return (f, d["z"], d["lnk"], d["theta"], d["idx"], d["failed_idx"],
                    np.log(d["pm"]), np.log(d["pcb"]))

    t0 = time.monotonic()
    shards = [None] * n_files
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_read, f): i for i, f in enumerate(files)}
        for done, fut in enumerate(cf.as_completed(futures), 1):
            shards[futures[fut]] = fut.result()
            if done % 200 == 0 or done == n_files:
                dt = time.monotonic() - t0
                print(f"  {done}/{n_files} shards  {dt:6.1f} s  "
                      f"(eta {dt * (n_files - done) / done:6.1f} s)", flush=True)

    X, Ym, Ycb, idx, failed = [], [], [], [], []
    z = lnk = None
    for f, z_f, lnk_f, th, idx_f, failed_f, ln_pm, ln_pcb in shards:
        if z is None:
            z, lnk = z_f, lnk_f
        elif not (np.array_equal(z, z_f) and np.array_equal(lnk, lnk_f)):
            raise ValueError(f"{f.name} uses a different grid from {files[0].name}")
        failed.extend(failed_f.tolist())
        if len(idx_f) == 0:
            continue
        n = len(th)
        # (n, n_z) -> rows of (theta, z)
        X.append(np.concatenate(
            [np.repeat(th, len(z), axis=0), np.tile(z, n)[:, None]], axis=1))
        Ym.append(ln_pm.reshape(n * len(z), len(lnk)))
        Ycb.append(ln_pcb.reshape(n * len(z), len(lnk)))
        idx.extend(idx_f.tolist())

    X = np.concatenate(X).astype(dtype)
    Ym = np.concatenate(Ym).astype(dtype)
    Ycb = np.concatenate(Ycb).astype(dtype)
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # **Written in parts, for the same reason the shards are read in parallel.**
    # One 9.7 GB file is a single sequential read, and /bettik gives this
    # account a few MB/s, so `train.py` spent the better part of an hour in
    # `np.load` before printing anything -- once per job, and a besteffort job
    # restarts.  Split across `parts` files, a thread pool recovers it, exactly
    # as it did for the 3000 input shards.  The row order is the concatenation
    # of the parts in index order, so the design indices still line up.
    # Never more parts than rows: a smoke run with six rows should not write
    # thirty-two files, thirty of them empty.
    parts = max(1, min(int(parts), len(X)))
    bounds = np.linspace(0, len(X), parts + 1).astype(int)
    manifest = [f"{out.stem}.part{i:03d}.npz" for i in range(parts)]

    # Written in parallel as well as read in parallel.  The write is
    # latency-bound in the same way: measured on Dahu, the parts went out at
    # about 2 MB/s serially, which is 80 minutes for the set.  `np.savez`
    # releases the GIL over its I/O, so threads help here exactly as they do on
    # the read.
    def _write(i):
        a, b = bounds[i], bounds[i + 1]
        np.savez(out.with_name(manifest[i]),
                 X=X[a:b], ln_pm=Ym[a:b], ln_pcb=Ycb[a:b])

    t1 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=min(workers, parts)) as pool:
        for done, _ in enumerate(pool.map(_write, range(parts)), 1):
            if done % 8 == 0 or done == parts:
                print(f"  wrote {done}/{parts} parts  "
                      f"{time.monotonic() - t1:5.1f} s", flush=True)
    np.savez(out, z=z, lnk=lnk, parts=np.array(manifest),
             n_rows=np.array(len(X)),
             idx=np.array(sorted(idx), dtype=np.int64),
             failed_idx=np.array(sorted(set(failed)), dtype=np.int64))
    total = out.stat().st_size + sum(
        out.with_name(f).stat().st_size for f in manifest)
    print(f"{len(files)} shards -> {X.shape[0]} rows x {Ym.shape[1]} modes "
          f"from {len(idx)} cosmologies ({len(set(failed))} CLASS failures)",
          flush=True)
    print(f"wrote {out} + {len(manifest)} parts  ({total / 1e6:.1f} MB)",
          flush=True)
    return out


def load_training_set(dataset, workers=16):
    """Read a part-written training set back, in parallel.

    Returns ``(X, ln_pm, ln_pcb, lnk)``.  Falls back to the single-file layout
    so a single-file dataset still loads.
    """
    dataset = pathlib.Path(dataset)
    with np.load(dataset) as d:
        if "parts" not in d.files:                       # single-file layout
            return d["X"], d["ln_pm"], d["ln_pcb"], d["lnk"]
        names = [str(n) for n in d["parts"]]
        lnk, n_rows = d["lnk"], int(d["n_rows"])

    def _read(name):
        with np.load(dataset.with_name(name)) as pd:
            return pd["X"], pd["ln_pm"], pd["ln_pcb"]

    t0 = time.monotonic()
    got = [None] * len(names)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_read, n): i for i, n in enumerate(names)}
        for done, fut in enumerate(cf.as_completed(futs), 1):
            got[futs[fut]] = fut.result()
            if done % 8 == 0 or done == len(names):
                print(f"  loaded {done}/{len(names)} parts  "
                      f"{time.monotonic() - t0:5.1f} s", flush=True)
    X = np.concatenate([g[0] for g in got])
    Ym = np.concatenate([g[1] for g in got])
    Ycb = np.concatenate([g[2] for g in got])
    if len(X) != n_rows:
        raise ValueError(f"{dataset.name} says {n_rows} rows, its parts hold "
                         f"{len(X)} -- a part is missing or truncated")
    return X, Ym, Ycb, lnk


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("ratio", "emu"), required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--parts", type=int, default=32,
                    help="how many files the training set is split across; the\n"
                         "read is latency-bound, so parts are what makes it\n"
                         "parallel")
    ap.add_argument("--workers", type=int, default=16,
                    help="threads reading shards (emu mode); the read is\nlatency-bound on a shared filesystem, so this is worth more than it looks")
    a = ap.parse_args(argv)
    if a.mode == "ratio":
        build_ratio(a.shards, a.out)
    else:
        build_training_set(a.shards, a.out or "training_set.npz",
                           workers=a.workers, parts=a.parts)


if __name__ == "__main__":  # pragma: no cover
    main()
