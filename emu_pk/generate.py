r"""Run CLASS.  One shard of work per process, one ``.npz`` per shard.

This is the only module that needs a Boltzmann solver, and the only one that
runs on the cluster.  Two products, from the same driver:

``--mode ratio``
    the Phase-1 correction grid: the fiducial cosmology, swept over
    ``(sum_mnu, w0, wa)``, every redshift in one solve.
``--mode emu``
    the Phase-2 training set: a Latin-hypercube slice of the eight-parameter
    box, every redshift in one solve.

**Shards skip if their output exists.**  That single property is what makes a
besteffort OAR job resumable: a killed array element re-runs and costs only the
work it had not finished, rather than redoing the shard from the top.  It is
also what makes a production run restartable after a walltime kill without any
bookkeeping about which elements landed.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from . import box, cosmo, grid

__all__ = ["solve", "ratio_shard", "emu_shard", "main"]


def solve(params: dict, z_nodes, k_h) -> tuple:
    """One CLASS solve.  Returns ``(P_m, P_cb)`` in (Mpc/h)^3 on ``k_h`` [h/Mpc].

    Both spectra always, and ``P_cb`` is a *copy* of ``P_m`` at zero neutrino
    mass rather than a separate CLASS call: with no massive species the cold
    field and the total field are the same field, and ``pk_cb_lin`` is not
    defined there.
    """
    from classy import Class

    h = params["h"]
    cl = Class()
    cl.set(params)
    try:
        cl.compute()
        k_phys = np.asarray(k_h) * h                      # h/Mpc -> 1/Mpc
        h3 = h ** 3
        z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
        pm = np.array([[cl.pk_lin(kk, zz) for kk in k_phys]
                       for zz in z_nodes]) * h3
        if params.get("N_ncdm", 0):
            pcb = np.array([[cl.pk_cb_lin(kk, zz) for kk in k_phys]
                            for zz in z_nodes]) * h3
        else:
            pcb = pm.copy()
    finally:
        cl.struct_cleanup()
        cl.empty()
    return pm, pcb


# ==========================================================================
# Phase 1 -- the correction grid
# ==========================================================================
def _ratio_design():
    """Every ``(sum_mnu, w0, wa)`` node, flattened, plus the reference index.

    The LambdaCDM massless corner is index 0 by construction: it is the
    denominator of every ratio, and putting it first means a shard that has run
    at all has run the one point everything else divides by.
    """
    pts = [(0.0, -1.0, 0.0)]
    for mnu in grid.MNU_NODES:
        for w0 in grid.W0_NODES:
            for wa in grid.WA_NODES:
                if (float(mnu), float(w0), float(wa)) != (0.0, -1.0, 0.0):
                    pts.append((float(mnu), float(w0), float(wa)))
    return pts


def ratio_shard(index: int, n_per_shard: int, out_dir) -> pathlib.Path:
    """Solve ``n_per_shard`` nodes of the correction grid, starting at ``index``."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ratio_{index:05d}.npz"
    if out.exists():
        print(f"  shard {index}: exists, skipped")
        return out

    design = _ratio_design()
    lo = index * n_per_shard
    chunk = design[lo:lo + n_per_shard]
    if not chunk:
        print(f"  shard {index}: past the end of the design ({len(design)}), nothing to do")
        return out

    k = grid.k_grid()
    z = grid.Z_NODES_RATIO
    fid = cosmo.PLANCK18
    pm_all, pcb_all, rows = [], [], []
    for n, (mnu, w0, wa) in enumerate(chunk):
        # Omega_m contains the neutrinos, so the cold sector shrinks as the mass
        # grows.  This is ggah_mod's convention and the table has to be built in
        # it, or the ratio is taken between two different cosmologies.
        om_nu = cosmo.omega_nu(mnu, fid["h"])
        o_cdm = fid["Omega_m"] - om_nu - fid["Omega_b"]
        t0 = time.time()
        pm, pcb = solve(cosmo.class_params(
            h=fid["h"], omega_b=fid["Omega_b"] * fid["h"] ** 2,
            omega_cdm=o_cdm * fid["h"] ** 2, n_s=fid["n_s"],
            ln10A_s=fid["ln10A_s"], sum_mnu=mnu, w0=w0, wa=wa,
            k_max_h=grid.K_MAX, z_max=grid.Z_MAX), z, k)
        pm_all.append(pm)
        pcb_all.append(pcb)
        rows.append((mnu, w0, wa, cosmo.f_nu(mnu, fid["h"], fid["Omega_m"])))
        print(f"  [{lo + n:4d}/{len(design)}] mnu={mnu:.3f} w0={w0:+.2f} "
              f"wa={wa:+.2f}  {time.time() - t0:6.1f} s", flush=True)

    np.savez_compressed(out, theta=np.array(rows), z=z, lnk=np.log(k),
                        pm=np.array(pm_all, dtype=np.float64),
                        pcb=np.array(pcb_all, dtype=np.float64),
                        h_fid=fid["h"], Omega_m_fid=fid["Omega_m"])
    return out


# ==========================================================================
# Phase 2 -- the training set
# ==========================================================================
#: Solves per output file -- not per array element, since an element writes
#: several.
#:
#: A shard that writes only when it finishes loses everything if it is killed,
#: and the emu family runs under OAR ``besteffort``, which means it *will* be
#: killed whenever a paying job wants the node.  At 6.5 s a solve, chunks of 50
#: cap that loss at about five minutes' work, and every chunk that did land is
#: skipped on the restart.
CHUNK = 50

#: A solve slower than this is printed with a marker.  Not a limit -- CLASS is C
#: and cannot be interrupted from Python mid-solve -- but the thing that says
#: whether the walltime is sized for the design or for the calibration sample.
SLOW_SOLVE_S = 20.0


def emu_shard(index: int, n_per_shard: int, out_dir, n_total: int,
              seed: int = 20260827, chunk: int = CHUNK) -> list:
    """Solve design points ``[index*n : (index+1)*n)``, writing one file per chunk.

    The design is regenerated from the seed rather than read from a file, so a
    chunk is reproducible from its indices alone and two workers can never
    disagree about which cosmology index *i* names.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    design = box.sample(n_total, seed=seed)
    lo = index * n_per_shard
    hi = min(lo + n_per_shard, n_total)
    if lo >= n_total:
        print(f"  shard {index}: past the end of the design ({n_total}), nothing to do")
        return []

    k, z = grid.k_grid(), grid.Z_NODES_EMU
    written = []
    for c0 in range(lo, hi, chunk):
        c1 = min(c0 + chunk, hi)
        out = out_dir / f"emu_{index:05d}_{c0:07d}.npz"
        if out.exists():
            print(f"  chunk {c0}-{c1}: exists, skipped", flush=True)
            written.append(out)
            continue
        keep, pm_all, pcb_all, failed = [], [], [], []
        t_chunk = time.time()
        for n in range(c0, c1):
            d = dict(zip(box.PARAMS, design[n]))
            t_solve = time.time()
            try:
                pm, pcb = solve(cosmo.class_params(
                    h=d["h"], omega_b=d["omega_b"], omega_cdm=d["omega_cdm"],
                    n_s=d["n_s"], ln10A_s=d["ln10A_s"], sum_mnu=d["sum_mnu"],
                    w0=d["w0"], wa=d["wa"], k_max_h=grid.K_MAX,
                    z_max=grid.Z_MAX), z, k)
            except Exception as e:
                # A failure is data, not a reason to abort: CLASS refuses some
                # corners of any wide box, and a run that dies on the first one
                # reports nothing.  The index and the reason are recorded so
                # the gaps are countable rather than invisible.
                failed.append((n, f"{type(e).__name__}: {e}"[:200]))
                print(f"    [{n:7d}] FAILED {type(e).__name__}", flush=True)
                continue
            keep.append(n)
            pm_all.append(pm.astype(np.float32))
            pcb_all.append(pcb.astype(np.float32))
        # Written to a temporary name and renamed, so a kill during the write
        # leaves no half-file for the audit to find and the assembler to trip
        # over.  Rename within one filesystem is atomic.
        #
        # The temporary name has to *end* in ``.npz``: `np.savez_compressed`
        # appends the extension itself when the path lacks it, so a `.part`
        # suffix produced `....npz.part.npz` on disk and the rename then looked
        # for a `....npz.part` that had never existed.  Caught by the devel
        # smoke, one array element before three hundred would have hit it.
        tmp = out.with_name(out.stem + ".part.npz")
        np.savez_compressed(
            tmp,
            idx=np.array(keep, dtype=np.int64),
            theta=(design[keep] if keep
                   else np.zeros((0, len(box.PARAMS)), dtype=float)),
            z=z, lnk=np.log(k),
            pm=np.array(pm_all, dtype=np.float32).reshape(len(keep), len(z), len(k)),
            pcb=np.array(pcb_all, dtype=np.float32).reshape(len(keep), len(z), len(k)),
            failed_idx=np.array([f[0] for f in failed], dtype=np.int64),
            failed_why=np.array([f[1] for f in failed], dtype="U200"),
        )
        tmp.rename(out)
        written.append(out)
        print(f"  chunk {c0}-{c1}: {len(keep)} solved, {len(failed)} failed, "
              f"{time.time() - t_chunk:.0f} s -> {out.name}", flush=True)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("ratio", "emu", "time"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-per-shard", type=int, default=25)
    ap.add_argument("--n-total", type=int, default=100_000,
                    help="size of the emu design (ignored for --mode ratio)")
    ap.add_argument("--out", default="shards")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="solves per output file; caps what a besteffort kill loses")
    a = ap.parse_args(argv)

    if a.mode == "ratio":
        print(f"ratio grid: {len(_ratio_design())} nodes, shard {a.shard}")
        ratio_shard(a.shard, a.n_per_shard, a.out)
    elif a.mode == "emu":
        emu_shard(a.shard, a.n_per_shard, a.out, a.n_total, a.seed, a.chunk)
    else:
        _time_calibration(a.n_per_shard, a.seed)


def _time_calibration(n: int = 8, seed: int = 20260827):
    """Time CLASS at production settings.  Sizes the run; guesses do not.

    Prints seconds per solve over a random slice of the *actual* design, so the
    number covers the expensive corners (high k_max, massive neutrinos, CPL)
    rather than the fiducial alone.
    """
    import resource
    k, z = grid.k_grid(), grid.Z_NODES_EMU
    design = box.sample(max(n * 8, 64), seed=seed)[:n]
    times, fails = [], 0
    for i, theta in enumerate(design):
        d = dict(zip(box.PARAMS, theta))
        t0 = time.time()
        try:
            solve(cosmo.class_params(
                h=d["h"], omega_b=d["omega_b"], omega_cdm=d["omega_cdm"],
                n_s=d["n_s"], ln10A_s=d["ln10A_s"], sum_mnu=d["sum_mnu"],
                w0=d["w0"], wa=d["wa"], k_max_h=grid.K_MAX, z_max=grid.Z_MAX),
                z, k)
        except Exception as e:
            fails += 1
            print(f"  [{i}] FAILED {type(e).__name__}: {e}"[:160], flush=True)
            continue
        times.append(time.time() - t0)
        print(f"  [{i}] {times[-1]:6.2f} s   mnu={d['sum_mnu']:.3f} "
              f"w0={d['w0']:+.2f} wa={d['wa']:+.2f} h={d['h']:.3f}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t = np.array(times)
    print("\n=== CLASS calibration ===")
    print(f"  solves        {len(t)} ok, {fails} failed")
    if len(t):
        print(f"  seconds/solve mean {t.mean():.2f}  median {np.median(t):.2f}  "
              f"min {t.min():.2f}  max {t.max():.2f}")
        print(f"  core-hours per 1e5 solves: {t.mean() * 1e5 / 3600:.0f}")
    print(f"  peak RSS      {rss:.0f} MB")
    print(f"  k grid        {len(k)} modes to {grid.K_MAX} h/Mpc")
    print(f"  z rows        {len(z)} per solve")


if __name__ == "__main__":  # pragma: no cover
    main()
