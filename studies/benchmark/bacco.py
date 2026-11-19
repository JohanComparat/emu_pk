#!/usr/bin/env python
r"""BACCO's linear emulator in the comparison of Paper I, App. B (Fig. 15).

App. B follows Sui et al. (2025, Fig. 10): the mean absolute fractional error of
each linear emulator against CLASS, over 200 log-spaced k in [9e-3, 9] h/Mpc at
z = 0, at the four Quijote cosmologies (Fid, Mnu+, w+, w-), against the CPU time
of one evaluation at z = 1.  This adds the BACCO linear emulator (Aricò et al.
2021, ``baccoemu.Matter_powerspectrum().get_linear_pk``) to it.

The truths are not recomputed: they are read from the files ``benchmark.py``'s
``truth`` and ``ref`` stages wrote, so
BACCO is scored against exactly the spectra the other emulators were scored
against.  ``--truth-dir`` holds

``cosmologies.json``  the four cosmologies, with the A_s fixed by sigma8 through CLASS;
``truth.npz``         CLASS at its defaults, Sui et al.'s setup (``class/<name>/0``);
``ref.npz``           converged CLASS (pk_ref.pre; REF_NU at Mnu+).

BACCO takes ``omega_cold`` (cold matter, without the neutrinos), ``A_s``,
``omega_baryon``, ``hubble``, ``ns``, ``neutrino_mass``, ``w0``, ``wa`` and the
expansion factor, and returns the total-matter spectrum with ``cold=False``.
The timing protocol is App. B's: one core, one thread, one warm-up call, then
the mean over repeated calls of one cosmology at z = 1.

    taskset -c 0 python bacco.py --truth-dir <dir>   # env with baccoemu (TensorFlow)

writes ``results/bacco_benchmark.json`` and ``results/bacco_benchmark.npz``,
which ``benchmark.py plot`` reads.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse      # noqa: E402
import json          # noqa: E402
import platform      # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np   # noqa: E402

ROOT = Path(__file__).resolve().parent
K = np.logspace(np.log10(9e-3), np.log10(9.0), 200)        # h/Mpc, App. B's grid
Z_ERR, Z_TIME = 0.0, 1.0
NU_DENOM = 93.14       # App. B's Omega_nu h^2 = mnu / 93.14


def bacco_params(c: dict, z: float) -> dict:
    onu = c["mnu"] / NU_DENOM / c["h"] ** 2
    return {"omega_cold": c["Om"] - onu, "A_s": c["As"], "omega_baryon": c["Ob"],
            "hubble": c["h"], "ns": c["ns"], "neutrino_mass": c["mnu"],
            "w0": c["w0"], "wa": c["wa"], "expfactor": 1.0 / (1.0 + z)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth-dir", type=Path, required=True)
    ap.add_argument("--n-time", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bacco_benchmark")
    a = ap.parse_args(argv)

    import baccoemu
    cos = json.load(open(a.truth_dir / "cosmologies.json"))
    truth = np.load(a.truth_dir / "truth.npz")
    ref = np.load(a.truth_dir / "ref.npz")
    assert np.allclose(truth["k"], K, rtol=1e-12), "truth k grid is not App. B's"

    emu = baccoemu.Matter_powerspectrum()

    def pk(c, z):
        _, p = emu.get_linear_pk(k=K, cold=False, **bacco_params(c, z))
        return np.asarray(p, dtype=float)

    spectra, err_default, err_ref = {}, {}, {}
    for name, c in cos.items():
        p = pk(c, Z_ERR)
        spectra[name] = p
        err_default[name] = float(np.mean(np.abs(p / truth[f"class/{name}/0"] - 1)))
        err_ref[name] = float(np.mean(np.abs(p / ref[name] - 1)))
        print(f"{name:5s} vs CLASS default {err_default[name]:.2e}   "
              f"vs converged CLASS {err_ref[name]:.2e}", flush=True)

    c = cos["Fid"]
    pk(c, Z_TIME)                                     # warm-up
    t0 = time.perf_counter()
    for _ in range(a.n_time):
        pk(c, Z_TIME)
    t_cpu = (time.perf_counter() - t0) / a.n_time
    print(f"CPU time per evaluation at z = 1: {t_cpu * 1e3:.3f} ms", flush=True)

    rec = {"emulator": "BACCO linear (baccoemu.Matter_powerspectrum.get_linear_pk)",
           "baccoemu": getattr(baccoemu, "__version__", "?"),
           "metric": "mean |P/P_truth - 1| over 200 log k in [9e-3, 9] h/Mpc, z = 0",
           "err_vs_class_default": err_default, "err_vs_class_converged": err_ref,
           "cpu_time_s": t_cpu, "n_time": a.n_time,
           "host": {"node": platform.node(), "cpu": next(
               (l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                if l.startswith("model name")), "?"),
               "affinity": sorted(os.sched_getaffinity(0)),
               "loadavg": os.getloadavg()}}
    try:
        import tensorflow as tf
        rec["tensorflow"] = tf.__version__
    except Exception:                                  # pragma: no cover
        pass
    a.out.parent.mkdir(exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    np.savez(a.out.with_suffix(".npz"), k=K, **spectra)
    print(f"wrote {a.out}.json/.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
