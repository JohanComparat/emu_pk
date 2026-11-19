"""Linear P(k) emulators: accuracy against run time, after Sui et al. (2025), Fig. 10.

SYREN-NEW (Sui, Bartlett, Pandey et al. 2025, A&A 698, A1) compares linear
power-spectrum emulators by the mean absolute fractional error against CLASS,
averaged over 200 log-spaced k in [9e-3, 9] h/Mpc at z = 0, at the four Quijote
cosmologies Fid, Mnu+, w+ and w-, against the time per evaluation on a CPU and on
a GPU (batch of 2000).  This script redoes that comparison with emu_pk in it;
it draws Fig. 15 of Paper I (App. "Linear power-spectrum emulators").

Two truths, one per row of the figure.  Top: CLASS at its default precision,
set up exactly as the SYREN-NEW example notebook does (three degenerate
neutrinos, N_ur = 0.00641, tau = 0.0561, Omega_m including the neutrinos), so
SYREN-NEW is scored against what it was fitted to, as in the paper.  Bottom:
the same cosmologies solved by converged CLASS, pk_ref.pre for Fid, w+ and w-
and REF_NU for Mnu+; CLASS at its default precision is then one more code.

Emulators, each applied only inside its parameter space as in the paper:
  emu_pk 2.1.2 (all four; 2.1.0 to 2.1.2 ship the same weights), SYREN-NEW (all four; numpy as published, and a JAX
  port made here by source substitution, checked against the numpy version),
  SYREN LCDM (Bartlett et al. 2024a; Fid), Eisenstein & Hu 1998 (Fid),
  CosmoPower mpk_lin as shipped by cosmopower_jax (Fid), CAMB at its default
  precision and at emu_pk's training precision (all four).
CosmoPower's TensorFlow original, the BACCO emulator and Orjuela-Quintana et al.
(2024), which the paper also shows, are not installed here.

CPU timings: one core (taskset), one thread, mean over repeated calls of one
cosmology at z = 1 (as in the paper), after a warm-up; JAX functions jitted.
GPU timings: JAX float32, a batch of 2000 cosmologies, median of 20 batches
divided by 2000.

Stages; each writes into --cache (default: results/ beside this script, which
holds the results of the run behind Paper I, so `plot` alone redraws the figure):
  python benchmark.py truth                     # CLASS, CAMB; about 2 min
  python benchmark.py ref                       # converged CLASS; about 11 min
  taskset -c 2 python benchmark.py cpu          # one core
  python benchmark.py gpu --cp-path <dir>       # in a GPU JAX environment
  python benchmark.py plot
Requirements: emu_pk[gen] (CAMB 1.6.6, classy 3.3.4), matplotlib,
cosmopower_jax (cpu, gpu), and a git clone of symbolic_pofk
(https://github.com/DeaglanBartlett/symbolic_pofk, commit cb4d2bb), given by
--syren or the environment variable SYMBOLIC_POFK.
`ref` reads CLASS's pk_ref.pre precision settings from class_pk_ref.json.
"""
import argparse
import ast
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
CACHE = os.path.join(HERE, "results")            # reset by --cache
SYREN = os.environ.get("SYMBOLIC_POFK", "")      # reset by --syren

K = np.logspace(np.log10(9e-3), np.log10(9.0), 200)        # h/Mpc
Z_ERR, Z_TIME = 0.0, 1.0
N_BATCH = 2000

# Quijote: Planck-like base; Fid, w+ and w- at sigma8 = 0.834, Mnu+ at the A_s
# of Fid (Villaescusa-Navarro et al. 2020).
BASE = dict(Om=0.3175, Ob=0.049, h=0.6711, ns=0.9624)
SIGMA8 = 0.834
COSMOS = {"Fid": dict(mnu=0.0, w0=-1.0, wa=0.0),
          "Mnu+": dict(mnu=0.1, w0=-1.0, wa=0.0),
          "w+": dict(mnu=0.0, w0=-0.95, wa=0.0),
          "w-": dict(mnu=0.0, w0=-1.05, wa=0.0)}
NU_DENOM = 93.14                       # SYREN-NEW's Omega_nu h^2 = mnu / 93.14


def omega_cdm(c):
    """Physical cold dark matter density, the same for every code."""
    onu = c["mnu"] / NU_DENOM / c["h"] ** 2
    return (c["Om"] - c["Ob"] - onu) * c["h"] ** 2


# ----------------------------------------------------------------- truth
def class_syren(c, z, extra=None):
    """CLASS as in symbolic_pofk/examples/syren_new_example.ipynb, P in (Mpc/h)^3.

    mnu = 0 is solved with three massless species (N_ur = 3.046) rather than
    three ncdm of zero mass, which CLASS refuses.
    """
    from classy import Class
    h = c["h"]
    p = {"h": h, "omega_b": c["Ob"] * h ** 2, "omega_cdm": omega_cdm(c),
         "A_s": c["As"], "n_s": c["ns"], "output": "mPk",
         "P_k_max_1/Mpc": K.max() * h, "w0_fld": c["w0"], "wa_fld": c["wa"],
         "Omega_Lambda": 0, "tau_reio": 0.0561, "z_max_pk": 3.0}
    if c["mnu"] > 0:
        p.update(N_ur=0.00641, N_ncdm=3, m_ncdm=",".join([str(c["mnu"] / 3.0)] * 3))
    else:
        p.update(N_ur=3.046)
    p.update(extra or {})
    cl = Class()
    cl.set(p)
    cl.compute()
    pk = np.array([cl.pk_lin(kk * h, z) for kk in K]) * h ** 3
    s8 = cl.sigma8()
    cl.struct_cleanup()
    cl.empty()
    return pk, s8


def camb_pk(c, z, precision):
    import camb
    from emu_pk import cosmo
    pars = cosmo.camb_params(h=c["h"], omega_b=c["Ob"] * c["h"] ** 2,
                             omega_cdm=omega_cdm(c), n_s=c["ns"],
                             ln10A_s=np.log(1e10 * c["As"]), sum_mnu=c["mnu"],
                             w0=c["w0"], wa=c["wa"], k_max_h=10.0,
                             redshifts=(z,), precision=precision)
    res = camb.get_results(pars)
    kh, _, pk = res.get_matter_power_spectrum(minkh=K[0] * 0.9, maxkh=K[-1] * 1.1,
                                              npoints=800)
    return np.exp(np.interp(np.log(K), np.log(kh), np.log(pk[0])))


def cosmologies():
    """Every test cosmology with its A_s, set by sigma8 through the truth."""
    path = os.path.join(CACHE, "cosmologies.json")
    if os.path.exists(path):
        return json.load(open(path))
    out = {}
    for name, extra in COSMOS.items():
        if name == "Mnu+":
            continue
        c = dict(BASE, **extra, As=2.1e-9)
        for _ in range(4):
            _, s8 = class_syren(c, 0.0)
            c["As"] *= (SIGMA8 / s8) ** 2
        c["sigma8"] = class_syren(c, 0.0)[1]
        out[name] = c
    c = dict(BASE, **COSMOS["Mnu+"], As=out["Fid"]["As"])
    c["sigma8"] = class_syren(c, 0.0)[1]
    out["Mnu+"] = c
    out = {n: out[n] for n in COSMOS}
    json.dump(out, open(path, "w"), indent=1)
    return out


def timed(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def stage_truth(a):
    from emu_pk import cosmo
    cos = cosmologies()
    res = {"k": K}
    tim = {}
    for name, c in cos.items():
        for z in (Z_ERR, Z_TIME):
            res[f"class/{name}/{z:g}"] = class_syren(c, z)[0]
            res[f"camb/{name}/{z:g}"] = camb_pk(c, z, {})
            res[f"camb_hi/{name}/{z:g}"] = camb_pk(c, z, cosmo.CAMB_PRECISION)
        print(name, "sigma8", c["sigma8"], "As", c["As"], flush=True)
    c = cos["Fid"]
    tim["CLASS"] = timed(lambda: class_syren(c, Z_TIME), 3)
    tim["CAMB"] = timed(lambda: camb_pk(c, Z_TIME, {}), 3)
    tim["CAMB (emu_pk precision)"] = timed(
        lambda: camb_pk(c, Z_TIME, cosmo.CAMB_PRECISION), 2)
    print("timings", tim, flush=True)
    np.savez(os.path.join(CACHE, "truth.npz"), **res)
    json.dump(tim, open(os.path.join(CACHE, "timing_truth.json"), "w"), indent=1)


#: CLASS without the ncdm fluid approximation and at pk_ref's perturbation and
#: momentum tolerances: the reference at Mnu+, where a full pk_ref solve with
#: massive neutrinos takes hours (about 10 min on one core).
REF_NU = {"ncdm_fluid_approximation": 3, "tol_perturbations_integration": 1e-6,
          "tol_ncdm_synchronous": 1e-10, "l_max_ncdm": 50, "q_linstep": 0.2,
          "q_logstep_spline": 20.0}


def stage_ref(a):
    """Converged CLASS at z = 0: pk_ref.pre for Fid, w+, w-; REF_NU for Mnu+."""
    CLASS_REF = json.load(open(os.path.join(HERE, "class_pk_ref.json")))
    cos = cosmologies()
    out = {}
    for name, c in cos.items():
        cached = os.path.join(CACHE, "class_mnu_nofluid+tol.npy")
        if name == "Mnu+":
            out[name] = (np.load(cached) if os.path.exists(cached)
                         else class_syren(c, Z_ERR, REF_NU)[0])
        else:
            out[name] = class_syren(c, Z_ERR, dict(CLASS_REF))[0]
        print(name, flush=True)
    np.savez(os.path.join(CACHE, "ref.npz"), **out)


# -------------------------------------------------------------- emulators
def syren_numpy():
    if not os.path.isdir(os.path.join(SYREN, "symbolic_pofk")):
        sys.exit("give the symbolic_pofk clone with --syren or SYMBOLIC_POFK")
    sys.path.insert(0, SYREN)
    from symbolic_pofk import linear, linear_new
    return linear, linear_new


def syren_jax():
    """SYREN-NEW's linear formula in JAX: the published source, np -> jax.numpy.

    The LCDM correction logF of Bartlett et al. (2024a) is taken from
    symbolic_pofk/linear.py without its out-of-range branch (k is inside
    [9e-3, 9] h/Mpc here), and the one Python branch on mnu becomes jnp.where.
    """
    import jax.numpy as jnp
    pkg = os.path.join(SYREN, "symbolic_pofk")
    src_l = open(os.path.join(pkg, "linear.py")).read()
    fn = next(n for n in ast.parse(src_l).body
              if isinstance(n, ast.FunctionDef) and n.name == "logF_fiducial")
    body = ast.get_source_segment(src_l, fn)
    cut = body.index("    # Use Bartlett et al. 2023 P(k) only in tested regime")
    lcdm = (body[:cut] + "    return logF\n").replace(
        "def logF_fiducial(", "def lcdm_logF_fiducial(")
    src = open(os.path.join(pkg, "linear_new.py")).read()
    src = src.replace("import numpy as np\n", "")
    src = src.replace("from .linear import logF_fiducial as lcdm_logF_fiducial\n", "")
    assert src.count("Nnu = (3 if mnu != 0.0 else 0)") == 1
    src = src.replace("Nnu = (3 if mnu != 0.0 else 0)", "Nnu = np.where(mnu != 0.0, 3, 0)")
    ns = {"np": jnp, "warnings": warnings}
    exec(compile(lcdm + "\n" + src, "syren_new_jax", "exec"), ns)
    return ns["plin_new_emulated"]


def emu_theta(c):
    return np.array([c["Ob"] * c["h"] ** 2, omega_cdm(c), c["h"], c["ns"],
                     np.log(1e10 * c["As"]), c["mnu"], c["w0"], c["wa"],
                     0.0, 1 / 3, 1 / 3])


def cp_theta(c, z):
    return np.array([c["Ob"] * c["h"] ** 2, omega_cdm(c), c["h"], c["ns"],
                     np.log(1e10 * c["As"]), z])


def cosmopower():
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
    return CosmoPowerJAX(probe="mpk_lin")


def cp_to_h(pk_mpc, modes, h):
    """CosmoPower's P in Mpc^3 on k in 1/Mpc, onto K in h/Mpc, (Mpc/h)^3."""
    return np.exp(np.interp(np.log(K), np.log(np.asarray(modes) / h),
                            np.log(np.asarray(pk_mpc) * h ** 3)))


def emulator_calls(cos, z):
    """name -> {cosmology: callable returning P(K) in (Mpc/h)^3 at z}."""
    import jax
    from emu_pk import PkEmulator
    linear, linear_new = syren_numpy()
    plin_jax = jax.jit(syren_jax())
    emu = PkEmulator()
    kj = jax.numpy.asarray(K)
    emu_jit = jax.jit(lambda th: emu.pk(kj, z, th))
    cp = cosmopower()
    cp_jit = jax.jit(cp.predict)
    a = 1.0 / (1.0 + z)
    calls = {"emu_pk": {}, "SYREN-NEW": {}, "SYREN-NEW (JAX)": {},
             "SYREN LCDM": {}, "Eisenstein & Hu": {}, "CosmoPower": {}}
    for name, c in cos.items():
        th = emu_theta(c)
        emu.pk(K, z, th)                          # box check, outside the jit
        calls["emu_pk"][name] = lambda th=th: emu_jit(th).block_until_ready()
        args = (K, 1e9 * c["As"], c["Om"], c["Ob"], c["h"], c["ns"], c["mnu"],
                c["w0"], c["wa"], a)
        calls["SYREN-NEW"][name] = lambda args=args: linear_new.plin_new_emulated(*args)
        calls["SYREN-NEW (JAX)"][name] = (
            lambda args=args: plin_jax(*args).block_until_ready())
        if name != "Fid":
            continue
        lc = (K, c["sigma8"], c["Om"], c["Ob"], c["h"], c["ns"])
        calls["SYREN LCDM"][name] = lambda lc=lc: linear.plin_emulated(*lc, a=a)
        # E&H is the z = 0 spectrum only; it is timed at z = 0
        calls["Eisenstein & Hu"][name] = lambda lc=lc: linear.pk_EisensteinHu_b(*lc)
        x = cp_theta(c, z)[None, :]
        calls["CosmoPower"][name] = (
            lambda x=x, h=c["h"]: cp_to_h(np.atleast_2d(cp_jit(x))[0], cp.modes, h))
    return calls


def stage_cpu(a):
    import jax
    jax.config.update("jax_enable_x64", True)
    cos = cosmologies()
    truth = np.load(os.path.join(CACHE, "truth.npz"))
    warnings.simplefilter("ignore")
    ref = np.load(os.path.join(CACHE, "ref.npz"))
    pk = {}
    for name, per in emulator_calls(cos, Z_ERR).items():
        pk[name] = {c: np.asarray(f()) for c, f in per.items()}
    for code, lab in (("camb", "CAMB"), ("camb_hi", "CAMB (emu_pk precision)"),
                      ("class", "CLASS")):
        pk[lab] = {c: truth[f"{code}/{c}/0"] for c in cos}

    def score(tr):
        return {n: {c: float(np.mean(np.abs(p / tr[c] - 1))) for c, p in per.items()}
                for n, per in pk.items()}

    err = score({c: truth[f"class/{c}/0"] for c in cos})
    err.pop("CLASS")
    err_ref = score({c: ref[c] for c in cos})
    # the JAX port reproduces the numpy formula
    plin_jax = syren_jax()
    linear, linear_new = syren_numpy()
    port = max(float(np.max(np.abs(np.asarray(plin_jax(K, 1e9 * c["As"], c["Om"],
                c["Ob"], c["h"], c["ns"], c["mnu"], c["w0"], c["wa"], 1.0)) /
                linear_new.plin_new_emulated(K, 1e9 * c["As"], c["Om"], c["Ob"],
                c["h"], c["ns"], c["mnu"], c["w0"], c["wa"], 1.0) - 1)))
               for c in cos.values())
    tim = {}
    for name, per in emulator_calls(cos, Z_TIME).items():
        n = 20 if name == "SYREN LCDM" else 1000
        tim[name] = float(np.median([timed(f, n) for f in per.values()]))
    out = {"error_z0": err, "error_ref_z0": err_ref, "time_cpu_z1": tim, "jax_port_max_rel_diff": port,
           "cpu": open("/proc/cpuinfo").read().split("model name")[1]
           .split("\n")[0].strip(": \t")}
    json.dump(out, open(os.path.join(CACHE, "cpu.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def stage_gpu(a):
    import jax
    import jax.numpy as jnp
    from emu_pk import PkEmulator
    if a.cp_path:
        sys.path.append(a.cp_path)
    cos = cosmologies()
    rng = np.random.default_rng(1)
    pick = rng.integers(0, len(cos), N_BATCH)
    cl = list(cos.values())
    kj = jnp.asarray(K)

    emu = PkEmulator()
    th = jnp.asarray(np.array([emu_theta(cl[i]) for i in pick]))
    f_emu = jax.jit(jax.vmap(lambda t: emu.pk(kj, Z_TIME, t)))

    plin = syren_jax()
    cols = np.array([[1e9 * c["As"], c["Om"], c["Ob"], c["h"], c["ns"], c["mnu"],
                      c["w0"], c["wa"]] for c in cl])[pick]
    sy = [jnp.asarray(cols[:, j])[:, None] for j in range(8)]
    f_sy = jax.jit(lambda *p: plin(kj[None, :], *p, 1.0 / (1.0 + Z_TIME)))

    cp = cosmopower()
    xcp = jnp.asarray(np.array([cp_theta(cl[0], Z_TIME)] * N_BATCH))
    f_cp = jax.jit(cp.predict)

    def bench(f, *x):
        f(*x).block_until_ready()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            f(*x).block_until_ready()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) / len(x[0])

    def big(x):
        return jnp.tile(x, (10,) + (1,) * (x.ndim - 1))

    tim = {"emu_pk": bench(f_emu, th), "SYREN-NEW (JAX)": bench(f_sy, *sy),
           "CosmoPower": bench(f_cp, xcp)}
    tim_big = {"emu_pk": bench(f_emu, big(th)),
               "SYREN-NEW (JAX)": bench(f_sy, *[big(v) for v in sy]),
               "CosmoPower": bench(f_cp, big(xcp))}
    out = {"time_gpu_z1": tim, "time_gpu_z1_batch20000": tim_big, "device": str(jax.devices()[0].device_kind),
           "dtype": str(f_emu(th).dtype)}
    json.dump(out, open(os.path.join(CACHE, "gpu.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ plot
COLOURS = {"emu_pk": "#1f6f8b", "SYREN-NEW": "#d4a017", "SYREN LCDM": "#7a5c00",
           "CosmoPower": "#c0392b", "Eisenstein & Hu": "#8c8c8c", "CAMB": "#8e44ad",
           "CAMB (emu_pk precision)": "#d291e6", "CLASS": "#2e8b57"}
LABELS = {"SYREN LCDM": r"SYREN $\Lambda$CDM", "CAMB (emu_pk precision)":
          "CAMB, emu_pk precision", "CLASS": "CLASS, default precision"}
MARKERS = {"Fid": "o", "Mnu+": "s", "w+": "^", "w-": "v"}
MLAB = {"Fid": "Fid", "Mnu+": r"$M_\nu^+$", "w+": r"$w^+$", "w-": r"$w^-$"}


def stage_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"savefig.bbox": "tight"})
    cpu = json.load(open(os.path.join(CACHE, "cpu.json")))
    gpu = json.load(open(os.path.join(CACHE, "gpu.json")))
    ttr = json.load(open(os.path.join(CACHE, "timing_truth.json")))
    tcpu = dict(cpu["time_cpu_z1"], **ttr)
    # the numpy SYREN-NEW on the CPU (filled), its JAX port on the GPU
    tcpu.pop("SYREN-NEW (JAX)")
    tgpu = dict(gpu["time_gpu_z1"])
    tgpu["SYREN-NEW"] = tgpu.pop("SYREN-NEW (JAX)")
    tbig = dict(gpu["time_gpu_z1_batch20000"])
    tbig["SYREN-NEW"] = tbig.pop("SYREN-NEW (JAX)")
    rows = ((cpu["error_z0"], r"against CLASS, default precision"),
            (cpu["error_ref_z0"], r"against converged CLASS"))
    # Paper I's figure, at the paper's 7.1 in width, and a PNG for the README
    styles = (dict(out=[os.path.join(OUT, "emulator_benchmark.pdf"),
                        os.path.join(OUT, "emulator_benchmark.png")],
                   size=(7.1, 5.6), fs=9, small=8.5, tiny=7, leg=7.5, ncol=5, s=26),)
    for st in styles:
        plt.rcParams.update({"font.size": st["fs"]})
        fig, ax = plt.subplots(2, 2, figsize=st["size"], sharey=True, sharex="col",
                               gridspec_kw={"wspace": 0.05, "hspace": 0.08})
        for r, (err, lab) in enumerate(rows):
            for i, times in enumerate((tcpu, tgpu)):
                for name, t in times.items():
                    for c, e in err.get(name, {}).items():
                        ax[r, i].scatter(t, 100 * e, marker=MARKERS[c], s=st["s"],
                                         zorder=3, facecolor=COLOURS[name],
                                         edgecolor="k", lw=0.7)
                ax[r, i].set_xscale("log")
                ax[r, i].grid(alpha=0.3)
            for name, t in tbig.items():
                for c, e in err.get(name, {}).items():
                    ax[r, 1].scatter(t, 100 * e, marker=MARKERS[c], s=st["s"], zorder=3,
                                     facecolor="white", edgecolor=COLOURS[name], lw=1.2)
            ax[r, 0].set_ylabel(r"mean $|\Delta P/P|$ [%]" + "\n" + lab,
                                fontsize=st["small"])
        ax[0, 0].set_yscale("log")
        ax[0, 0].axvline(ttr["CLASS"], color=COLOURS["CLASS"], ls=":", lw=1)
        ax[0, 0].text(ttr["CLASS"] * 0.75, 0.96, "CLASS\n(truth)", ha="right",
                      va="top", fontsize=st["tiny"],
                      transform=ax[0, 0].get_xaxis_transform())
        ax[0, 0].set_title("CPU, one core", fontsize=st["small"])
        ax[0, 1].set_title("GPU, batch of 2000 (filled)\nand 20 000 (open)",
                           fontsize=st["small"])
        ax[1, 0].set_xlabel("time per evaluation [s]")
        ax[1, 1].set_xlabel("time per evaluation [s]")
        ax[1, 1].set_xlim(1e-8, 1e-6)
        ax[1, 1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        hand = [Patch(color=COLOURS[n], label=LABELS.get(n, n)) for n in COLOURS]
        hand += [Line2D([], [], ls="", marker=m, mfc="white", mec="k", label=MLAB[c])
                 for c, m in MARKERS.items()]
        fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, 0.03),
                   ncol=st["ncol"], fontsize=st["leg"], frameon=False,
                   columnspacing=1.0, handlelength=1.2)
        for out in st["out"]:
            fig.savefig(out, dpi=150)
        plt.close(fig)
    summary = {"error_class_default_z0": cpu["error_z0"],
               "error_class_converged_z0": cpu["error_ref_z0"],
               "time_cpu_z1": tcpu, "time_gpu_z1_batch2000": tgpu,
               "time_gpu_z1_batch20000": tbig,
               "cpu": cpu["cpu"], "gpu": gpu["device"], "gpu_dtype": gpu["dtype"],
               "jax_port_max_rel_diff": cpu["jax_port_max_rel_diff"]}
    json.dump(summary, open(os.path.join(OUT, "emulator_benchmark.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("truth", "ref", "cpu", "gpu", "plot"))
    ap.add_argument("--cp-path", default=None,
                    help="gpu: a directory holding cosmopower_jax, if not installed")
    ap.add_argument("--cache", default=CACHE, help="where the stages write and read")
    ap.add_argument("--syren", default=SYREN, help="the symbolic_pofk git clone")
    a = ap.parse_args()
    CACHE, SYREN = a.cache, a.syren
    os.makedirs(CACHE, exist_ok=True)
    {"truth": stage_truth, "ref": stage_ref, "cpu": stage_cpu, "gpu": stage_gpu, "plot": stage_plot}[a.stage](a)
