"""Build every figure in the tutorial.

Run this locally and **commit the output**.  ReadTheDocs installs the core
package only -- it cannot compile CLASS -- so the figures cannot be generated at
build time.  Anything comparing against CLASS needs ``pip install
'emu_pk[gen]'``; the rest needs only the core install and matplotlib.

    python docs/make_figures.py            # everything it can
    python docs/make_figures.py --fast     # skip the CLASS comparisons

Each figure prints the package version it was made with.  The figures are
deterministic -- rerunning this against unchanged weights reproduces them byte
for byte -- so the check for a stale figure is to regenerate and see whether
git reports a diff.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import emu_pk
from emu_pk import box, grid
from emu_pk.model import PkEmulator

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "_static" / "figures"
PLANCK = np.array([0.02237, 0.1200, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
        "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        # Titles collide without this, and a figure whose labels overlap is a
        # figure nobody trusts.
        "figure.constrained_layout.use": True,
    })
    return plt


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p)
    print(f"  wrote {p.relative_to(HERE)}  (emu_pk {emu_pk.__version__})")
    return p


# ---------------------------------------------------------------- core only
def fig_spectrum(plt, emu):
    """P(k, z) for a few cosmologies: what the package does."""
    k = np.logspace(-4, np.log10(grid.K_MAX), 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))

    for z in (0.0, 0.5, 1.0, 2.0, 5.0):
        a.loglog(k, emu.pk(k, z, PLANCK), lw=1.4, label=f"$z = {z:g}$")
    a.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
          ylabel=r"$P_m(k,z)\ [(h^{-1}\mathrm{Mpc})^3]$",
          title="Planck 2018, across the trained range")
    a.legend(frameon=False, fontsize=8)

    ref = np.asarray(emu.pk(k, 0.0, PLANCK))
    for lbl, j, v in ((r"$\Sigma m_\nu = 0.4$ eV", 5, 0.4),
                      (r"$w_0 = -0.7$", 6, -0.7),
                      (r"$w_a = +0.5$", 7, 0.5),
                      (r"$n_s = 1.02$", 3, 1.02)):
        t = PLANCK.copy()
        t[j] = v
        b.semilogx(k, np.asarray(emu.pk(k, 0.0, t)) / ref - 1.0, lw=1.4, label=lbl)
    b.axhline(0, color="k", lw=0.7)
    b.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
          ylabel=r"$P/P_{\rm Planck} - 1$",
          title="response to parameters CosmoPower lacks")
    b.legend(frameon=False, fontsize=8)
    return _save(fig, "01_spectrum")


def fig_derivatives(plt, emu):
    """dlnP/dtheta for all eight parameters, two of them exact."""
    import jax
    import jax.numpy as jnp
    from emu_pk import cosmo

    k = np.logspace(-3, 1, 200)
    jac = np.asarray(jax.jacfwd(
        lambda t: jnp.log(emu.pk(k, 0.0, t)))(jnp.asarray(PLANCK)))

    # The two analytic ones are shown as the *residual* against their closed
    # form.  Plotted as values they look like wild oscillation, because they
    # are 1 +/- 6e-8 and the axis auto-scales to the roundoff.
    exact = {"ln10A_s": np.ones_like(k),
             "n_s": np.log(k * PLANCK[2] / cosmo.K_PIVOT)}
    eps32 = np.finfo(np.float32).eps

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.4), sharex=True)
    for i, (p, ax) in enumerate(zip(box.PARAMS, axes.ravel())):
        if p in exact:
            resid = np.abs(jac[:, i] - exact[p])
            ax.loglog(k, np.maximum(resid, 1e-12), lw=1.3, color="C2")
            ax.axhline(eps32, color="C3", ls="--", lw=1.0,
                       label="float32 $\\epsilon$")
            ax.set_title(f"`{p}`: |error| vs closed form", fontsize=9)
            ax.legend(frameon=False, fontsize=8)
            ax.set_ylim(1e-12, 1e-4)
        else:
            ax.semilogx(k, jac[:, i], lw=1.4)
            ax.set_title(f"$\\partial\\ln P/\\partial$ `{p}`", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    fig.suptitle("Automatic differentiation at $z=0$.  `ln10A_s` and `n_s` are "
                 "not fitted: they are added back in closed form, so their "
                 "error is float32 roundoff.", fontsize=10)
    return _save(fig, "03_derivatives")


def fig_box(plt):
    """The training box against CosmoPower's."""
    cp = {"omega_b": (0.01875, 0.02625), "omega_cdm": (0.05, 0.255),
          "h": (0.64, 0.82), "n_s": (0.84, 1.10), "ln10A_s": (1.61, 3.91)}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, p in enumerate(box.PARAMS):
        lo, hi = box.BOX[p]
        ax.plot([0, 1], [i, i], color="C0", lw=6, solid_capstyle="butt",
                label="emu_pk" if i == 0 else None)
        if p in cp:
            c0, c1 = cp[p]
            ax.plot([(c0 - lo) / (hi - lo), (c1 - lo) / (hi - lo)], [i, i],
                    color="C1", lw=2.5, solid_capstyle="butt",
                    label="CosmoPower" if i == 0 else None)
        else:
            ax.text(1.02, i, "absent from CosmoPower", va="center",
                    fontsize=8, color="C1")
    ax.set_yticks(range(len(box.PARAMS)))
    ax.set_yticklabels([f"`{p}`" for p in box.PARAMS])
    ax.set_xlabel("fraction of the emu_pk range")
    ax.set_xlim(-0.02, 1.45)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Training box, normalised to emu_pk's")
    return _save(fig, "04_the_box")


def fig_correction(plt):
    """The neutrino / CPL correction, exactly 1 at the LambdaCDM corner."""
    from emu_pk import ratio

    k = np.logspace(-3, 1, 200)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for mnu in (0.0, 0.06, 0.2, 0.4, 0.6):
        f_nu = mnu / (93.14 * 0.6736 ** 2) / 0.31
        a.semilogx(k, ratio.suppression_m(k, 0.0, f_nu, -1.0, 0.0), lw=1.4,
                   label=rf"$\Sigma m_\nu = {mnu:g}$ eV")
    a.axhline(1.0, color="k", lw=0.7)
    a.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$", ylabel=r"$r(k)$",
          title="massive neutrinos, at $z=0$")
    a.legend(frameon=False, fontsize=8)

    # Just inside the table's own bounds: `ratio` refuses its edges rather
    # than clamping, which is the right behaviour and makes -0.7 and 0.5
    # unusable here even though they are the endpoints.
    for w0, wa in ((-1.0, 0.0), (-1.28, 0.0), (-0.72, 0.0), (-1.0, -0.68),
                   (-1.0, 0.48)):
        b.semilogx(k, ratio.suppression_m(k, 0.0, 0.0, w0, wa), lw=1.4,
                   label=rf"$w_0={w0:g},\ w_a={wa:+g}$")
    b.axhline(1.0, color="k", lw=0.7)
    b.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$", ylabel=r"$r(k)$",
          title="CPL dark energy, massless, at $z=0$")
    b.legend(frameon=False, fontsize=8)
    return _save(fig, "05_correction")


def fig_validation(plt):
    """The shipped validation record, drawn rather than tabulated."""
    v = json.loads((pathlib.Path(emu_pk.__file__).parent / "data"
                    / "validation.json").read_text())
    zs = sorted(v["shape"]["m"], key=float)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))

    med = [v["shape"]["m"][z]["median"] * 100 for z in zs]
    p90 = [v["shape"]["m"][z]["p90"] * 100 for z in zs]
    mx = [v["shape"]["m"][z]["max"] * 100 for z in zs]
    x = [float(z) for z in zs]
    a.plot(x, med, "o-", label="median")
    a.plot(x, p90, "s--", label="90th percentile")
    a.plot(x, mx, "^:", label="max")
    a.axhline(0.159, color="C3", ls="-.", lw=1.2, label="CosmoPower, 0.159 %")
    a.set(xlabel="$z$", ylabel="shape error vs CLASS [%]",
          title=f"held-out, {v['shape']['m'][zs[0]]['n_scored']} cosmologies")
    a.legend(frameon=False, fontsize=8)

    d = v["derivative"][zs[0]]
    names = list(box.PARAMS)
    err = [max(d[p]["err"], 1e-16) * 100 for p in names]
    flo = [(d[p]["floor"] or 1e-16) * 100 for p in names]
    y = np.arange(len(names))
    b.barh(y, err, color="C0", label="network")
    b.plot(flo, y, "k|", ms=10, label="the metric's own floor")
    b.set_yticks(y)
    b.set_yticklabels([f"`{p}`" for p in names])
    b.set_xscale("log")
    b.set(xlabel=r"$|\Delta\,\partial\ln P/\partial\theta|$ [%], at $z=0$",
          title="derivative error")
    b.legend(frameon=False, fontsize=8)
    return _save(fig, "02_accuracy")


# ------------------------------------------------------------- needs classy
def fig_against_class(plt, emu):
    """Residual against CLASS for held-out cosmologies."""
    from emu_pk import validate as V

    k = np.logspace(-3, 1, 300)
    i0 = int(np.argmin(abs(k - V.K_NORM)))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for th in box.sample(12, seed=991):
        try:
            ref = V._class_pk(th, 0.0, k)[0][0]
        except Exception:
            continue
        got = np.asarray(emu.pk(k, 0.0, th))
        r = (got / got[i0]) / (ref / ref[i0]) - 1.0
        ax.semilogx(k, 100 * r, lw=0.9, alpha=0.8)
    ax.axhline(0, color="k", lw=0.7)
    ax.axhspan(-0.159, 0.159, color="C3", alpha=0.12,
               label="CosmoPower's median, 0.159 %")
    ax.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
           ylabel="fractional residual vs CLASS [%]",
           title="Twelve held-out cosmologies at $z=0$, renormalised at "
                 rf"$k={V.K_NORM}$")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "02_residuals")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="skip figures that need CLASS")
    a = ap.parse_args(argv)

    plt = _style()
    emu = PkEmulator(check_box=False)
    print(f"emu_pk {emu_pk.__version__}")
    fig_spectrum(plt, emu)
    fig_validation(plt)
    fig_derivatives(plt, emu)
    fig_box(plt)
    try:
        fig_correction(plt)
    except Exception as e:                       # the table is optional data
        print(f"  skipped 05_correction: {type(e).__name__}: {e}")
    if a.fast:
        print("  --fast: skipping the CLASS comparison")
        return
    try:
        fig_against_class(plt, emu)
    except ImportError:
        print("  skipped 02_residuals: needs `pip install 'emu_pk[gen]'`")


if __name__ == "__main__":
    main()
