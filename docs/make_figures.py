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
from emu_pk import box, grid, model
from emu_pk.model import PkEmulator

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "_static" / "figures"
#: The fiducial, built from the box by name.  A literal array here would be
#: silently one short the next time `box.PARAMS` grows, and a short theta used
#: to return a spectrum rather than raise.
_PLANCK_BY_NAME = {"omega_b": 0.02237, "omega_cdm": 0.1200, "h": 0.6736,
                   "n_s": 0.9649, "ln10A_s": 3.044, "sum_mnu": 0.06,
                   "w0": -1.0, "wa": 0.0, "Omega_k": 0.0,
                   "nu_r1": 1.0 / 3.0, "nu_r2": 1.0 / 3.0}
PLANCK = np.array([_PLANCK_BY_NAME[p] for p in box.PARAMS])


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
    """P(k, z), and the same curves' residual against CLASS.

    Needs `[gen]`: the right panel is the emulator against the solver it is
    distilled from, on the same lines as the left, which is the comparison a
    reader most wants next to the spectrum itself.
    """
    from emu_pk import validate as V

    zs = (0.0, 0.5, 1.0, 2.0, 5.0)
    k = np.logspace(-4, np.log10(grid.K_MAX), 400)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.6))

    # One CLASS solve returns every redshift.
    ref = V._class_pk(PLANCK, np.array(zs), k)[0]
    got = np.asarray(emu.pk(k, np.array(zs), PLANCK))

    for i, z in enumerate(zs):
        a.loglog(k, got[i], lw=1.4, label=f"$z = {z:g}$")
    a.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
          ylabel=r"$P_m(k,z)\ [(h^{-1}\mathrm{Mpc})^3]$",
          title="Planck 2018, across the trained range")
    a.legend(frameon=False, fontsize=8)

    for i, z in enumerate(zs):
        b.semilogx(k, 100.0 * (got[i] / ref[i] - 1.0), lw=1.2,
                   label=f"$z = {z:g}$")
    b.axhline(0, color="k", lw=0.7)
    b.axvspan(V.K_TRUSTED[0], V.K_TRUSTED[1], color="0.85", alpha=0.45,
              zorder=0, label="scored range")
    b.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$",
          ylabel=r"$P_{\rm emu}/P_{\rm CLASS} - 1$ [%]",
          title="residual against CLASS, same curves")
    b.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, "01_spectrum")


def fig_derivatives(plt, emu):
    """dlnP/dtheta for every parameter in the box, two of them exact."""
    import jax
    import jax.numpy as jnp
    from emu_pk import cosmo

    k = np.logspace(-3, 1, 200)
    jac = np.asarray(jax.jacfwd(
        lambda t: jnp.log(emu.pk(k, 0.0, t)))(jnp.asarray(PLANCK)))

    # The two analytic ones are shown as the *residual* against their closed
    # form.  Plotted as values they look like wild oscillation, because the
    # axis auto-scales to arithmetic noise on a constant.  `ln10A_s` sits at
    # float32 epsilon; `n_s` sits ~250x higher, because its closed form is
    # added on the node grid and then interpolated, and the Catmull-Rom stencil
    # differences values of order 10 in float32.
    exact = {"ln10A_s": np.ones_like(k),
             "n_s": np.log(k * PLANCK[2] / cosmo.K_PIVOT)}
    eps32 = np.finfo(np.float32).eps

    # **Sized from the box, not written down.**  `zip` stops at the shorter of
    # its arguments, so a grid with fewer panels than there are parameters drops
    # the last ones silently -- the figure would simply not show the newest axis
    # and nothing would raise.  A 2x4 grid did exactly that when the box grew to
    # nine.
    ncol = int(np.ceil(np.sqrt(len(box.PARAMS))))
    nrow = int(np.ceil(len(box.PARAMS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.7 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(box.PARAMS):]:
        ax.set_visible(False)
    for i, (p, ax) in enumerate(zip(box.PARAMS, axes)):
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
    for ax in axes[:len(box.PARAMS)]:
        ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    fig.suptitle("Automatic differentiation at $z=0$.  `ln10A_s` and `n_s` are "
                 "not fitted: they are restored in closed form, so their two "
                 "panels show float32 arithmetic rather than fit error.",
                 fontsize=10)
    return _save(fig, "03_derivatives")


def fig_box(plt):
    """The two planes where `sample` rejects, and what it keeps there."""
    d = box.sample(4000, seed=7)
    i = box.PARAMS.index
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))

    lo0, hi0 = box.BOX["w0"]
    loa, hia = box.BOX["wa"]
    a.add_patch(plt.Rectangle((lo0, loa), hi0 - lo0, hia - loa,
                              fc="0.88", ec="0.6", lw=0.8, zorder=0))
    a.plot(d[:, i("w0")], d[:, i("wa")], ".", ms=1.2, color="C0", zorder=2)
    a.plot([lo0, hi0], [-lo0, -hi0], "k--", lw=1.0, zorder=3,
           label="$w_0 + w_a = 0$")
    a.set(xlabel="$w_0$", ylabel="$w_a$", xlim=(lo0 - .05, hi0 + .05),
          ylim=(loa - .05, hia + .05), title="CPL: $w_0 + w_a < 0$")
    a.legend(fontsize=8, loc="lower left", framealpha=0.92)

    lo1, hi1 = box.BOX["nu_r1"]
    lo2, hi2 = box.BOX["nu_r2"]
    b.add_patch(plt.Rectangle((lo1, lo2), hi1 - lo1, hi2 - lo2,
                              fc="0.88", ec="0.6", lw=0.8, zorder=0))
    b.plot(d[:, i("nu_r1")], d[:, i("nu_r2")], ".", ms=1.2, color="C0", zorder=2)
    r1 = np.linspace(lo1, hi1, 64)
    b.plot(r1, r1, "k--", lw=1.0, zorder=3, label="$r_1 = r_2$")
    b.plot(r1, (1 - r1) / 2, "k-.", lw=1.0, zorder=3, label="$r_2 = (1-r_1)/2$")
    b.plot(1 / 3, 1 / 3, "*", ms=11, color="C3", zorder=4, label="degenerate")
    b.set(xlabel="$r_1$", ylabel="$r_2$", title=r"neutrinos: the ordered simplex")
    b.legend(fontsize=8, loc="lower right", framealpha=0.92)
    return _save(fig, "04_the_box")


def fig_correction(plt):
    """The neutrino / CPL correction, exactly 1 at the LambdaCDM corner."""
    from emu_pk import cosmo, ratio

    k = np.logspace(-3, 1, 200)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for mnu in (0.0, 0.06, 0.2, 0.4, 0.6):
        # `cosmo.f_nu` rather than a local 93.14 eV: the conversion belongs to
        # one place, and the figure has to agree with what `ratio` is indexed on.
        f_nu = cosmo.f_nu(mnu, h=0.6736, Omega_m=0.31)
        a.semilogx(k, ratio.suppression_m(k, f_nu, 0.0, -1.0, 0.0), lw=1.4,
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
    floor = [v["shape_floor"][z]["median"] * 100 for z in zs]
    a.plot(x, floor, "-.", color="C3", lw=1.2, label="the metric's own floor")
    a.set(xlabel="$z$", ylabel="shape error vs CLASS [%]",
          title=f"held-out, {v['shape']['m'][zs[0]]['n_scored']} cosmologies")
    a.legend(frameon=False, fontsize=8)

    d = v["derivative"][zs[0]]
    # `ln10A_s` and `n_s` are restored in closed form rather than fitted, so
    # they score 1e-14 and 1e-6 and would stretch a shared log axis over twelve
    # decades, squashing every fitted axis against the right edge.  They are
    # their own figure (`03_derivatives`); this one shows what was fitted.
    names = [p for p in box.PARAMS if p not in model.ANALYTIC]
    err = [d[p]["err"] * 100 for p in names]
    flo = [(d[p]["floor"] or 0.0) * 100 for p in names]
    y = np.arange(len(names))
    b.barh(y, err, color="C0", label="network")
    b.plot(flo, y, "k|", ms=10, label="the metric's own floor")
    b.set_yticks(y)
    b.set_yticklabels([f"`{p}`" for p in names])
    b.set_xscale("log")
    b.set(xlabel=r"$|\Delta\,\partial\ln P/\partial\theta|$ [%], at $z=0$",
          title="derivative error, the fitted axes")
    b.legend(frameon=False, fontsize=8, loc="lower right")
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
    med = 100 * json.loads((pathlib.Path(emu_pk.__file__).parent / "data"
                            / "validation.json").read_text())["shape"]["m"]["0"]["median"]
    ax.axhspan(-med, med, color="C3", alpha=0.12,
               label=f"median shape error, {med:.3f} %")
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
    fig_validation(plt)
    fig_derivatives(plt, emu)
    fig_box(plt)
    try:
        fig_correction(plt)
    except Exception as e:                       # the table is optional data
        print(f"  skipped 05_correction: {type(e).__name__}: {e}")
    if a.fast:
        print("  --fast: skipping the figures that need CLASS "
              "(01_spectrum, 02_residuals)")
        return
    try:
        fig_spectrum(plt, emu)
        fig_against_class(plt, emu)
    except ImportError:
        print("  skipped the CLASS figures: needs `pip install 'emu_pk[gen]'`")


if __name__ == "__main__":
    main()
