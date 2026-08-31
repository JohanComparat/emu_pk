r"""What the emulator is worth: shape error, and derivative error.

Two numbers, and the second is the one that matters and the one emulator papers
usually leave out.

*Shape* error is the familiar one: the largest fractional departure from CLASS
over the trusted range of :math:`k`, after renormalising, so a spectrum that is
right in shape and wrong in amplitude is not scored as both.

*Derivative* error is what a Fisher forecast actually consumes.  An emulator can
reproduce :math:`P(k)` to a tenth of a percent and still get
:math:`\partial\ln P/\partial\theta` wrong, because the error surface is smooth
in :math:`k` and rough in :math:`\theta`; nothing in a shape comparison can see
it.  Here it is measured directly: automatic differentiation of the network
against central differences of *CLASS*, per parameter -- not against central
differences of the network, which agree with autodiff perfectly whenever the
network is smooth and say nothing about whether it is right.

Four things this measures that a shape comparison at one redshift does not:

**Every redshift, not just zero.**  A derivative *with respect to redshift* --
:math:`f\sigma_8` is built from one -- is invisible at a single redshift by
construction.  A CLASS solve returns every redshift it is asked for, so the
sweep costs nothing but the loop.

**The redshift derivative itself.**  :math:`f = -\,\mathrm{d}\ln D/\mathrm{d}\ln
(1+z)`, so :math:`\partial\ln P/\partial z` is the thing :math:`f\sigma_8` is
built from.  It is scored here the same way the eight parameters are.

**The metric's own noise floor.**  The reference is a central difference of
CLASS, which is not exact: it carries a truncation error going as the square of
the step and a solver-noise term going as its inverse.  Repeating at two step
sizes says which of "``wa`` is 5.6 % wrong" is the network and which is the
ruler.  Without it a run can spend a week chasing its own finite
difference.

**The amplitude, which the shape metric removes.**  Scoring renormalises at
:data:`K_NORM`, so a spectrum right in shape and wrong in amplitude scores
zero.  The discarded factor is reported as ``amplitude`` beside each shape
summary, which is what makes the shape number readable as an accuracy claim
about :math:`P(k)` rather than about its shape alone.

**Where in the box.**  An eight-dimensional Latin hypercube essentially never
samples a corner, so a median over the design says nothing about the walls --
and the walls are where a sampler with a wide prior spends its time.  Points
within ``EDGE_FRAC`` of any bound are reported separately, and so is the
extreme-quintessence corner where CLASS refused 0.02 % of the training solves
and the training set therefore has a hole.

Needs ``classy``: this is a comparison against the solver, so it belongs to the
``[gen]`` install, not the core one.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from . import box, cosmo, generate, grid
from .model import PkEmulator

__all__ = ["shape_error", "derivative_error", "redshift_derivative_error",
           "main"]

#: The range the comparison is scored over, which is not the full grid.  The
#: emulator is trained to 200 h/Mpc but a linear spectrum there is far inside
#: the regime the halo model replaces, and scoring it would report a number
#: nobody uses.
K_TRUSTED = (1e-3, 10.0)

#: Where the shape comparison is renormalised, in h/Mpc.  Distinct from
#: :data:`emu_pk.cosmo.K_PIVOT`, which is the *primordial* pivot in 1/Mpc and a
#: property of the cosmology rather than of this measurement.  They are not the
#: same number and they are not in the same units.
K_NORM = 0.05

#: Redshifts every run scores at.  Round numbers a user evaluates at, spanning
#: the trained range.
Z_NODES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)

#: A point is "on the edge" if some parameter sits within this fraction of its
#: range from a bound.
EDGE_FRAC = 0.10

#: `w0 + wa` above this is the corner where CPL dark energy behaves like matter
#: before recombination, where CLASS refused during generation, and where the
#: training set is therefore thinnest.  The design rejects `w0 + wa >= 0`.
QUINTESSENCE_CORNER = -0.15


def _class_pk(theta, z, k):
    """One CLASS solve -> ``(P_m, P_cb)``, each ``(n_z, n_k)``.

    Every redshift from one solve.  Scoring a z sweep by re-solving per
    redshift would cost six times as much for the same numbers.
    """
    d = dict(zip(box.PARAMS, theta))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return generate.solve(cosmo.class_params(
        h=d["h"], omega_b=d["omega_b"], omega_cdm=d["omega_cdm"],
        n_s=d["n_s"], ln10A_s=d["ln10A_s"], sum_mnu=d["sum_mnu"],
        w0=d["w0"], wa=d["wa"], k_max_h=grid.K_MAX,
        z_max=max(grid.Z_MAX, float(z.max()))), z, k)


def _pick(pm, pcb, which):
    return pm if which == "m" else pcb


def where_in_box(theta) -> dict:
    """Whereabouts of one design point: how close to a wall, and to the corner.

    Returned rather than printed so the caller can group by it.  ``edge`` is the
    smallest distance to any bound as a fraction of that parameter's range, so 0
    is on the wall and 0.5 is dead centre.
    """
    lo = np.array([box.BOX[p][0] for p in box.PARAMS])
    hi = np.array([box.BOX[p][1] for p in box.PARAMS])
    u = (np.asarray(theta) - lo) / (hi - lo)
    w0, wa = theta[box.PARAMS.index("w0")], theta[box.PARAMS.index("wa")]
    return {"edge": float(np.min(np.minimum(u, 1.0 - u))),
            "w0_plus_wa": float(w0 + wa)}


def _summary(errs, where, n_requested):
    """Median/p90/max overall, and again for the edge and the corner alone."""
    errs, where = np.asarray(errs), list(where)
    if errs.size == 0:
        return {"n_scored": 0, "n_requested": int(n_requested)}

    def stats(mask):
        e = errs[mask]
        if e.size == 0:
            return {"n": 0}
        return {"n": int(e.size), "median": float(np.median(e)),
                "p90": float(np.percentile(e, 90)), "max": float(e.max())}

    edge = np.array([w["edge"] < EDGE_FRAC for w in where])
    corner = np.array([w["w0_plus_wa"] > QUINTESSENCE_CORNER for w in where])
    out = stats(np.ones(errs.size, bool))
    out.pop("n")
    out.update(n_scored=int(errs.size), n_requested=int(n_requested),
               edge=stats(edge), interior=stats(~edge),
               quintessence_corner=stats(corner))
    return out


# ==========================================================================
# Shape
# ==========================================================================
def shape_error(emu, n: int = 32, z_nodes=Z_NODES, seed: int = 991,
                which=("m", "cb"), verbose=True):
    """Max ``|shape/CLASS - 1|`` over held-out cosmologies, at every redshift.

    Held out by construction: the design is drawn from a *different* seed from
    the training set's, so no point scored here was trained on.

    Both spectra from one solve.  CLASS returns ``P_m`` and ``P_cb`` together
    and the network has two heads, so scoring them in separate passes would
    double the only expensive part of this for no new information.  Returns
    ``{which: {z: summary}}``.
    """
    k = np.logspace(np.log10(K_TRUSTED[0]), np.log10(K_TRUSTED[1]), 300)
    i0 = int(np.argmin(abs(k - K_NORM)))
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
    which = (which,) if isinstance(which, str) else tuple(which)
    design = box.sample(n, seed=seed)

    errs = {w: {float(zz): [] for zz in z_nodes} for w in which}
    # What the shape metric divides out, kept rather than discarded: the
    # renormalisation at K_NORM makes a spectrum right in shape and wrong in
    # amplitude score zero, so the amplitude has to be reported separately or
    # it is not reported at all.
    amps = {w: {float(zz): [] for zz in z_nodes} for w in which}
    where = []
    for theta in design:
        try:
            solved = _class_pk(theta, z_nodes, k)
        except Exception as e:
            print(f"  CLASS refused a validation point ({type(e).__name__}); skipped")
            continue
        where.append(where_in_box(theta))
        for w in which:
            ref = _pick(*solved, w)
            got = np.asarray(emu.predict(k, z_nodes, theta, w))
            for j, zz in enumerate(z_nodes):
                a = got[j, i0] / ref[j, i0]
                r = (got[j] / got[j, i0]) / (ref[j] / ref[j, i0])
                errs[w][float(zz)].append(float(np.max(np.abs(r - 1))))
                amps[w][float(zz)].append(float(abs(a - 1)))

    out = {w: {f"{zz:g}": _summary(errs[w][float(zz)], where, n)
               for zz in z_nodes} for w in which}
    for w in which:
        for zz in z_nodes:
            s = out[w][f"{zz:g}"]
            if s.get("n_scored"):
                s["amplitude"] = _summary(amps[w][float(zz)], where, n)
    if verbose:
        for w in which:
            print(f"shape error vs CLASS, P_{w}, {len(where)}/{n} held-out points:")
            print(f"  {'z':>5}  {'median':>9} {'90th':>9} {'max':>9}   "
                  f"{'edge p90':>9} {'corner max':>10}   {'amp med':>9}")
            for zz in z_nodes:
                s = out[w][f"{zz:g}"]
                if not s.get("n_scored"):
                    continue
                e = s["edge"].get("p90")
                c = s["quintessence_corner"].get("max")
                am = s.get("amplitude", {}).get("median")
                print(f"  {zz:5g}  {s['median']:8.4%} {s['p90']:8.4%} "
                      f"{s['max']:8.4%}   "
                      f"{'--' if e is None else format(e, '8.4%')} "
                      f"{'--' if c is None else format(c, '9.4%')}   "
                      f"{'--' if am is None else format(am, '8.4%')}")
    return out


# ==========================================================================
# Derivatives with respect to the parameters
# ==========================================================================
def _class_dlnp(theta, j, hstep, z_nodes, k, which):
    """Central difference of CLASS in parameter ``j``.  ``None`` if it cannot."""
    lo, hi = box.BOX[box.PARAMS[j]]
    tp, tm = np.array(theta, dtype=float), np.array(theta, dtype=float)
    tp[j] += hstep
    tm[j] -= hstep
    if not (lo <= tp[j] <= hi and lo <= tm[j] <= hi):
        return None
    try:
        up = _pick(*_class_pk(tp, z_nodes, k), which)
        dn = _pick(*_class_pk(tm, z_nodes, k), which)
    except Exception:
        return None
    return (np.log(up) - np.log(dn)) / (2 * hstep)


def derivative_error(emu, n: int = 16, z_nodes=Z_NODES, seed: int = 991,
                     rel_step=0.02, which="m", convergence=True, verbose=True):
    r"""Autodiff of the network against **central differences of CLASS**.

    Reported per parameter as the median over ``k`` of
    :math:`|\partial\ln P/\partial\theta` (emulator) :math:`-\ \partial\ln
    P/\partial\theta` (CLASS):math:`|` relative to the CLASS value, so a
    parameter the emulator is simply blind to reports 1 rather than something
    small.  That distinction is the whole point: a derivative that is *absent*
    shows up in a Fisher matrix as a flat direction, which is visible; one that
    is merely wrong does not.

    With ``convergence``, the same reference is recomputed at half the step and
    the two are compared.  That difference is the **floor**: the metric cannot
    resolve an error below it, and a score at or under its own floor is a
    statement about the ruler rather than about the network.  It doubles the
    number of CLASS solves, which is why it is a flag.
    """
    k = np.logspace(np.log10(K_TRUSTED[0]), np.log10(K_TRUSTED[1]), 120)
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
    design = box.sample(n, seed=seed)
    out = {f"{zz:g}": {} for zz in z_nodes}

    for j, p in enumerate(box.PARAMS):
        rel = {float(zz): [] for zz in z_nodes}
        floor = {float(zz): [] for zz in z_nodes}
        lo, hi = box.BOX[p]
        for theta in design:
            hstep = rel_step * (hi - lo)
            dref = _class_dlnp(theta, j, hstep, z_nodes, k, which)
            if dref is None:
                continue
            dhalf = (_class_dlnp(theta, j, hstep / 2, z_nodes, k, which)
                     if convergence else None)
            # jacfwd over the whole z vector at once: the network is cheap and
            # this keeps the emulator and the reference on identical rows.
            demu = np.asarray(jax.jacfwd(
                lambda t: jnp.log(emu.predict(k, z_nodes, t, which))
            )(jnp.asarray(theta)))[..., j]
            for i, zz in enumerate(z_nodes):
                scale = np.maximum(np.abs(dref[i]), 1e-8)
                rel[float(zz)].append(
                    float(np.median(np.abs(demu[i] - dref[i]) / scale)))
                if dhalf is not None:
                    floor[float(zz)].append(
                        float(np.median(np.abs(dhalf[i] - dref[i]) / scale)))
        for zz in z_nodes:
            r, f = rel[float(zz)], floor[float(zz)]
            out[f"{zz:g}"][p] = {
                "err": float(np.median(r)) if r else float("nan"),
                "floor": float(np.median(f)) if f else None,
                "n_scored": len(r)}

    if verbose:
        _print_deriv(f"derivative error vs CLASS finite differences, P_{which}",
                     out, z_nodes, box.PARAMS)
    return out


# ==========================================================================
# The redshift derivative -- what f sigma_8 is made of
# ==========================================================================
def redshift_derivative_error(emu, n: int = 16, z_nodes=Z_NODES,
                              seed: int = 991, dz=0.05, which="m",
                              verbose=True):
    r""":math:`\partial\ln P/\partial z`, autodiff against CLASS.

    An emulator can be level with another on the spectrum itself and much worse
    on :math:`f\sigma_8`, because that weakness lives in the redshift direction
    specifically and nothing in a shape comparison at fixed z can see it.  So it
    is scored directly.

    At :math:`z = 0` a central difference would need :math:`z < 0`, so a
    second-order *forward* stencil is used there instead of shifting the node --
    :math:`z = 0` is where :math:`\sigma_8` is quoted, so it is the one node
    worth the extra term.
    """
    k = np.logspace(np.log10(K_TRUSTED[0]), np.log10(K_TRUSTED[1]), 120)
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))

    # Both step sizes up front, so the floor costs extra `pk_lin` calls and not
    # extra solves: the union of every z any stencil needs goes into one solve.
    wanted, plans = [], []
    for step in (dz, dz / 2):
        stencils = []
        for zz in z_nodes:
            if zz >= step:
                pts, cf = [zz - step, zz + step], [-0.5 / step, 0.5 / step]
            else:
                # z=0 is where sigma_8 is quoted, so it gets a second-order
                # forward stencil rather than being moved off the node.
                pts = [zz, zz + step, zz + 2 * step]
                cf = [-1.5 / step, 2.0 / step, -0.5 / step]
            stencils.append((len(wanted), cf))
            wanted.extend(pts)
        plans.append(stencils)
    wanted = np.array(wanted)

    def _fd(ref, stencils, i):
        off, cf = stencils[i]
        return sum(c * ref[off + m] for m, c in enumerate(cf))

    design = box.sample(n, seed=seed)
    rel = {float(zz): [] for zz in z_nodes}
    floor = {float(zz): [] for zz in z_nodes}
    where = []
    for theta in design:
        try:
            ref = np.log(_pick(*_class_pk(theta, wanted, k), which))
        except Exception:
            continue
        # (n_z, n_k, n_z): the emulator at every node differentiated against
        # every node.  Only the diagonal in z is meaningful -- the off-diagonal
        # blocks are zero because each row of `predict` depends on its own z
        # alone -- so it is `demu[i, :, i]` below and not `demu[i]`.
        demu = np.asarray(jax.jacfwd(
            lambda zs: jnp.log(emu.predict(k, zs, theta, which))
        )(jnp.asarray(z_nodes)))
        where.append(where_in_box(theta))
        for i, zz in enumerate(z_nodes):
            dref = _fd(ref, plans[0], i)
            scale = np.maximum(np.abs(dref), 1e-8)
            rel[float(zz)].append(
                float(np.median(np.abs(demu[i, :, i] - dref) / scale)))
            # The forward stencil at z=0 has a larger truncation error than the
            # central ones, and z=0 is the node that matters most -- so the
            # floor is measured here too rather than assumed small.
            floor[float(zz)].append(
                float(np.median(np.abs(_fd(ref, plans[1], i) - dref) / scale)))

    out = {f"{zz:g}": {"err": float(np.median(rel[float(zz)]))
                       if rel[float(zz)] else float("nan"),
                       "floor": float(np.median(floor[float(zz)]))
                       if floor[float(zz)] else None,
                       "n_scored": len(rel[float(zz)])} for zz in z_nodes}
    if verbose:
        print(f"dlnP/dz vs CLASS finite differences, P_{which} "
              f"({len(where)}/{n} points):")
        for zz in z_nodes:
            s = out[f"{zz:g}"]
            fl = "" if s["floor"] is None else f"   floor {s['floor']:.3%}"
            print(f"  z = {zz:<5g}  {s['err']:.3%}{fl}"
                  f"{'   (forward stencil)' if zz < dz else ''}")
    return out


def _print_deriv(title, out, z_nodes, names):
    print(f"{title}:")
    print(f"  {'':<10}" + "".join(f"{f'z={zz:g}':>12}" for zz in z_nodes))
    for p in names:
        row = "".join(f"{out[f'{zz:g}'][p]['err']:11.2%} " for zz in z_nodes)
        print(f"  {p:<10}{row}")
    floors = [out[f"{z_nodes[0]:g}"][p]["floor"] for p in names]
    if any(f is not None for f in floors):
        print(f"  {'-- floor at z=%g (the metric, not the network)' % z_nodes[0]:<10}")
        print(f"  {'':<10}" + "".join(
            f"{('%.2f%%' % (100 * f)) if f is not None else '--':>12}"
            for f in floors))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n-shape", type=int, default=32)
    ap.add_argument("--n-deriv", type=int, default=16)
    ap.add_argument("--z", type=float, nargs="+", default=list(Z_NODES),
                    help="redshifts to score at; one CLASS solve covers all")
    ap.add_argument("--no-convergence", action="store_true",
                    help="skip the step-size check that measures the metric's "
                         "own noise floor; halves the CLASS solves")
    ap.add_argument("--json", default=None,
                    help="also write the numbers here, so whatever quotes them "
                         "reads a file rather than a terminal.  A validation "
                         "figure retyped by hand is a validation figure that "
                         "can silently outlive the weights it describes.")
    a = ap.parse_args(argv)
    emu = PkEmulator(a.weights, check_box=False)
    z_nodes = tuple(a.z)
    out = {"z_nodes": list(z_nodes), "n_shape": a.n_shape, "n_deriv": a.n_deriv,
           "k_trusted": list(K_TRUSTED), "k_norm": K_NORM,
           "edge_frac": EDGE_FRAC,
           # Everything that changes what the network *is*.  A validation file
           # that does not say which network it scored is a number without a
           # subject, and a file that omits any of them silently reads as
           # the default.
           "target_form": "reduced" if emu._reduced else "raw",
           "output_form": emu._output_form,
           "z_var": emu._z_var,
           "loss_form": str(emu.w.get("loss_form", "whitened_mse")),
           "epoch": int(emu.w.get("epoch", -1)),
           "weights": str(a.weights or "shipped")}
    out["shape"] = shape_error(emu, a.n_shape, z_nodes)
    out["derivative"] = derivative_error(
        emu, a.n_deriv, z_nodes, convergence=not a.no_convergence)
    out["derivative_z"] = redshift_derivative_error(emu, a.n_deriv, z_nodes)
    if a.json:
        import json
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
