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
built from.  It is scored here the same way the eleven parameters are.

**The metric's own noise floor.**  The reference is a central difference of
CLASS, which is not exact: it carries a truncation error going as the square of
the step and a solver-noise term going as its inverse.  Repeating at two step
sizes says which of "``wa`` is 5.6 % wrong" is the network and which is the
ruler.  Without it a run can spend a week chasing its own finite
difference.

**The amplitude, which the shape metric removes.**  Scoring renormalises at
``K_NORM``, so a spectrum right in shape and wrong in amplitude scores
zero.  The discarded factor is reported as ``amplitude`` beside each shape
summary, and the comparison with nothing removed is reported as ``total``.
The three together say what the emulator does to :math:`P(k)`: an amplitude, a
shape, and the whole thing.

**Where in the box.**  An eleven-dimensional Latin hypercube essentially never
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
from .model import _catmull_rom as model_catmull_rom

__all__ = ["shape_error", "derivative_error", "redshift_derivative_error",
           "flat_slice_error", "interpolation_floor", "main"]

#: The range the comparison is scored over, which is not the full grid.  The
#: emulator is trained to 200 h/Mpc but a linear spectrum there is far inside
#: the regime the halo model replaces, and scoring it would report a number
#: nobody uses.
K_TRUSTED = (1e-3, 10.0)

#: The decade *below* :data:`K_TRUSTED`, scored separately rather than not at
#: all.
#:
#: The curvature scale is :math:`k_{\rm curv} = \sqrt{|\Omega_k|}\,H_0/c`, which
#: at :math:`|\Omega_k| = 0.15` is :math:`1.29\times10^{-4}\ h\,{\rm Mpc}^{-1}`
#: -- inside the grid, and above :data:`emu_pk.grid.K_MIN`.  Measured against
#: CLASS: above :math:`k \sim 10^{-2}` the curvature response is a
#: k-independent growth rescaling, flat in k to better than 0.5 %; below
#: :math:`k \sim 6\times10^{-4}` it turns over, reaches 2.5 in ``ln P`` at
#: :math:`\Omega_k = +0.15`, and is *not monotonic* in :math:`\Omega_k`.
#:
#: ``K_TRUSTED`` sees none of that.  Sixteen per cent of the network's outputs
#: live in this band and until now they were generated and never scored, which
#: is a claim nobody had checked rather than a limitation anybody had stated.
#: Reported separately: the headline numbers stay defined exactly as they were,
#: and the cost of the grid reaching to 1e-4 becomes visible instead of hidden.
#:
#: Note that :data:`K_NORM` lies *outside* this band, so the number to read here
#: is ``total`` -- renormalising inside it would divide out the feature being
#: scored.
K_LOWK = (1e-4, 1e-3)

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

#: Below this, the closure :math:`\Omega_{de} = 1 - \Omega_k - \Omega_m -
#: \Omega_r` has gone negative and the cosmology has a negative dark-energy
#: density.
#:
#: Not rejected by the design, and reported for the same reason the
#: quintessence corner is: it is a region the box deliberately contains and a
#: median over a Latin hypercube says nothing about.  0.34 % of the *flat* box
#: already sits here -- the shipped weights are trained through it -- and
#: curvature takes that to 0.78 %, almost all of it on the open side, where
#: CLASS solves without complaining.  See :func:`emu_pk.box.sample`.
NEGATIVE_DE = 0.0

#: Where the neutrino mass axis crosses from observationally live into
#: observationally dead, in eV.  Cosmological bounds put :math:`\Sigma m_\nu`
#: below roughly 0.1--0.2 eV; the box goes to 0.6, so about half of it is mass
#: no measurement supports.
#:
#: **The box is not narrowed to match, and this stratum is why it does not have
#: to be.**  A bound on :math:`\Sigma m_\nu` is a *posterior*, and deriving one
#: means evaluating the likelihood far above it -- ``ggah_mod`` declares
#: ``"mnu": None`` and takes the prior from the user's sampler configuration,
#: where flat priors to 0.5 or 1.0 eV are ordinary.  A box capped at the
#: posterior could not support the run that produces the posterior.
#:
#: What the high-mass half costs is therefore reported rather than assumed --
#: and the answer, once measured properly, is that it costs nothing.  Two
#: independent 32-point runs, sixteen points a side, median shape error at
#: z = 0:
#:
#:     arm            light (<= 0.3 eV)   heavy (> 0.3 eV)
#:     Omega_k pinned      0.4953 %           0.3532 %
#:     ratios pinned       0.5234 %           0.4272 %
#:
#: **The heavy half fits better**, on median and on p90, in both.  That agrees
#: with where the target actually bends: the departure from a straight line in
#: ``ln P`` is 0.0027 over [0, 0.3] against 0.0004 over [0.3, 0.6], so the low
#: mass end is the curved one and the high end is nearly linear.  Narrowing the
#: box to the observational bound would discard the easy region and keep the
#: hard one.
#:
#: An earlier fourteen-point sample said the opposite -- tail 2.3x worse above
#: the split -- and was noise.  That is what the stratum is for.
HEAVY_NU = 0.30

#: `w0 + wa` above this is the corner where CPL dark energy behaves like matter
#: before recombination, where CLASS refused during generation, and where the
#: training set is therefore thinnest.  The design rejects `w0 + wa >= 0`.
QUINTESSENCE_CORNER = -0.15


def _class_pk(theta, z, k):
    """One CLASS solve -> ``(P_m, P_cb)``, each ``(n_z, n_k)``.

    Every redshift from one solve.  Scoring a z sweep by re-solving per
    redshift would cost six times as much for the same numbers.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return generate.solve(generate.class_params_for(
        theta, k_max_h=grid.K_MAX,
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
    d = dict(zip(box.PARAMS, np.asarray(theta, dtype=float)))
    # The closure CLASS applies, recomputed here so the stratum can be read off
    # the design without a solve.  Radiation is ~1e-4 and is left out: it is far
    # below the width of the region this is used to select.
    om = ((d["omega_b"] + d["omega_cdm"]) / d["h"] ** 2
          + cosmo.omega_nu(d["sum_mnu"], d["h"]))
    return {"edge": float(np.min(np.minimum(u, 1.0 - u))),
            "w0_plus_wa": float(d["w0"] + d["wa"]),
            "omega_k": float(d["Omega_k"]),
            "omega_de": float(1.0 - d["Omega_k"] - om),
            "sum_mnu": float(d["sum_mnu"]),
            # How far the three masses are from equal, as a fraction of the
            # sum.  0 is degenerate -- the convention itself, and the vertex
            # of the sampled simplex, so the corner a network is least
            # constrained at and the one every published result sits on.
            "nu_spread": float((1.0 - d["nu_r1"] - d["nu_r2"]) - d["nu_r1"])}


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
    # Two curvature strata, for two different questions.  `negative_de` is the
    # region the design deliberately keeps and nothing else can see; `curved`
    # is simply "away from flat", which is what says whether the ninth
    # parameter is being paid for by the cosmologies that use it.
    neg_de = np.array([w.get("omega_de", 1.0) < NEGATIVE_DE for w in where])
    curved = np.array([abs(w.get("omega_k", 0.0)) > 0.10 for w in where])
    # The observationally dead half of the mass axis, and its complement.
    heavy = np.array([w.get("sum_mnu", 0.0) > HEAVY_NU for w in where])
    # Near-degenerate neutrinos: the vertex of the sampled simplex, and the
    # slice every result predating this box was computed on.
    degen = np.array([abs(w.get("nu_spread", 0.0)) < 0.05 for w in where])
    out = stats(np.ones(errs.size, bool))
    out.pop("n")
    out.update(n_scored=int(errs.size), n_requested=int(n_requested),
               edge=stats(edge), interior=stats(~edge),
               quintessence_corner=stats(corner),
               negative_de=stats(neg_de), curved=stats(curved),
               heavy_nu=stats(heavy), light_nu=stats(~heavy),
               degenerate_nu=stats(degen))
    return out


# ==========================================================================
# Shape
# ==========================================================================
def interpolation_floor(n: int = 8, z_nodes=(0.0,), seed: int = 991,
                        band=K_TRUSTED, design=None, verbose=True):
    r"""What the *metric* costs, before any network is involved.

    :func:`shape_error` asks CLASS at 300 fresh log points while the network
    predicts on :data:`emu_pk.grid.k_grid`'s 400 nodes and
    ``model._interp_lnk`` interpolates linearly in :math:`\ln k` between them.
    That interpolation is scored as though it were network error.

    So it is measured the same way :func:`derivative_error` measures its own
    finite-difference floor: push a *CLASS* spectrum through the same path --
    solve on the native grid, interpolate to the scoring points, compare
    against CLASS solved at the scoring points -- and report what comes out.

    **The answer is that the shipped headline number is the ruler.**  Measured
    on the same ``seed=991`` design the shipped model was scored on, with the
    same renormalisation, at ``z = 0``, ``n = 16``:

    ==========================  =========  =========  =========
    .                           median     p90        max
    ==========================  =========  =========  =========
    interpolation floor         0.1124 %   0.2337 %   0.3068 %
    shipped network, reported   0.1113 %   0.2244 %   0.6213 %
    ratio                       **1.01**   **1.04**   0.49
    ==========================  =========  =========  =========

    The reported median and p90 are accounted for entirely by linear
    interpolation from the 400-node grid onto the 300 scoring points; the
    network's own error is below what this measurement can resolve.  Half the
    max is the ruler too.

    Linear interpolation of an acoustic wiggle at roughly six nodes per period
    is an :math:`O(h^2)` error of exactly this size, and the error sits in the
    acoustic band.  :meth:`PkEmulator._interp_lnk` is cubic now for that reason,
    which is :math:`O(h^4)` on the same nodes and takes the floor to 0.0208 %
    median -- back below the network.

    **This function calls that interpolant rather than reimplementing it.**  It
    did reimplement it, with ``np.interp``, and kept doing so after the emulator
    became cubic -- so it measured a path nothing takes and reported a floor
    five times too high, *above* the error it exists to bound.  A floor that
    does not share the code it is a floor for is worse than no floor.

    Returns ``{z: summary}``, in the same shape as everything else here.
    """
    k_nat = grid.k_grid()
    k = np.logspace(np.log10(band[0]), np.log10(band[1]), 300)
    # Renormalised at K_NORM and maxed over k, exactly as `shape_error` scores
    # a network -- otherwise this is a floor for a different measurement.
    i0 = int(np.argmin(abs(k - K_NORM)))
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
    design = box.sample(n, seed=seed) if design is None else np.asarray(design)
    errs, tots, where = ({float(zz): [] for zz in z_nodes},
                         {float(zz): [] for zz in z_nodes}, [])
    for theta in design:
        try:
            ref = _class_pk(theta, z_nodes, k)[0]
            nat = _class_pk(theta, z_nodes, k_nat)[0]
        except Exception as e:
            print(f"  CLASS refused a floor point ({type(e).__name__}); skipped")
            continue
        where.append(where_in_box(theta))
        for j, zz in enumerate(z_nodes):
            # **Through the emulator's own interpolant, not a copy of it.**
            # This floor was written against `jnp.interp` and kept using it
            # after `_interp_lnk` became cubic, so it measured a path nothing
            # takes any more and over-stated itself fivefold -- reporting a
            # floor *above* the error it was supposed to bound.  Calling the
            # real function is what stops the two drifting again.
            got = np.exp(np.asarray(model_catmull_rom(
                jnp.asarray(np.log(k_nat)), jnp.asarray(np.log(nat[j])),
                jnp.asarray(np.log(k)))))
            r = (got / got[i0]) / (ref[j] / ref[j][i0])
            errs[float(zz)].append(float(np.max(np.abs(r - 1.0))))
            tots[float(zz)].append(float(np.max(np.abs(got / ref[j] - 1.0))))
    out = {f"{zz:g}": _summary(errs[float(zz)], where, len(design))
           for zz in z_nodes}
    for zz in z_nodes:
        if out[f"{zz:g}"].get("n_scored"):
            out[f"{zz:g}"]["total"] = _summary(tots[float(zz)], where,
                                               len(design))
    if verbose:
        print(f"interpolation floor of the shape metric, k in "
              f"[{band[0]:g}, {band[1]:g}], {len(where)}/{len(design)} points:")
        print(f"  {'z':>5}  {'median':>9} {'90th':>9} {'max':>9}")
        for zz in z_nodes:
            r = out[f"{zz:g}"]
            if r.get("n_scored"):
                print(f"  {zz:5g}  {r['median']:8.4%} {r['p90']:8.4%} "
                      f"{r['max']:8.4%}")
    return out


def shape_error(emu, n: int = 32, z_nodes=Z_NODES, seed: int = 991,
                which=("m", "cb"), verbose=True, band=K_TRUSTED, design=None,
                label="shape error"):
    """Max ``|shape/CLASS - 1|`` over held-out cosmologies, at every redshift.

    Held out by construction: the design is drawn from a *different* seed from
    the training set's, so no point scored here was trained on.

    Both spectra from one solve.  CLASS returns ``P_m`` and ``P_cb`` together
    and the network has two heads, so scoring them in separate passes would
    double the only expensive part of this for no new information.  Returns
    ``{which: {z: summary}}``.
    """
    k = np.logspace(np.log10(band[0]), np.log10(band[1]), 300)
    i0 = int(np.argmin(abs(k - K_NORM)))
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
    which = (which,) if isinstance(which, str) else tuple(which)
    # An explicit design is what makes two arms comparable: `box.sample` under a
    # wider box draws different values for the older parameters than it did
    # under eight, so scoring two networks on "seed 991" alone compares them on
    # different cosmologies.  At n = 32 the median scatters enough to matter.
    design = box.sample(n, seed=seed) if design is None else np.asarray(design)
    n = len(design)

    errs = {w: {float(zz): [] for zz in z_nodes} for w in which}
    # What the shape metric divides out, kept rather than discarded: the
    # renormalisation at K_NORM makes a spectrum right in shape and wrong in
    # amplitude score zero, so the amplitude has to be reported separately or
    # it is not reported at all.
    amps = {w: {float(zz): [] for zz in z_nodes} for w in which}
    # The same comparison with nothing removed: amplitude and shape are
    # separate statements, and this is the one number that covers P(k) itself.
    tots = {w: {float(zz): [] for zz in z_nodes} for w in which}
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
                tots[w][float(zz)].append(
                    float(np.max(np.abs(got[j] / ref[j] - 1))))

    out = {w: {f"{zz:g}": _summary(errs[w][float(zz)], where, n)
               for zz in z_nodes} for w in which}
    for w in which:
        for zz in z_nodes:
            s = out[w][f"{zz:g}"]
            if s.get("n_scored"):
                s["amplitude"] = _summary(amps[w][float(zz)], where, n)
                s["total"] = _summary(tots[w][float(zz)], where, n)
    if verbose:
        for w in which:
            print(f"{label} vs CLASS, P_{w}, k in "
                  f"[{band[0]:g}, {band[1]:g}], {len(where)}/{n} held-out "
                  f"points:")
            print(f"  {'z':>5}  {'median':>9} {'90th':>9} {'max':>9}   "
                  f"{'edge p90':>9} {'corner max':>10}   {'amp med':>9}"
                  f" {'total med':>10}")
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
                      f"{'--' if am is None else format(am, '8.4%')}"
                      f" {format(s['total']['median'], '9.4%')}")
    return out


def flat_slice_error(emu, n: int = 32, z_nodes=Z_NODES, seed: int = 991,
                     which=("m", "cb"), verbose=True, band=K_TRUSTED):
    r"""The same score, on the flat slice: :math:`\Omega_k` pinned to zero.

    **The number that says what the ninth parameter cost the other eight.**

    A curvature axis widens the space the same network capacity has to cover,
    and a user who never leaves :math:`\Omega_k = 0` should not pay much for
    that.  Scoring the full eleven-dimensional design cannot answer it -- the
    curved points are a different question -- so this pins the column and scores
    the flat cosmologies alone, against the released flat-box figure of 0.111 %.

    Held out exactly as :func:`shape_error` is: the design comes from a
    different seed from the training set's.  Pinning is applied after the draw,
    so the other ten columns are the ones the seed names, and a flat control
    trained on ``box.sample(pin={"Omega_k": 0})`` is scored here on cosmologies
    drawn the same way.
    """
    design = box.sample(n, seed=seed, pin={"Omega_k": 0.0})
    return shape_error(emu, n, z_nodes, seed, which, verbose, band, design,
                       label="flat-slice shape error")


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
    ap.add_argument("--allow-narrow-box", action="store_true",
                    help="score a checkpoint that has no input for some "
                         "parameter this box samples.  For a pilot arm fitted "
                         "without an axis; never for weights that ship.")
    ap.add_argument("--no-floor", action="store_true",
                    help="skip the shape metric's own interpolation floor.  "
                         "It costs 2 CLASS solves per point and it is what "
                         "says how much of the tail is the network and how "
                         "much is linear interpolation across the acoustic "
                         "wiggles")
    ap.add_argument("--no-lowk", action="store_true",
                    help="skip the k < 1e-3 diagnostic band, which costs one "
                         "more CLASS pass over the same design")
    ap.add_argument("--no-flat-slice", action="store_true",
                    help="skip the Omega_k = 0 score")
    ap.add_argument("--pin-score", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="score only the slice with these columns pinned, e.g. "
                         "--pin-score nu_r1=0.3333 --pin-score nu_r2=0.3333.  "
                         "The general form of --flat-only, for any control arm "
                         "trained with a column held fixed: that column has "
                         "zero variance, so the checkpoint's x_std along it is "
                         "~1e-30 and the network means anything only at the "
                         "pinned value.")
    ap.add_argument("--flat-only", action="store_true",
                    help="score the flat slice and nothing else.  For a "
                         "control arm trained with the curvature column "
                         "pinned: that column has zero variance, so the "
                         "checkpoint's x_std along it is ~1e-30 and the "
                         "network is meaningful only at Omega_k = 0.  "
                         "Scoring it on a curved design divides by that and "
                         "returns numbers that mean nothing.")
    ap.add_argument("--no-convergence", action="store_true",
                    help="skip the step-size check that measures the metric's "
                         "own noise floor; halves the CLASS solves")
    ap.add_argument("--json", default=None,
                    help="also write the numbers here, so whatever quotes them "
                         "reads a file rather than a terminal.  A validation "
                         "figure retyped by hand is a validation figure that "
                         "can silently outlive the weights it describes.")
    a = ap.parse_args(argv)
    emu = PkEmulator(a.weights, check_box=False,
                     allow_narrow_box=a.allow_narrow_box)
    z_nodes = tuple(a.z)
    out = {"z_nodes": list(z_nodes), "n_shape": a.n_shape, "n_deriv": a.n_deriv,
           "k_trusted": list(K_TRUSTED), "k_lowk": list(K_LOWK),
           "k_norm": K_NORM, "negative_de": NEGATIVE_DE,
           "heavy_nu": HEAVY_NU,
           "params": list(box.PARAMS),
           # What the network was *not* fed, for a pilot arm scored with
           # --allow-narrow-box.  Empty for anything that ships, and the
           # difference between "this arm is flat" and "this arm is bad".
           "narrow": list(getattr(emu, "_narrow", ())),
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
    # A control arm is one number, and asking it for any other is asking a
    # network about a direction it was never shown.
    pin_score = dict(kv.split("=", 1) for kv in (a.pin_score or []))
    pin_score = {k: float(v) for k, v in pin_score.items()}
    if a.flat_only:
        # `--flat-only` is `--pin-score Omega_k=0` under its old name, kept
        # because it is what the curvature arm's job script already passes.
        pin_score.setdefault("Omega_k", 0.0)
    out["flat_only"] = bool(a.flat_only)
    out["pinned_score"] = pin_score
    if pin_score:
        design = box.sample(a.n_shape, seed=991, pin=pin_score)
        what = ", ".join(f"{k}={v:g}" for k, v in sorted(pin_score.items()))
        out["shape_flat"] = shape_error(
            emu, a.n_shape, z_nodes, design=design,
            label=f"pinned-slice shape error ({what})")
        if not a.no_lowk:
            out["shape_flat_lowk"] = shape_error(
                emu, a.n_shape, z_nodes, band=K_LOWK, design=design,
                label=f"pinned low-k band ({what})")
    else:
        out["shape"] = shape_error(emu, a.n_shape, z_nodes)
        if not a.no_flat_slice:
            out["shape_flat"] = flat_slice_error(emu, a.n_shape, z_nodes)
        if not a.no_lowk:
            out["shape_lowk"] = shape_error(emu, a.n_shape, z_nodes,
                                            band=K_LOWK, label="low-k band")
        out["derivative"] = derivative_error(
            emu, a.n_deriv, z_nodes, convergence=not a.no_convergence)
        out["derivative_z"] = redshift_derivative_error(emu, a.n_deriv, z_nodes)
        if not a.no_floor:
            # The shape metric's own ruler, reported beside the number it
            # limits, the way `derivative_error` reports its finite-difference
            # floor.  The tail is where the difference shows.
            out["shape_floor"] = interpolation_floor(
                n=min(a.n_deriv, 8), z_nodes=z_nodes)
    if a.json:
        import json
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
