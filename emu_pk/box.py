r"""The training hypercube: bounds, sampler, and the guard that refuses to leave it.

A neural emulator is valid inside the box it was trained on and nowhere else.
Outside it the network does not fail, it *extrapolates* -- returning a number
that is finite, smooth and unwarranted.  So the box is data, checked on every
call that can afford to look, rather than a sentence in a docstring.

The bounds below are deliberately wider than CosmoPower's ``mpk_lin``, which is
the emulator this one replaces::

    parameter     CosmoPower        here
    omega_b       0.01875 0.02625   0.0170 0.0280
    omega_cdm     0.05    0.255     0.0500 0.3000
    h             0.64    0.82      0.5500 0.8500
    n_s           0.84    1.10      0.8400 1.1000
    ln10A_s       1.61    3.91      1.6100 4.0000
    sum_mnu       --                0.0000 0.6000
    w0            --               -1.5000 -0.5000
    wa            --               -1.0000  0.6000
    Omega_k       --               -0.1500  0.1500

Three of those matter more than the rest.  CosmoPower's floor on ``h`` is 0.64,
which sits 0.03 below the Planck fiducial -- close enough that a sampler with a
wide ``h`` prior leaves the box in ordinary use.  And ``w0``/``wa`` are absent
from it entirely, which is why the differentiable path in ``ggah_mod`` returns
``dP/dw0 = 0`` today: not a small response, an absent one.

And ``Omega_k`` is absent from *every* differentiable predictor, not only from
CosmoPower -- which is why ``ggah_mod`` refused a curved cosmology on this path
rather than approximating one, and sent it to a Boltzmann solver instead.  The
bound is measured rather than chosen: see :func:`sample`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PARAMS", "BOX", "sample", "inside", "check"]

#: Network input order.  Everything downstream -- the generator's shard columns,
#: the training design matrix, the predictor's argument packing -- reads this
#: tuple rather than repeating the order, because a silently permuted column is
#: the kind of error that trains perfectly well and predicts nonsense.
#: ``Omega_k`` is appended rather than inserted, deliberately: every existing
#: ``PARAMS.index(...)`` keeps its value, so a checkpoint's ``_in_idx`` and a
#: shard's column order stay comparable across the change.  The capital ``O``
#: breaks the lowercase habit of the other eight and is kept anyway, because it
#: is CLASS's own key *and* ``ggah_mod.cosmology.Cosmology``'s field name -- and
#: that agreement is what lets one mapping serve all three.
PARAMS = ("omega_b", "omega_cdm", "h", "n_s", "ln10A_s", "sum_mnu", "w0", "wa",
          "Omega_k")

#: Closed bounds, inclusive.  ``z`` is not here: it is a network input but not a
#: sampled axis -- one CLASS solve yields every redshift in
#: :data:`emu_pk.grid.Z_NODES_EMU`, so it is enumerated rather than drawn.
BOX = {
    "omega_b":  (0.0170, 0.0280),
    "omega_cdm": (0.0500, 0.3000),
    "h":        (0.5500, 0.8500),
    "n_s":      (0.8400, 1.1000),
    "ln10A_s":  (1.6100, 4.0000),
    "sum_mnu":  (0.0000, 0.6000),
    "w0":       (-1.5000, -0.5000),
    "wa":       (-1.0000, 0.6000),
    "Omega_k":  (-0.1500, 0.1500),
}


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube on the unit cube, one sample per stratum per axis.

    Written out rather than taken from ``scipy.stats.qmc`` so that generating a
    design needs nothing beyond numpy -- the sampler is the one piece of the
    generation path that the *inference* install also wants, for reproducing a
    design without a Boltzmann solver anywhere near it.
    """
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, d))
    pts = cut[:n, None] + u * (cut[1:, None] - cut[:n, None])
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def sample(n: int, seed: int = 20260827, pin: dict | None = None) -> np.ndarray:
    """``(n, len(PARAMS))`` Latin-hypercube design, columns in :data:`PARAMS` order.

    Deterministic in ``seed``: the design is reproducible from the seed alone,
    so a shard can be regenerated years later without shipping the design
    matrix, and two shards can never disagree about which cosmology index *i*
    means.

    Points violating ``w0 + wa < 0`` are rejected and redrawn.  That is not a
    taste constraint: with ``w0 + wa >= 0`` the CPL dark-energy density grows
    without bound towards early times, dark energy dominates before
    recombination, and CLASS either refuses or returns a spectrum that is not a
    cosmology anyone means to train on.

    **Curvature is not rejected anywhere, and the bound is why.**  ``Omega_k``
    spans ``[-0.15, 0.15]`` because that is the widest interval over which CLASS
    solves the *whole* box.  Measured: on the closed side, at the low-density
    corner (``omega_cdm = 0.05``, ``h = 0.85``, ``Omega_m = 0.108``) CLASS fails
    from ``Omega_k = -0.275`` onward, and at ``Omega_m = 0.261`` it survives to
    ``-0.40``.  On the open side it never refuses at all.

    That asymmetry is the trap.  Positive ``Omega_k`` drives the closure
    ``Omega_de = 1 - Omega_m - Omega_r - Omega_k`` negative, and CLASS solves
    those without a word -- a negative dark-energy density is exotic, not
    ill-posed.  **It is deliberately not rejected.**  0.34 % of the *flat* box
    already sits there and the shipped 1.0.0 weights were trained through it
    (``omega_cdm = 0.30``, ``h = 0.55`` gives ``Omega_m = 1.084``); curvature
    takes that to 0.78 %.  Carving it out now would silently narrow the flat box
    in the same release that widens it, and would break the one comparison that
    says what the ninth parameter cost the other eight.  ``validate`` reports
    the region as its own stratum instead, the way it does the quintessence
    corner.

    ``pin`` holds named columns at fixed values *after* the draw --
    ``sample(n, pin={"Omega_k": 0.0})`` is the flat control the curved design is
    compared against.  The design stays reproducible from the seed because the
    pin is part of the call.  Note that a pinned column has zero variance, so a
    network trained on it standardises that input to a constant and cannot be
    evaluated anywhere else along that axis.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([BOX[p][0] for p in PARAMS])
    hi = np.array([BOX[p][1] for p in PARAMS])
    i_w0, i_wa = PARAMS.index("w0"), PARAMS.index("wa")

    kept = np.empty((0, len(PARAMS)))
    # Draw generously and filter; the accepted fraction is ~0.8, so two rounds
    # are almost always enough and the loop is a guarantee rather than a plan.
    while len(kept) < n:
        want = int((n - len(kept)) * 1.6) + 16
        pts = lo + _lhs(want, len(PARAMS), rng) * (hi - lo)
        ok = pts[:, i_w0] + pts[:, i_wa] < 0.0
        kept = np.vstack([kept, pts[ok]])
    kept = kept[:n]
    for name, value in (pin or {}).items():
        if name not in PARAMS:
            raise ValueError(
                f"cannot pin {name!r}: this box samples {list(PARAMS)}.")
        kept[:, PARAMS.index(name)] = float(value)
    return kept


def inside(theta) -> dict:
    """Map ``{name: (value, bounds)}`` for every parameter outside the box.

    Empty when the point is inside.  Takes a mapping or a sequence in
    :data:`PARAMS` order.
    """
    if not hasattr(theta, "keys"):
        theta = dict(zip(PARAMS, np.asarray(theta)))
    out = {}
    for p, v in theta.items():
        if p not in BOX or v is None:
            continue
        lo, hi = BOX[p]
        if not lo <= float(v) <= hi:
            out[p] = (float(v), (lo, hi))
    return out


def check(theta, what: str = "the emulator training box"):
    """Raise if ``theta`` is outside the box.  Names every offending axis.

    Callers pass ``None`` for any value that is a tracer, which is how this is
    skipped rather than attempted inside a ``jax.jit`` trace -- see
    :func:`emu_pk.interp.concrete`.
    """
    bad = inside(theta)
    if bad:
        raise ValueError(
            f"outside {what}, where the network extrapolates with no accuracy "
            "guarantee: "
            + "; ".join(f"{p} = {v:.5g} not in [{lo:g}, {hi:g}]"
                        for p, (v, (lo, hi)) in sorted(bad.items()))
            + ".")
