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

Two of those matter more than the rest.  CosmoPower's floor on ``h`` is 0.64,
which sits 0.03 below the Planck fiducial -- close enough that a sampler with a
wide ``h`` prior leaves the box in ordinary use.  And ``w0``/``wa`` are absent
from it entirely, which is why the differentiable path in ``ggah_mod`` returns
``dP/dw0 = 0`` today: not a small response, an absent one.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PARAMS", "BOX", "sample", "inside", "check"]

#: Network input order.  Everything downstream -- the generator's shard columns,
#: the training design matrix, the predictor's argument packing -- reads this
#: tuple rather than repeating the order, because a silently permuted column is
#: the kind of error that trains perfectly well and predicts nonsense.
PARAMS = ("omega_b", "omega_cdm", "h", "n_s", "ln10A_s", "sum_mnu", "w0", "wa")

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


def sample(n: int, seed: int = 20260827) -> np.ndarray:
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
    return kept[:n]


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
