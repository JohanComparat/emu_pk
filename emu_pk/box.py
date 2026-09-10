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
    nu_r1         --                0.0000  0.3333
    nu_r2         --                0.0000  0.5000

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
#: ``nu_r1``/``nu_r2`` are appended for the same reason and split the neutrino
#: mass over three species: :math:`m_i = r_i \Sigma m_\nu` with
#: :math:`r_3 = 1 - r_1 - r_2`, ordered :math:`r_1 \le r_2 \le r_3`.  ``sum_mnu``
#: keeps index 5 and keeps its meaning -- it is still the sum -- so every prior,
#: every published result and every existing column stay where they were.  What
#: is new is *how the sum is divided*, which the degenerate convention fixed at
#: (1/3, 1/3, 1/3) and oscillation experiments say is not what the world does.
PARAMS = ("omega_b", "omega_cdm", "h", "n_s", "ln10A_s", "sum_mnu", "w0", "wa",
          "Omega_k", "nu_r1", "nu_r2")

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
    # The ordered simplex, not a rectangle: `sample` rejects the corner where
    # r2 < r1 or r2 > (1 - r1)/2.  r1 cannot exceed 1/3 under that constraint,
    # so its bound is the constraint rather than a choice.
    "nu_r1":    (0.0000, 1.0 / 3.0),
    "nu_r2":    (0.0000, 0.5000),
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
    r"""``(n, len(PARAMS))`` Latin-hypercube design, columns in :data:`PARAMS` order.

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

    **The neutrino masses.**  ``sum_mnu`` is still the sum; ``nu_r1`` and
    ``nu_r2`` say how it is divided, with :math:`m_i = r_i \Sigma m_\nu` and
    :math:`r_3 = 1 - r_1 - r_2`.  Points outside the *ordered* simplex
    :math:`0 \le r_1 \le r_2 \le (1-r_1)/2` are rejected and redrawn.  The
    ordering is not a taste constraint either: CLASS sums the species'
    contributions and cannot tell them apart, so the six permutations of one
    mass vector are the same cosmology, and sampling all six would spend
    network capacity learning an exact symmetry rather than the physics.
    ``r_1 \le 1/3`` follows from the constraint rather than being imposed.

    Both orderings and the degenerate limit live inside it.  Measured against
    the Esteban et al. (2024) splittings, in :math:`r` coordinates: normal at
    :math:`\Sigma = 0.059` eV is (0.000, 0.147), inverted at 0.101 eV is
    (0.015, 0.489), and both tend to (1/3, 1/3) as the mass grows and the
    splittings stop mattering.

    **The degenerate point is a vertex of that simplex**, not an interior
    point -- it is where ``r_1`` meets its bound and the constraint is tight at
    once -- and it cannot be made interior, because a spread is non-negative.
    That is the same situation as ``sum_mnu = 0`` and ``z = 0``, and it matters
    more than either because (1/3, 1/3) is where every published result and the
    whole of 1.0.0 sit.  ``validate`` scores it as its own stratum.

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
    i_r1, i_r2 = PARAMS.index("nu_r1"), PARAMS.index("nu_r2")

    kept = np.empty((0, len(PARAMS)))
    # Draw generously and filter.  Two cuts now, and together they accept ~0.40
    # -- 0.80 for the CPL one and 0.50 for the simplex -- so the oversample is
    # 3.0 rather than 1.6 and two rounds are still almost always enough.  The
    # loop remains the guarantee.
    while len(kept) < n:
        want = int((n - len(kept)) * 3.0) + 16
        pts = lo + _lhs(want, len(PARAMS), rng) * (hi - lo)
        ok = pts[:, i_w0] + pts[:, i_wa] < 0.0
        # The neutrino masses are *interchangeable* -- CLASS sums their
        # contributions and knows nothing of which is which -- so the six
        # permutations of one mass vector are the same cosmology.  Ordering
        # them is what stops the network spending capacity learning an exact
        # symmetry instead of the physics.
        ok &= pts[:, i_r1] <= pts[:, i_r2]
        ok &= pts[:, i_r2] <= (1.0 - pts[:, i_r1]) / 2.0
        kept = np.vstack([kept, pts[ok]])
    kept = kept[:n]
    for name, value in (pin or {}).items():
        if name not in PARAMS:
            raise ValueError(
                f"cannot pin {name!r}: this box samples {list(PARAMS)}.")
        kept[:, PARAMS.index(name)] = float(value)
    return kept


#: Slack on the bounds, in units of the axis's own width.  A bound is a
#: physical statement, not a bit pattern, and the network is fed float32: a
#: point that is inside in double precision can land outside once rounded.
#: ``nu_r1``'s upper bound *is* 1/3, and ``np.float32(1/3)`` is 9.9e-9 above it
#: -- so the degenerate neutrino point, the convention every result before
#: version 2 was published at, was refused for a rounding error.  The slack is
#: two float32 epsilons of the range, which is far below any width at which the
#: fit changes and far above the largest rounding the dtype can produce.
_SLACK = 2.0 * float(np.finfo(np.float32).eps)


def inside(theta) -> dict:
    """Map ``{name: (value, bounds)}`` for every parameter outside the box.

    Empty when the point is inside.  Takes a mapping or a sequence in
    :data:`PARAMS` order.

    The comparison carries :data:`_SLACK` of the axis's width, so a value that
    is inside in double precision is not refused once it has been rounded to
    the float32 the network runs in.
    """
    if not hasattr(theta, "keys"):
        theta = dict(zip(PARAMS, np.asarray(theta)))
    out = {}
    for p, v in theta.items():
        if p not in BOX or v is None:
            continue
        lo, hi = BOX[p]
        eps = _SLACK * (hi - lo)
        if not lo - eps <= float(v) <= hi + eps:
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
