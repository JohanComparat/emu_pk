r"""Gradient-safe interpolation primitives.

Ported from ``ggah_mod.numerics`` and ``ggah_mod.cosmology.nu_ratio`` when the
table moved here.  They are copied rather than imported because ``ggah_mod``
depends on *this* package: importing back would be a cycle.

Each one encodes a correctness detail that is invisible in the output when it is
got wrong, which is why they are written once and tested directly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["lin_weights", "pchip_slopes", "hermite", "concrete",
           "cascaded_arrays", "interp_cascaded",
           "tensor_arrays", "interp_tensor"]


def concrete(x):
    """``float(x)`` if the value is available, ``None`` under tracing.

    The mechanism by which validation is *skipped* rather than attempted inside
    a ``jax.jit`` trace: calling ``float()`` on a tracer raises, and a validity
    check that raises inside a trace breaks the gradient it was meant to guard.
    """
    try:
        return float(x)
    except Exception:
        return None


def lin_weights(x, grid):
    """Index and fraction for a clamped linear interpolation.

    The fraction is differentiable in ``x``; the index is not, which is what
    makes the gradient flow through the value rather than the lookup.

    **The clamp is on x, not on the fraction, and uses `where` rather than
    `clip`.**  Both details are load-bearing, and getting either wrong halves a
    gradient silently:

    * clamping the fraction with ``jnp.clip(t, 0, 1)`` means that when ``x``
      lands exactly on a grid node the fraction is exactly 1.0 -- sitting on the
      clip boundary.  ``clip`` is ``minimum(maximum(...))``, and JAX splits the
      gradient of a ``minimum`` tie **50/50** between its arguments, so the
      derivative comes back at exactly half its true value.
    * the same tie arises at the ends of the grid, where the clamp is genuinely
      needed; ``where`` selects a branch cleanly and carries the full gradient.

    This is not hypothetical.  The correction table's mass axis has a node at
    ``Sigma m_nu = 0.06`` eV, which is the fiducial, so every Fisher forecast
    differentiating the neutrino mass at the fiducial got a derivative wrong by
    a factor of two -- with no symptom other than the number.
    """
    lo, hi = grid[0], grid[-1]
    x_c = jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))
    i = jnp.clip(jnp.searchsorted(grid, x_c) - 1, 0, grid.size - 2)
    t = (x_c - grid[i]) / (grid[i + 1] - grid[i])
    return i, t


def pchip_slopes(y, x):
    """Monotone cubic (Fritsch-Carlson) slopes along the leading axis.

    Linear interpolation is C0, so its derivative **jumps at every node**.  That
    matters on every sparse axis of the correction table, and every axis of that
    shape gets the same treatment:

    * ``f_nu``.  The fiducial ``Sigma m_nu = 0.06`` eV is a node, where the left
      and right slopes differ by about 5 percent.
    * ``z``.  The round numbers a redshift is most likely to be evaluated at are
      exactly the kinks.  ``d ln(dn/dM)/dz`` was wrong by 2.6e-4 at ``z = 0.5``
      and 1.0e-3 at ``z = 1.0`` under linear interpolation, against 1e-10 at an
      off-node redshift --
      step-independently, which is what distinguishes a kink from a truncation
      error.
    * ``w0``, ``wa``.  Five nodes each, and the fiducial ``w0 = -1``, ``wa = 0``
      sits on a node of both.  Same argument, same treatment.

    Autodiff returns one side's slope while a central difference straddles the
    kink and averages both, so the disagreement is not noise: it is the
    interpolant having two derivatives there and the two methods picking
    differently.

    Monotone cubic Hermite is C1, so the derivative is continuous everywhere,
    and monotone, so it cannot introduce oscillations between sparse nodes --
    which matters because the interpolated quantity is a log-ratio that must
    stay physical.  Slopes are computed once, at load time, in numpy.
    """
    y = np.asarray(y)
    x = np.asarray(x)
    if y.shape[0] == 1:
        # A degenerate axis (one node) has no secant; a flat slope is the only
        # consistent answer and keeps the Hermite form well defined.
        return np.zeros_like(y)
    h = np.diff(x)                                            # (n-1,)
    shape = (len(h),) + (1,) * (y.ndim - 1)
    delta = np.diff(y, axis=0) / h.reshape(shape)             # secants
    d = np.zeros_like(y)
    if y.shape[0] > 2:
        # Interior: weighted harmonic mean, zeroed where the secants change sign.
        h0 = h[:-1].reshape((len(h) - 1,) + (1,) * (y.ndim - 1))
        h1 = h[1:].reshape((len(h) - 1,) + (1,) * (y.ndim - 1))
        s0, s1 = delta[:-1], delta[1:]
        same = s0 * s1 > 0
        w0, w1 = 2.0 * h1 + h0, h1 + 2.0 * h0
        with np.errstate(divide="ignore", invalid="ignore"):
            harm = (w0 + w1) / (w0 / np.where(s0 == 0, 1.0, s0)
                                + w1 / np.where(s1 == 0, 1.0, s1))
        d[1:-1] = np.where(same, harm, 0.0)
    d[0], d[-1] = delta[0], delta[-1]                         # one-sided ends
    return d


def hermite(y0, y1, d0, d1, h, t):
    """Cubic Hermite on one interval, evaluated at fraction ``t``."""
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y0
            + (t3 - 2 * t2 + t) * h * d0
            + (-2 * t3 + 3 * t2) * y1
            + (t3 - t2) * h * d1)


# ==========================================================================
# N-axis interpolation
# ==========================================================================
def cascaded_arrays(cube, axes_grids):
    """``[cube, d/dx0, d/dx1, ...]`` -- ``N+1`` arrays, not :math:`2^N`.

    A full tensor-product Hermite over ``N`` sparse axes needs every mixed
    partial, which is :math:`2^N` copies of the cube.  For the correction
    table's four axes over a 1.8-million-element cube that is 230 MB resident,
    for a table a likelihood loads on import.  This is the affordable
    construction: the cube plus one first derivative per axis.

    What it gives up is the *mixed* partials, so the result is C1 in each axis
    separately rather than jointly.  That is the property a Fisher matrix
    actually consumes -- first derivatives, one parameter at a time -- and the
    place it could still bite is measured directly in
    ``tests/test_ratio.py::test_derivative_is_continuous_in_every_axis``
    rather than argued about here.
    """
    n = len(axes_grids)
    out = [np.asarray(cube)]
    for j in range(n):
        moved = np.moveaxis(out[0], j, 0)
        out.append(np.moveaxis(pchip_slopes(moved, axes_grids[j]), 0, j))
    return out


def interp_cascaded(arrays, grids, queries):
    """Collapse each sparse axis in turn, leading axis first.

    The value is interpolated along axis *j* by Hermite, using that axis's own
    precomputed slopes -- which is what makes the result C1 in *j*.  The slope
    arrays for the axes still to come are carried along linearly, because the
    alternative is to carry their cross-derivatives, which is the
    :math:`2^N` construction this exists to avoid.
    """
    val = arrays[0]
    ders = list(arrays[1:])
    for j in range(len(grids)):
        i, t = lin_weights(queries[j], grids[j])
        h = grids[j][i + 1] - grids[j][i]
        d = ders[0]
        val = hermite(val[i], val[i + 1], d[i], d[i + 1], h, t)
        # The remaining slopes lose their leading axis the cheap way.  The
        # Hermite basis coefficients multiplying them vanish at a node, so this
        # approximation is exactly inert wherever the query sits on a grid
        # point -- which is where the fiducial sits in every axis.
        ders = [dd[i] + t * (dd[i + 1] - dd[i]) for dd in ders[1:]]
    return val


def tensor_arrays(cube, axes_grids):
    r"""Every mixed partial: a dense mask-indexed list of :math:`2^N` arrays.

    Bit *j* of the index means the array has been differentiated along sparse
    axis *j*; entry 0 is the cube itself.  A list rather than a dict because
    this is handed to a jitted function, and an ``int`` key in a pytree becomes
    a tracer -- the ordering has to live in the structure, not in the leaves.

    This is the expensive construction and the correct one.  :func:`interp_cascaded`
    is C1 in each axis *at a node* but not between nodes: carrying the
    still-to-come slope arrays linearly leaves the value depending on the
    collapsed axis through a C0 term, and the measured cost of that is a
    1.6 percent jump in :math:`\partial/\partial w_0` at an off-node
    :math:`w_0`.  With the mixed partials there is no such term.
    """
    n = len(axes_grids)
    out = [None] * (1 << n)
    out[0] = np.asarray(cube)
    dtype = out[0].dtype
    for mask in range(1, 1 << n):
        j = (mask & -mask).bit_length() - 1          # lowest set bit
        parent = out[mask & ~(1 << j)]
        moved = np.moveaxis(parent, j, 0)
        d = np.moveaxis(pchip_slopes(moved, axes_grids[j]), 0, j)
        # Back to the cube's dtype.  `pchip_slopes` divides by a float64 grid
        # spacing and so promotes; left alone, 2^N float64 copies of a float32
        # cube is the difference between 186 MB resident and 371 MB, for a
        # quantity whose useful precision is nowhere near either.
        out[mask] = d.astype(dtype, copy=False)
    return out


def interp_tensor(arrays, grids, queries):
    """Collapse every sparse axis by Hermite, leading axis first.

    Each step pairs every array not differentiated along the current leading
    axis (even mask) with the one that is (mask | 1) -- exactly the Hermite
    value/slope pair -- and halves the list.  After ``N`` steps one array
    remains.

    Cheapest-axis-first is why this is affordable inside a likelihood:
    interpolating the whole cube onto the output grid directly would cost
    ``O(prod(n_axes) * n_out)``.
    """
    cur = list(arrays)
    for j in range(len(grids)):
        i, t = lin_weights(queries[j], grids[j])
        h = grids[j][i + 1] - grids[j][i]
        cur = [hermite(cur[m][i], cur[m][i + 1],
                       cur[m | 1][i], cur[m | 1][i + 1], h, t)
               for m in range(0, len(cur), 2)]
    return cur[0]
