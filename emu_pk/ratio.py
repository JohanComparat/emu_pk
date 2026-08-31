r"""The CLASS-distilled correction to a massless-LambdaCDM linear spectrum.

A network trained on massless LambdaCDM cannot be asked for a spectrum with
neutrinos in it, and gets no response at all to dark energy.  This module
supplies both corrections, as ratios measured from CLASS on a grid and
interpolated:

.. math::

    r_m (k,z) = \frac{P_m (k,z;\,\Sigma m_\nu, w_0, w_a)}{P_m(k,z;\,0,-1,0)},
    \qquad
    r_{cb}(k,z) = \frac{P_{cb}(k,z;\,\Sigma m_\nu, w_0, w_a)}{P_m(k,z;\,0,-1,0)}

The neutrino part is indexed by :math:`f_\nu = \Omega_\nu/\Omega_m` rather than
by :math:`\Sigma m_\nu`, so one table serves every :math:`h` and
:math:`\Omega_m` instead of being tied to the cosmology it was built at.

Both ratios are **exactly 1 at the LambdaCDM massless corner** -- the stored
log-ratio is exactly zero at :math:`f_\nu = 0`, :math:`w_0 = -1`, :math:`w_a =
0` -- so the correction is applied unconditionally, with no ``if sum_mnu > 0``
branch.  That is not tidiness: a Python branch on :math:`\Sigma m_\nu` is a
branch on a value that is a tracer under ``jax.grad``, and would break the
gradient the whole differentiable path exists to provide.

Factorisation
-------------
The correction ships as **two factors**, neutrinos and dark energy,

.. math::  r(k,z;\,f_\nu,w_0,w_a) \simeq r^{\nu}(k,z;f_\nu)\,r^{\rm DE}(k,z;w_0,w_a)

because the full five-axis cube needs 16 derivative arrays for a tensor-product
Hermite and the factors need 4 and 8 over much smaller cubes -- megabytes
against hundreds.  Whether that is *allowed* is not assumed:
:func:`~emu_pk.assemble.build_ratio` runs
the full grid regardless, measures the largest residual of the factorisation
against it, and stores the number in the table as ``resid_max``.  If it is not
comfortably below the emulator's own shape error the factorisation is the wrong
call, and the number is there to say so rather than to be trusted.

Validity
--------
Outside the grid the interpolation would clamp, silently understating the
correction, so :func:`suppression_m` raises instead.  A plausible wrong number
is worse than an exception.  The check calls ``float()``, which raises on a
tracer, so it is **skipped** under tracing rather than attempted -- a jitted
forward model is checked once when it is built, outside the trace.

The table is regenerated in two steps, with ``classy`` installed -- solve the
grid, then assemble it::

    python -m emu_pk.generate --mode ratio --shard 0 --n-per-shard 300 --out shards_ratio
    python -m emu_pk.assemble --mode ratio --shards shards_ratio

The shipped ``.npz`` means ordinary use never needs either.
"""

from __future__ import annotations

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from .interp import concrete, interp_tensor, tensor_arrays

__all__ = ["load", "suppression_m", "suppression_cb",
           "MNU_MAX", "Z_MAX", "W0_RANGE", "WA_RANGE"]

_NPZ = pathlib.Path(__file__).resolve().parent / "data" / "class_pk_ratio.npz"

# Validity limits.  These are *hand-written constants*, not read from the
# table: a caller has to be able to ask where the correction is valid without
# paying to load a 13 MB array first, and `_refuse_outside` enforces these and
# not the axes.  So they can disagree with what is actually on disk, and once
# did -- the shipped file carried one set of mass nodes while the builder's
# defaults produced another.  `tests/test_ratio.py` asserts the two agree,
# which is the only thing keeping this comment true.  The builder is
# `emu_pk.assemble.build_ratio`.

#: Largest summed neutrino mass the table covers, in eV.
MNU_MAX = 0.60

#: Largest redshift the table covers.
Z_MAX = 5.0

#: Range of ``w0`` the table covers, as ``(low, high)``.  Narrower than
#: ``emu_pk.box``'s own bounds: the table was built for a different purpose and
#: its grid was sized for that.
W0_RANGE = (-1.30, -0.70)

#: Range of ``wa`` the table covers, as ``(low, high)``.
WA_RANGE = (-0.70, 0.50)


@jax.jit
def _evaluate(arrays, grids, lnk_grid, lnk, queries):
    """Jitted kernel.  Every leaf is an array, so nothing here becomes static.

    Sparse axes collapse by cascaded Hermite; the wavenumber axis is linear in
    ``ln k``, because it carries 400 nodes over six decades of a smooth
    broadband ratio and the sparse-node argument that motivates Hermite
    elsewhere does not apply to it.
    """
    return jnp.interp(lnk, lnk_grid, interp_tensor(arrays, grids, queries))


# ==========================================================================
# Loading
# ==========================================================================
def load(path=None) -> dict:
    """Resolve the path first, then hit the cache.

    ``lru_cache`` keys on the call signature, so ``load()`` and ``load(None)``
    are two different keys and would build -- and hold -- two independent
    copies of a table that is over a hundred megabytes once its derivative
    arrays exist.  Resolving to a single canonical string before the cache is
    what makes them the same call.
    """
    return _load_resolved(str(_NPZ if path is None else pathlib.Path(path)))


@functools.lru_cache(maxsize=2)
def _load_resolved(path: str) -> dict:
    """The distilled table, cached, with every derivative array precomputed.

    Held as **numpy**, not ``jnp``.  A ``jnp`` array cached during a ``jit``
    trace carries that trace with it, and the next transformation dies with an
    ``UnexpectedTracerError``; converting at the point of use folds the values
    into the jaxpr as constants instead, which is also faster eagerly.

    Reads both layouts.  The three-axis table from ``ggah_mod``
    (``f_nu, z, lnk``, no dark energy) loads as a neutrino factor with the
    dark-energy factor absent, so it means the same thing in both packages.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing.  Regenerate it with `python -m emu_pk.generate "
            "--mode ratio` and then `python -m emu_pk.assemble --mode ratio` "
            "(both need classy).")
    with np.load(p) as d:
        raw = {k: np.asarray(d[k]) for k in d.files}

    tab = {"lnk": raw["lnk"], "mnu": raw.get("mnu"),
           "resid_max": float(raw["resid_max"]) if "resid_max" in raw else None}

    if "w0" in raw:
        # The full cube.  A factorised form is not accurate enough: its cross
        # term reaches 1.6 percent at high neutrino mass with strongly
        # non-LambdaCDM dark energy, an order of magnitude above the emulator's
        # own shape error, because the two effects genuinely couple -- more
        # late-time growth means more time for free streaming to suppress.
        grids = [raw["f_nu"], raw["w0"], raw["wa"], raw["z"]]
        tab["axes"] = ("f_nu", "w0", "wa", "z")
    else:
        # The three-axis table from ggah_mod: neutrinos only, no dark energy.
        # It loads unchanged, so a spectrum computed against it here matches
        # one computed against it there bit for bit.
        grids = [raw["f_nu"], raw["z"]]
        tab["axes"] = ("f_nu", "z")

    tab["grids"] = grids
    # The cubes are kept raw and their 2^N derivative arrays are built on first
    # use, per spectrum.  A run that only ever asks for the total matter
    # spectrum pays for one set, not two -- which halves what every worker of a
    # parallel chain holds resident.
    tab["_cubes"] = {w: raw[w] for w in ("lnr_m", "lnr_cb")}
    tab["_derivs"] = {}
    for k in ("h_fid", "Omega_m_fid"):
        if k in raw:
            tab[k] = float(raw[k])
    return tab


def _arrays(tab, which):
    """The ``2^N`` mixed partials for one spectrum, built once and cached."""
    if which not in tab["_derivs"]:
        tab["_derivs"][which] = tensor_arrays(tab["_cubes"][which], tab["grids"])
    return tab["_derivs"][which]


def _limits(tab):
    """Validity limits read off the loaded table rather than hard-coded."""
    lim = {name: (float(g[0]), float(g[-1]))
           for name, g in zip(tab["axes"], tab["grids"])}
    if tab.get("mnu") is not None:
        lim["sum_mnu"] = (float(tab["mnu"][0]), float(tab["mnu"][-1]))
    return lim


def _check(tab, queries):
    """Refuse to extrapolate.  Clamping would understate the correction.

    Skipped under tracing, where the values are not available: attempting it
    would raise ``ConcretizationTypeError`` and break the gradient.  A jitted
    forward model is checked once when it is built, outside the trace.
    """
    lim = _limits(tab)
    for name, v in zip(tab["axes"], queries):
        c = concrete(jnp.max(jnp.asarray(v)))
        if c is None or name not in lim:
            continue
        lo, hi = lim[name]
        if not lo <= c <= hi:
            raise ValueError(
                f"{name} = {c:.6g} is outside the distilled table's range "
                f"[{lo:g}, {hi:g}].  Interpolation would clamp to the edge and "
                f"silently understate the correction; use the CLASS backend, "
                f"which is exact, or rebuild the table over a wider grid.")


def _ratio(which: str, k, f_nu, z=0.0, w0=-1.0, wa=0.0, table=None):
    tab = load(table)
    z = jnp.asarray(z, dtype=float)
    if tab["axes"][1] == "w0":
        queries = [jnp.asarray(f_nu, dtype=float), jnp.asarray(w0, dtype=float),
                   jnp.asarray(wa, dtype=float), z]
    else:
        queries = [jnp.asarray(f_nu, dtype=float), z]
    _check(tab, queries)
    return jnp.exp(_evaluate(
        [jnp.asarray(a) for a in _arrays(tab, which)],
        [jnp.asarray(g) for g in tab["grids"]],
        jnp.asarray(tab["lnk"]), jnp.log(jnp.asarray(k)), queries))


def suppression_m(k, f_nu, z=0.0, w0=-1.0, wa=0.0, table=None):
    r""":math:`P_m(k,z;\Sigma m_\nu,w_0,w_a)/P_m(k,z;0,-1,0)`.

    Exactly 1 at the LambdaCDM massless corner.
    """
    return _ratio("lnr_m", k, f_nu, z, w0, wa, table)


def suppression_cb(k, f_nu, z=0.0, w0=-1.0, wa=0.0, table=None):
    r""":math:`P_{cb}(k,z;\Sigma m_\nu,w_0,w_a)/P_m(k,z;0,-1,0)`.

    Note the denominator is the **massless total**, not the massive total: this
    converts a massless-trained spectrum straight to the cold one, in a single
    multiplication, without ever forming the massive total in between.
    """
    return _ratio("lnr_cb", k, f_nu, z, w0, wa, table)
