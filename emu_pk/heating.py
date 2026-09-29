r"""Post-reionization baryon heating, carried from CLASS onto CAMB's spectrum.

From 2.1.0 the training spectra are CAMB's (:func:`emu_pk.cosmo.camb_params`),
and CAMB's linear :math:`P(k)` does not heat the baryons at reionization.  CLASS
does: the heated gas has pressure, and :math:`P_m` is suppressed below its
Jeans length -- 3 per cent at :math:`k = 200\,h\,{\rm Mpc}^{-1}` at the
fiducial, 13 per cent where :math:`\omega_{\rm cdm}` is lowest and the baryons
are a third of the matter (``ggah_mod_benchmark`` precision scan, the
``no_reio`` rung).  The 2.0 training set carried it, as a consequence of being
CLASS's, and 2.1.0 keeps it on purpose: each CAMB spectrum is multiplied by

.. math::

    R(k, z;\,\theta) = \frac{P^{\rm CLASS}(k, z;\,\theta)}
                            {P^{\rm CLASS}_{\rm no\ reio}(k, z;\,\theta)},

for :math:`P_m` and for :math:`P_{cb}` separately, from a CLASS pair at the
same point.  The reionization history is CLASS's default, which fixes
:math:`z_{\rm reio} = 7.6711` everywhere in the box (:math:`\tau = 0.0543` at
the fiducial) -- a convention, stated here because :math:`\tau` is not a
parameter of the emulator.

**Per point, because nothing smaller works.**  Moving one parameter at a time
to the box's edges changes :math:`R-1` at :math:`k = 200` by a factor 4 and
0.17 in :math:`\omega_{\rm cdm}`, :math:`-30`/:math:`+50` per cent in
:math:`h`, 17 per cent in :math:`\omega_b` and :math:`w_0`, 10 per cent in
:math:`\Omega_k` and a few per cent in :math:`w_a` and :math:`\Sigma m_\nu`.
Seven parameters is not a table.

**Cheap, because it is a ratio.**  Both solves share one set of settings, so
their errors cancel: tightening the neutrino tolerance a thousandfold moves
:math:`R` by :math:`\sim10^{-5}`.  :data:`SETTINGS` are therefore loosened well
below CLASS's defaults, at a measured cost in accuracy on :math:`R` that is
recorded beside them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cosmo

__all__ = ["SETTINGS", "Heating", "pair"]

#: CLASS settings for the pair: CLASS's own defaults.  What makes them enough
#: is where the ratio is taken -- see :func:`pair`.
#:
#: Measured against a pair tightened everywhere (``ggah_mod_benchmark``
#: ``scripts/58_kmax_extension.py``, grid to 300 h/Mpc, five redshifts) at the
#: fiducial, massless, 0.6 eV, omega_cdm = 0.05, closed, and a split-mass curved
#: w0wa point: within 6e-5 in 1e-3 < k < 200 everywhere, and within 3e-5 above
#: k = 200 except at omega_cdm = 0.05, 1.3e-4 where the suppression is 24 per
#: cent.  CAMB's own precision leaves about 4e-4.
#:
#: Refused on the way: the same pair with the ratio taken *after* CLASS's P(k)
#: spline (1.3e-3 above k = 200 -- two spectra of different shape interpolated
#: between nodes 11 a decade apart); that pair with the output sampled at 20 a
#: decade (as accurate, 2.4x the cost, because the extra nodes are the
#: expensive high-k solves); and a pair loosened 2x below the defaults (6.5e-4
#: massless at every k: with all 3.044 species relativistic the loosened
#: integration does not cancel).
SETTINGS: dict = {}


@dataclass(frozen=True)
class Heating:
    """One CLASS pair on the generator's grid, each array ``(n_z, n_k)``.

    ``r_m`` and ``r_cb`` are the heating ratios; ``pm_class`` is the heated
    CLASS spectrum itself, whose *shape* continues CAMB's below its first
    transfer mode (curved cosmologies, see :func:`emu_pk.generate.solve_camb`).
    """
    r_m: np.ndarray
    r_cb: np.ndarray
    pm_class: np.ndarray
    pcb_class: np.ndarray


def pair(theta: dict, z_nodes, k_h, settings: dict | None = None) -> Heating:
    """The CLASS pair at one box point: heated and not, same settings.

    **The ratio is taken on CLASS's own k nodes**, where its P(k) spline is
    exact, and only the ratio -- smooth in k, with no BAO and no turnover --
    is interpolated onto ``k_h``, in log-log.  Interpolating each spectrum
    first and dividing after leaves each one's spline error in the ratio, and
    above k = 200 h/Mpc the two spectra differ enough in shape for those
    errors to stop cancelling.  The two solves share their nodes (they differ
    only after reionization, which the k sampling does not see); that is
    checked, not assumed.
    """
    from classy import Class
    from scipy.interpolate import CubicSpline

    from . import grid
    s = dict(SETTINGS if settings is None else settings)
    z_nodes = np.atleast_1d(np.asarray(z_nodes, dtype=float))
    k_h = np.asarray(k_h, dtype=float)
    h = float(theta["h"])
    base = {**cosmo.class_params(**theta,
                                 k_max_h=max(grid.K_MAX, float(k_h.max())),
                                 z_max=max(grid.Z_MAX, float(z_nodes.max()))),
            **s}
    massive = bool(base.get("N_ncdm", 0))
    knat = None
    at_nodes = []
    for extra in ({}, {"reio_parametrization": "reio_none"}):
        cl = Class()
        cl.set({**base, **extra})
        try:
            cl.compute()
            _, k, _ = cl.get_pk_and_k_and_z(nonlinear=False)       # 1/Mpc
            if knat is None:
                knat = np.array(k, dtype=float)
                knat[-1] *= 1.0 - 1e-12                            # inside CLASS's bound
            elif not np.allclose(k[:-1], knat[:-1], rtol=1e-9, atol=0.0):
                raise RuntimeError("the heated and unheated CLASS solves "
                                   "sampled different k nodes")
            pm = np.array([[cl.pk_lin(kk, zz) for kk in knat] for zz in z_nodes])
            pcb = (np.array([[cl.pk_cb_lin(kk, zz) for kk in knat] for zz in z_nodes])
                   if massive else pm)
            if not extra:
                # The heated spectrum itself on the caller's grid: its shape
                # continues CAMB's below CAMB's first mode.
                kk_h = k_h * h
                pm_grid = np.array([[cl.pk_lin(q, zz) for q in kk_h] for zz in z_nodes]) * h ** 3
                pcb_grid = (np.array([[cl.pk_cb_lin(q, zz) for q in kk_h] for zz in z_nodes]) * h ** 3
                            if massive else pm_grid.copy())
        finally:
            cl.struct_cleanup()
            cl.empty()
        at_nodes.append((pm, pcb))
    ln_kn = np.log(knat / h)
    ln_k = np.log(k_h)

    def _ratio(i):
        lnr = np.log(at_nodes[0][i] / at_nodes[1][i])
        return np.exp(np.array([CubicSpline(ln_kn, row)(ln_k) for row in lnr]))

    return Heating(r_m=_ratio(0), r_cb=_ratio(1), pm_class=pm_grid, pcb_class=pcb_grid)
