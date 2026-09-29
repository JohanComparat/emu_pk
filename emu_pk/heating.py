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

#: CLASS settings for the pair: ``ggah_mod_benchmark``'s ``fast`` rung, about
#: 2x looser than CLASS's defaults everywhere, and 5 per cent wrong in P(k)
#: itself.  In the ratio that cancels.  Measured at six extreme box points (the
#: fiducial; omega_cdm = 0.05; h = 0.85 with omega_b = 0.028; 0.6 eV;
#: Omega_k = -0.15; w0 = -0.5) against R from a default pair, over
#: 1 < k < 200 h/Mpc and six redshifts, the worst error on R is 7.7e-5 -- five
#: times inside the 4e-4 CAMB's own precision leaves -- for 0.69x a default
#: pair's cost.  Coarsening the k sampling as well was 2-3x faster and 1-2e-3
#: wrong near k = 175, which is not.
SETTINGS: dict = {"tol_perturbations_integration": 1e-4, "k_step_sub": 0.1,
                  "l_max_ncdm": 10, "l_max_ur": 10,
                  "tol_ncdm_synchronous": 1e-2, "q_linstep": 0.8}


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
    """The CLASS pair at one box point: heated and not, same settings."""
    from . import grid
    from .generate import solve
    s = dict(SETTINGS if settings is None else settings)
    base = {**cosmo.class_params(**theta,
                                 k_max_h=max(grid.K_MAX, float(np.max(k_h))),
                                 z_max=max(grid.Z_MAX, float(np.max(z_nodes)))),
            **s}
    pm1, pcb1 = solve(base, z_nodes, k_h)
    pm0, pcb0 = solve({**base, "reio_parametrization": "reio_none"}, z_nodes, k_h)
    return Heating(r_m=pm1 / pm0, r_cb=pcb1 / pcb0, pm_class=pm1, pcb_class=pcb1)
