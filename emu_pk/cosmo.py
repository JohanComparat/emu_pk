r"""Density conventions, matching ``ggah_mod.cosmology`` exactly.

Small and duplicated on purpose.  This package cannot import ``ggah_mod`` --
that would be a cycle -- but a correction table built under a *different*
neutrino convention from the code that reads it is wrong in a way no test on
either side can see.  So the conventions are written out here, and
``tests/test_conventions.py`` asserts they agree with ``ggah_mod``'s whenever
that package happens to be importable.

The one that matters is which quantity :math:`\Omega_m` names.
"""

from __future__ import annotations

__all__ = ["NU_DENOM_EV", "N_EFF", "N_NU_MASSIVE", "NCDM_UR_PER_SPECIES",
           "T_CMB", "K_PIVOT", "PLANCK18", "omega_nu", "f_nu", "class_params"]

#: The 93.14 eV convention, ``Omega_nu = sum_mnu / (93.14 h^2)``.  A
#: *convention*,
#: 0.53 percent from the exact Fermi-Dirac integral, and committed to
#: everywhere for exactly that reason -- mixing the two is how a density budget
#: stops closing.
NU_DENOM_EV = 93.14
N_EFF = 3.044
N_NU_MASSIVE = 3
#: The ultra-relativistic equivalent of one massive species in CLASS's
#: bookkeeping.  ``N_ur = N_eff - 3 * 1.0132`` is the split that matches CLASS
#: to CAMB; giving the single non-cold species the full degeneracy instead
#: roughly doubles their disagreement.
NCDM_UR_PER_SPECIES = 1.0132
T_CMB = 2.7255

#: Primordial pivot, in **1/Mpc** -- CLASS's unit, not this package's h/Mpc.
#:
#: It is CLASS's own default value, but the training target is ``ln P`` with
#: the primordial power law divided out (see :func:`emu_pk.train.reduce_target`),
#: which puts the pivot in the *inference* path, and a pivot there cannot be a
#: default: change CLASS's and every shipped weight file silently means a
#: different spectrum, with nothing raising.  So it is stated here, passed to
#: CLASS explicitly, and written into the ``.npz`` beside the weights trained
#: against it.
K_PIVOT = 0.05

#: Planck 2018 TT,TE,EE+lowE+lensing, the fiducial the correction is built at.
PLANCK18 = {"Omega_m": 0.3100, "Omega_b": 0.0493, "h": 0.6736,
            "n_s": 0.9649, "ln10A_s": 3.044}


def omega_nu(sum_mnu: float, h: float) -> float:
    r""":math:`\Omega_\nu = \Sigma m_\nu / (93.14\,h^2)`."""
    return sum_mnu / (NU_DENOM_EV * h * h)


def f_nu(sum_mnu: float, h: float, Omega_m: float) -> float:
    r""":math:`f_\nu = \Omega_\nu/\Omega_m`, the axis the table is indexed on.

    Indexed on the *fraction* rather than on the mass so one table serves every
    :math:`h` and :math:`\Omega_m` instead of being tied to the cosmology it was
    built at.
    """
    return omega_nu(sum_mnu, h) / Omega_m


def class_params(*, h, omega_b, omega_cdm, n_s, ln10A_s, sum_mnu=0.0,
                 w0=-1.0, wa=0.0, k_max_h=200.0, z_max=5.0, T_cmb=T_CMB):
    """The CLASS input dict, mirroring ``ggah_mod.cosmology.power.ClassPk``.

    Physical densities in, so nothing here has to decide what ``Omega_m``
    means; the caller does that once.

    Four settings are explicit rather than left to CLASS's defaults, and each
    is a statement:

    * ``Omega_k = 0``.  ``ggah_mod`` is flat throughout and configures CAMB with
      ``omk=0.0``, but left CLASS on its default -- the same assumption, stated
      in one place and implied in the other.  Stated in both here.
    * ``non linear = none``.  The non-linear spectrum in ``ggah_mod`` is
      assembled by the halo model, not fitted; a halofit correction leaking into
      the training set would be silently absorbed into the network.
    * ``use_ppf = yes``.  The sampling box contains ``w(a)`` that cross -1, and
      the fluid parameterisation is singular there without PPF.
    * ``k_pivot``.  Also CLASS's default, but the training target divides the
      primordial power law out analytically, so the pivot is now a term in the
      predictor rather than a detail of the solver.  See :data:`K_PIVOT`.
    """
    params = {
        "output": "mPk",
        "non linear": "none",
        "P_k_max_h/Mpc": k_max_h * 1.05,
        "z_max_pk": float(max(z_max, 1.0)),
        "Omega_k": 0.0,
        "h": float(h),
        "omega_b": float(omega_b),
        "omega_cdm": float(omega_cdm),
        "n_s": float(n_s),
        "ln10^{10}A_s": float(ln10A_s),
        "T_cmb": float(T_cmb),
        "k_pivot": float(K_PIVOT),
    }
    if sum_mnu > 0.0:
        params.update({
            "N_ncdm": 1,
            "deg_ncdm": float(N_NU_MASSIVE),
            "m_ncdm": float(sum_mnu) / N_NU_MASSIVE,
            "N_ur": N_EFF - N_NU_MASSIVE * NCDM_UR_PER_SPECIES,
        })
    else:
        params["N_ur"] = N_EFF
    if (w0, wa) != (-1.0, 0.0):
        params.update({"Omega_Lambda": 0.0, "w0_fld": float(w0),
                       "wa_fld": float(wa), "use_ppf": "yes"})
    return params
