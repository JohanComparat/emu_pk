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
                 w0=-1.0, wa=0.0, Omega_k=0.0, nu_r1=1.0 / 3.0,
                 nu_r2=1.0 / 3.0, k_max_h=200.0, z_max=5.0, T_cmb=T_CMB):
    """The CLASS input dict, mirroring ``ggah_mod.cosmology.power.ClassPk``.

    Physical densities in, so nothing here has to decide what ``Omega_m``
    means; the caller does that once.

    Four settings are explicit rather than left to CLASS's defaults, and each
    is a statement:

    * ``Omega_k``.  A *sampled* parameter, passed through rather than assumed.
      It is stated rather than left to CLASS's default for the same reason it
      always was: a default here is a cosmology nobody wrote down.  Positive is
      open, CLASS's convention and ``ggah_mod``'s.

      What follows from it is the thing to know.  CLASS closes the budget with
      whichever dark-energy component is left free -- ``Omega_Lambda`` in the
      LambdaCDM branch, ``Omega_fld`` once ``w0``/``wa`` engage the fluid below
      -- so ``Omega_de = 1 - Omega_k - Omega_m - Omega_r``, and over this box
      that goes **negative** in about 0.8 % of the design.  CLASS solves those
      without complaint; a negative dark-energy density is exotic, not
      ill-posed.  See :func:`emu_pk.box.sample` for why they are kept.
    * The **three neutrino masses**, ``m_i = nu_r_i * sum_mnu``.  CLASS is asked
      for three separate species rather than one carrying a degeneracy of
      three, because the oscillation experiments say the masses are not equal
      and the difference is not always negligible: measured against CLASS, the
      degenerate approximation is wrong by 0.32 % in ``P(k)`` at
      ``sum_mnu = 0.10`` eV in an inverted ordering -- three times the
      emulator's own median error -- and by under 0.012 % above 0.25 eV, where
      the masses really are nearly equal.  It is a good approximation exactly
      where it does not matter.

      ``N_ur`` does not move: it removes what three species would have
      contributed had they stayed relativistic, and there are still three.
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
        "Omega_k": float(Omega_k),
        "h": float(h),
        "omega_b": float(omega_b),
        "omega_cdm": float(omega_cdm),
        "n_s": float(n_s),
        "ln10^{10}A_s": float(ln10A_s),
        "T_cmb": float(T_cmb),
        "k_pivot": float(K_PIVOT),
    }
    if sum_mnu > 0.0:
        # Three *separate* species, not one with a degeneracy of three.  The
        # masses are `r_i * sum_mnu`; the default `r = (1/3, 1/3, 1/3)` is the
        # degenerate convention, so a caller that names no ratios gets the same
        # physics as before -- but through the general path, so there is only
        # one path to be right.
        #
        # `N_ncdm` is an int and not a string, deliberately: `generate.solve`
        # decides whether to ask CLASS for `pk_cb_lin` by the *truthiness* of
        # this value, and the string "0" is truthy.
        m = (float(nu_r1) * float(sum_mnu),
             float(nu_r2) * float(sum_mnu),
             (1.0 - float(nu_r1) - float(nu_r2)) * float(sum_mnu))
        params.update({
            "N_ncdm": N_NU_MASSIVE,
            "m_ncdm": ",".join(f"{x:.12g}" for x in m),
            # Unchanged, and it has to be: this removes what three species
            # would have contributed had they stayed relativistic, and there
            # are still three of them whatever their masses.
            "N_ur": N_EFF - N_NU_MASSIVE * NCDM_UR_PER_SPECIES,
        })
    else:
        params["N_ur"] = N_EFF
    if (w0, wa) != (-1.0, 0.0):
        params.update({"Omega_Lambda": 0.0, "w0_fld": float(w0),
                       "wa_fld": float(wa), "use_ppf": "yes"})
    return params
