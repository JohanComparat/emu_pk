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
           "T_CMB", "K_PIVOT", "PLANCK18", "omega_nu", "f_nu", "class_params",
           "CAMB_PRECISION", "nu_energy_factor", "camb_params"]

#: ``Omega_nu = sum_mnu / (NU_DENOM_EV h^2)``: the neutrinos' rest mass, the
#: pressureless limit of the density CLASS integrates for three massive states
#: at its default ``T_ncdm = 0.71611`` -- the temperature the training solves
#: use.
#:
#: ``ggah_mod`` derives it (``ggah_mod.cosmology.constants.NU_DENOM_EV``,
#: 0.9.8) from CODATA and that temperature, and this is its float, restated
#: because this package cannot import that one.  It reaches the correction
#: tables and the validation, which call :func:`omega_nu`; the network does not
#: read it.  ``tests/test_conventions.py`` compares the two floats exactly.
NU_DENOM_EV = 93.14338613172058
N_EFF = 3.044
N_NU_MASSIVE = 3
#: The ultra-relativistic equivalent of one massive species in CLASS's
#: bookkeeping.  ``N_ur = N_eff - 3 * 1.0132`` is the split that matches CLASS
#: to CAMB; giving the single non-cold species the full degeneracy instead
#: roughly doubles their disagreement.
NCDM_UR_PER_SPECIES = 1.0132
T_CMB = 2.7255

#: Primordial pivot, in **1/Mpc** -- the solvers' unit, not this package's h/Mpc.
#:
#: It is CAMB's and CLASS's own default value, but the training target is
#: ``ln P`` with the primordial power law divided out (see
#: :func:`emu_pk.train.reduce_target`), which puts the pivot in the *inference*
#: path, and a pivot there cannot be a default: change a solver's and every
#: shipped weight file silently means a different spectrum, with nothing
#: raising.  So it is stated here, passed to both solvers explicitly, and
#: written into the ``.npz`` beside the weights trained against it.
K_PIVOT = 0.05

#: Planck 2018 TT,TE,EE+lowE+lensing, the fiducial the correction is built at.
PLANCK18 = {"Omega_m": 0.3100, "Omega_b": 0.0493, "h": 0.6736,
            "n_s": 0.9649, "ln10A_s": 3.044}


def omega_nu(sum_mnu: float, h: float) -> float:
    r""":math:`\Omega_\nu = \Sigma m_\nu / (93.143\,h^2)`, the rest mass."""
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
                 nu_r2=1.0 / 3.0, k_max_h=300.0, z_max=5.0, T_cmb=T_CMB):
    """The CLASS input dict, mirroring ``ggah_mod.cosmology.power.ClassPk``.

    Physical densities in, so nothing here has to decide what ``Omega_m``
    means; the caller does that once.

    Four settings are explicit rather than left to CLASS's defaults, and each
    is a statement:

    * ``Omega_k``.  A *sampled* parameter, passed through rather than assumed.
      It is stated rather than left to CLASS's default because a default
      here is a cosmology nobody wrote down.  Positive is open, CLASS's
      convention, CAMB's and ``ggah_mod``'s.

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
      ``sum_mnu = 0.10`` eV in an inverted ordering -- five times the
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
      primordial power law out analytically, so the pivot is a term in the
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



# --------------------------------------------------------------------------
# CAMB: the generator's solver from 2.1.0
# --------------------------------------------------------------------------
# ggah_mod_benchmark's precision scan (scripts/57_precision_scan.py, commit
# 516eb0a) measured the shipped training set -- CLASS 3.3.4 at its defaults --
# 0.41 % from CLASS's own converged answer in the median and 0.81 % at worst
# over this box, smooth in the parameters, so the network learnt it and its
# validation against the same CLASS could not see it.  No affordable CLASS
# setting fixes the heavy-neutrino end (the ncdm fluid approximation dominates
# above 0.3 eV); CAMB at the settings below is 0.039 % from its own reference
# and 0.12 % from CLASS's -- the floor between the two codes -- for 2.5x the
# old generation cost.  They are ggah_mod 0.9.8's CAMB_PRECISION, so the
# emulator and ggah_mod's CambPk solve the same thing by construction.
CAMB_PRECISION = {"lAccuracyBoost": 3.0, "AccuracyBoost": 2.0,
                  "DoLateRadTruncation": False, "WantCls": False}

#: The neutrino convention CLASS integrates by default and ggah_mod 0.9.8
#: adopted: three massive states at ``T_ncdm = 0.71611 T_CMB``, each of
#: degeneracy ``(T_ncdm/T_nu)^4`` in CAMB's spelling, and the massless
#: remainder ``N_eff - 3 (T_ncdm/T_nu)^4 = 0.004395``.  Restated from
#: ``ggah_mod.cosmology.constants``; ``tests/test_camb_input.py`` compares
#: what :func:`camb_params` builds with ``ggah_mod``'s ``camb_input``.
T_NCDM_OVER_T_GAMMA = 0.71611
T_NU_OVER_T_GAMMA = (4.0 / 11.0) ** (1.0 / 3.0)
N_MASSIVE_EFF = N_NU_MASSIVE * (T_NCDM_OVER_T_GAMMA / T_NU_OVER_T_GAMMA) ** 4
N_UR_REMAINDER = N_EFF - N_MASSIVE_EFF
K_B_EV_PER_K = 8.617333262e-5
_SIGMA_SB, _C_M_S, _G_SI, _MPC_M = 5.670374419e-8, 299_792_458.0, 6.67430e-11, 3.0856775814913673e22
_RHO_CRIT_100_SI = 3.0 * (1.0e5 / _MPC_M) ** 2 / (8.0 * 3.141592653589793 * _G_SI)
OMEGA_GAMMA_H2 = 4.0 * _SIGMA_SB * T_CMB ** 4 / _C_M_S ** 3 / _RHO_CRIT_100_SI
NU_REL_COEF = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)

_FD_EDGES = (0.0, 0.5, 2.0, 6.0, 16.0, 40.0)
_FD_NODES_PER_PANEL = (10, 10, 10, 12, 12)


def _fd_rule():
    import numpy as np
    xs, ws = [], []
    for a, b, n in zip(_FD_EDGES[:-1], _FD_EDGES[1:], _FD_NODES_PER_PANEL):
        xg, wg = np.polynomial.legendre.leggauss(n)
        xs.append(0.5 * (b - a) * (xg + 1.0) + a)
        ws.append(0.5 * (b - a) * wg)
    x = np.concatenate(xs)
    wf = np.concatenate(ws) * x ** 2 / (np.exp(x) + 1.0)
    return x, wf / np.sum(wf * x)


def nu_energy_factor(y):
    r"""Fermi-Dirac energy of one species relative to massless, :math:`F(y)`.

    ``ggah_mod.cosmology.parameters.nu_energy_factor`` in numpy: the same
    54-node rule, summed without the cancellation, so :math:`F(0) = 1`
    exactly and :math:`F \to \kappa y` when cold.  ``y = m / k_B T_ncdm``.
    """
    import numpy as np
    x, w = _fd_rule()
    y = np.asarray(y, dtype=float)[..., None]
    y2 = y * y
    return 1.0 + np.sum(w * y2 / (np.sqrt(x * x + y2) + x), axis=-1)


_CAMB_MATTER_KEYS = ("k_per_logint", "accurate_massive_neutrino_transfers")
_CAMB_TOP_KEYS = ("WantCls", "DoLateRadTruncation")


def camb_params(*, h, omega_b, omega_cdm, n_s, ln10A_s, sum_mnu=0.0,
                w0=-1.0, wa=0.0, Omega_k=0.0, nu_r1=1.0 / 3.0,
                nu_r2=1.0 / 3.0, k_max_h=300.0, redshifts=(0.0,),
                precision=None, T_cmb=T_CMB):
    """A ``CAMBparams`` for one box point: the generator's solve.

    The keyword names are :data:`emu_pk.box.PARAMS`, as for
    :func:`class_params`, and the physics is the same point: ``omch2`` is the
    ``omega_cdm`` CLASS is given, the three masses are ``r_i * sum_mnu``, and
    the neutrinos are CLASS's in CAMB's interface (ggah_mod 0.9.8's
    ``camb_input``).  CAMB describes each massive state by a degeneracy and a
    share of ``omnuh2``, the states' density *today* -- rest mass and kinetic
    energy -- and infers each mass from its share.  So it is handed each
    state's exact density from :func:`nu_energy_factor`, never the mass
    shares: a mass share read as a density share puts a normal ordering's
    lightest state 20 per cent light.

    ``precision`` is :data:`CAMB_PRECISION` when ``None`` and CAMB's own
    defaults when ``{}``; keys are routed as ``ggah_mod`` routes them, and an
    unknown one raises.  Dark energy is CPL through CAMB's PPF module, as the
    CLASS path uses CLASS's; the pivot is :data:`K_PIVOT` [1/Mpc], CAMB's
    default, stated.
    """
    import camb
    import numpy as np

    s = dict(CAMB_PRECISION if precision is None else precision)
    pars = camb.CAMBparams()
    massive = float(sum_mnu) > 0.0
    pars.set_cosmology(H0=100.0 * float(h), ombh2=float(omega_b),
                       omch2=float(omega_cdm), mnu=float(sum_mnu),
                       num_massive_neutrinos=N_NU_MASSIVE if massive else 0,
                       nnu=N_EFF, TCMB=float(T_cmb), omk=float(Omega_k))
    if massive:
        r = np.array([float(nu_r1), float(nu_r2), 1.0 - float(nu_r1) - float(nu_r2)])
        m = r * float(sum_mnu)
        y = m / (K_B_EV_PER_K * T_NCDM_OVER_T_GAMMA * float(T_cmb))
        omega_gamma_h2 = OMEGA_GAMMA_H2 * (float(T_cmb) / T_CMB) ** 4
        per_state_h2 = NU_REL_COEF * N_MASSIVE_EFF * omega_gamma_h2 / N_NU_MASSIVE
        rho_h2 = per_state_h2 * nu_energy_factor(y)
        pars.nu_mass_eigenstates = N_NU_MASSIVE
        pars.nu_mass_numbers = [1] * N_NU_MASSIVE
        pars.nu_mass_degeneracies = [N_MASSIVE_EFF / N_NU_MASSIVE] * N_NU_MASSIVE
        pars.num_nu_massless = N_UR_REMAINDER
        pars.omnuh2 = float(rho_h2.sum())
        pars.nu_mass_fractions = list(rho_h2 / rho_h2.sum())
    pars.InitPower.set_params(As=float(np.exp(ln10A_s)) * 1e-10, ns=float(n_s),
                              pivot_scalar=K_PIVOT)
    if (float(w0), float(wa)) != (-1.0, 0.0):
        pars.set_dark_energy(w=float(w0), wa=float(wa), dark_energy_model="ppf")
    matter = {k: s.pop(k) for k in _CAMB_MATTER_KEYS if k in s}
    pars.set_matter_power(redshifts=sorted(map(float, redshifts), reverse=True),
                          kmax=float(k_max_h) * 1.05 * float(h), silent=True,
                          **matter)
    pars.NonLinear = camb.model.NonLinear_none
    for key in _CAMB_TOP_KEYS:
        if key in s:
            setattr(pars, key, bool(s.pop(key)))
    for key, val in s.items():
        if not hasattr(pars.Accuracy, key):
            raise KeyError(f"{key!r} is not a CAMB accuracy parameter")
        setattr(pars.Accuracy, key, val)
    return pars
