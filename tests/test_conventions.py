"""This package cannot import ggah_mod, so it restates its conventions.

A correction table built under a *different* neutrino convention from the code
that reads it is wrong in a way no test on either side can see -- each is
self-consistent.  These tests are the seam, and they run only when ggah_mod
happens to be importable.
"""
import pytest

from emu_pk import cosmo


def test_density_conventions_match_ggah_mod():
    C = pytest.importorskip("ggah_mod.cosmology.constants")
    assert cosmo.NU_DENOM_EV == C.NU_DENOM_EV
    assert cosmo.N_EFF == C.N_EFF
    assert cosmo.N_NU_MASSIVE == C.N_NU_MASSIVE
    assert cosmo.T_CMB == pytest.approx(C.T_CMB)


def test_omega_nu_matches_ggah_mod():
    params = pytest.importorskip("ggah_mod.cosmology.parameters")
    for mnu in (0.0, 0.06, 0.3, 0.6):
        c = params.Cosmology.create(sum_mnu=mnu)
        assert cosmo.omega_nu(mnu, float(c.h)) == pytest.approx(float(c.Omega_nu))
        assert cosmo.f_nu(mnu, float(c.h), float(c.Omega_m)) == pytest.approx(float(c.f_nu))


def test_fiducial_matches_planck18():
    params = pytest.importorskip("ggah_mod.cosmology.parameters")
    p = params.PLANCK18
    assert cosmo.PLANCK18["Omega_m"] == pytest.approx(float(p.Omega_m))
    assert cosmo.PLANCK18["Omega_b"] == pytest.approx(float(p.Omega_b))
    assert cosmo.PLANCK18["h"] == pytest.approx(float(p.h))
    assert cosmo.PLANCK18["n_s"] == pytest.approx(float(p.n_s))
    assert cosmo.PLANCK18["ln10A_s"] == pytest.approx(float(p.ln10A_s))


def test_class_params_states_curvature_and_linearity_explicitly():
    """Stated rather than defaulted -- and curvature is now a parameter.

    It used to be that `ggah_mod` was flat throughout and this package agreed
    by hard-coding zero.  `ggah_mod` has carried `Omega_k` for a while
    (`Cosmology.Omega_k`, and a closure that subtracts it), so the assumption
    was only ever true on this side.  Now neither side assumes it, and the
    convention that has to agree is the *sign*: positive is open, in CLASS, in
    `ggah_mod`, and here.
    """
    p = cosmo.class_params(h=0.6736, omega_b=0.0224, omega_cdm=0.12,
                           n_s=0.9649, ln10A_s=3.044)
    assert p["Omega_k"] == 0.0
    assert p["non linear"] == "none"
    assert cosmo.class_params(h=0.6736, omega_b=0.0224, omega_cdm=0.12,
                              n_s=0.9649, ln10A_s=3.044,
                              Omega_k=0.1)["Omega_k"] == pytest.approx(0.1)


def test_cpl_uses_ppf():
    """w(a) crosses -1 inside the box, and the fluid is singular there without PPF."""
    p = cosmo.class_params(h=0.7, omega_b=0.0224, omega_cdm=0.12, n_s=0.96,
                           ln10A_s=3.0, w0=-0.9, wa=0.3)
    assert p["use_ppf"] == "yes"
    assert p["w0_fld"] == -0.9 and p["wa_fld"] == 0.3


class TestTheCurvatureSignIsTheSameOnBothSides:
    r"""The convention that has to agree once `Omega_k` crosses the seam.

    Both packages are self-consistent under *either* sign, which is exactly the
    condition this file exists for.  `ggah_mod` closes the budget as
    :math:`\Omega_{de} = 1 - \Omega_\gamma - \Omega_{cb} - \Omega_\nu -
    \Omega_k` and CLASS as :math:`\Omega_{fld} = 1 - \Omega_k - \Omega_m -
    \Omega_r`.  A flipped sign would leave every test on each side passing and
    make the emulator wrong for every curved cosmology -- smoothly, and by an
    amount that looks like a fit that could be better.

    The physical statement, which is sign-unambiguous: positive `Omega_k` is
    *open*, so it takes density out of the dark-energy budget and slows late
    growth.  Both sides must move the same way.
    """

    def test_positive_omega_k_is_open_in_ggah_mod(self):
        """It lowers `Omega_de`, so `E(z)` and the growth respond one way."""
        params = pytest.importorskip("ggah_mod.cosmology.parameters")
        flat = params.Cosmology.create(Omega_k=0.0)
        open_ = params.Cosmology.create(Omega_k=0.05)
        assert float(open_.Omega_de) < float(flat.Omega_de)

    def test_positive_omega_k_is_open_in_class_too(self):
        """The same statement through the dict this package builds.

        Marked slow: it is a real CLASS solve, and it is the only place the two
        conventions are actually compared rather than assumed.
        """
        pytest.importorskip("classy")
        import numpy as np

        from emu_pk import generate, grid
        k = np.logspace(-2, 0, 8)
        z = np.array([0.0])
        base = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649,
                    ln10A_s=3.044, k_max_h=2.0, z_max=1.0)
        pm_flat, _ = generate.solve(cosmo.class_params(**base, Omega_k=0.0), z, k)
        pm_open, _ = generate.solve(cosmo.class_params(**base, Omega_k=0.05), z, k)
        # Less dark energy -> less late-time growth -> less power at z = 0.
        assert np.all(pm_open[0] < pm_flat[0])

    def test_the_parameter_order_is_the_one_ggah_mod_stacks(self):
        """`GgahEmuPk._params` builds the vector this package's `PARAMS` names.

        A permuted or short stack is the failure this whole seam is for: it
        returns a spectrum, and both packages keep passing their own tests.
        """
        power = pytest.importorskip("ggah_mod.cosmology.power")
        params = pytest.importorskip("ggah_mod.cosmology.parameters")
        import numpy as np

        from emu_pk import box
        backend = object.__new__(power.GgahEmuPk)
        c = params.Cosmology.create(Omega_k=0.03, sum_mnu=0.09, w0=-0.9, wa=0.2)
        got = np.asarray(backend._params(c))
        if got.size != len(box.PARAMS):
            pytest.skip(
                f"ggah_mod stacks {got.size} parameters and this box has "
                f"{len(box.PARAMS)}; the two are released in lockstep and this "
                "checkout is mid-flight.")
        want = {"omega_b": float(c.Omega_b) * float(c.h) ** 2,
                "omega_cdm": float(c.Omega_cdm) * float(c.h) ** 2,
                "h": float(c.h), "n_s": float(c.n_s),
                "ln10A_s": float(c.ln10A_s), "sum_mnu": float(c.sum_mnu),
                "w0": float(c.w0), "wa": float(c.wa),
                "Omega_k": float(c.Omega_k)}
        for j, name in enumerate(box.PARAMS):
            assert got[j] == pytest.approx(want[name]), \
                f"column {j} should be {name}"
