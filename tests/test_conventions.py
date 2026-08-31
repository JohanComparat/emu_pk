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


def test_class_params_sets_flatness_and_linearity_explicitly():
    """Both are assumptions ggah_mod makes; stated here rather than defaulted.

    ggah_mod configures CAMB with `omk=0.0` but left CLASS on its default --
    the same assumption, written down in one place and implied in the other.
    """
    p = cosmo.class_params(h=0.6736, omega_b=0.0224, omega_cdm=0.12,
                           n_s=0.9649, ln10A_s=3.044)
    assert p["Omega_k"] == 0.0
    assert p["non linear"] == "none"


def test_cpl_uses_ppf():
    """w(a) crosses -1 inside the box, and the fluid is singular there without PPF."""
    p = cosmo.class_params(h=0.7, omega_b=0.0224, omega_cdm=0.12, n_s=0.96,
                           ln10A_s=3.0, w0=-0.9, wa=0.3)
    assert p["use_ppf"] == "yes"
    assert p["w0_fld"] == -0.9 and p["wa_fld"] == 0.3
