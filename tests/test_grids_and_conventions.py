"""Invariants of the grids and the density conventions.

Cheap, fast, and worth having: these are constants that other modules index
into, and a change to one of them is a change to every shipped table and every
trained network at once.  `test_conventions.py` already checks them against
`ggah_mod` when that is importable; this checks them against themselves, so
the suite still says something in an environment that has neither `ggah_mod`
nor `classy`.
"""
import numpy as np
import pytest

from emu_pk import box, cosmo, grid


class TestTheWavenumberGrid:
    def test_it_spans_what_it_says(self):
        k = grid.k_grid()
        assert len(k) == grid.N_K
        assert k[0] == pytest.approx(grid.K_MIN)
        assert k[-1] == pytest.approx(grid.K_MAX)

    def test_it_is_log_spaced_and_increasing(self):
        lnk = grid.lnk_grid()
        d = np.diff(lnk)
        assert np.all(d > 0)
        assert np.allclose(d, d[0]), "not uniform in ln k"

    def test_lnk_grid_is_the_log_of_k_grid(self):
        assert np.allclose(grid.lnk_grid(), np.log(grid.k_grid()))

    def test_it_reaches_what_its_consumer_integrates(self):
        """`ggah_mod` quadratures sigma(M) to 200 h/Mpc.  A network that stops
        short leaves `jnp.interp` clamping above its last mode, returning a flat
        P(k) where it should be falling.  The reach is what prevents that."""
        assert grid.K_MAX >= 200.0


class TestTheRedshiftGrids:
    def test_the_training_nodes_are_increasing_and_start_at_zero(self):
        z = grid.Z_NODES_EMU
        assert np.all(np.diff(z) > 0)
        assert z[0] == 0.0, "z=0 must be a node: sigma_8 is quoted there"
        assert z[-1] == pytest.approx(grid.Z_MAX)

    def test_the_training_nodes_resolve_low_redshift(self):
        """Twenty uniform nodes put the first interior one at z = 0.263, which
        leaves the slope at z=0 unconstrained on one side.  See
        `grid.Z_NODES_EMU`."""
        z = grid.Z_NODES_EMU
        assert z[1] < 0.02, f"first gap is {z[1]:.4f}; the z=0 slope needs better"
        assert (z < 1.0).sum() >= 10, "too few nodes below z=1"

    def test_the_ratio_table_nodes_are_increasing(self):
        assert np.all(np.diff(grid.Z_NODES_RATIO) > 0)
        assert grid.Z_NODES_RATIO[0] == 0.0

    def test_the_round_numbers_a_user_evaluates_at_are_nodes(self):
        """A node is where the ratio table's interpolation is exact."""
        for z in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            assert np.any(np.isclose(grid.Z_NODES_RATIO, z)), z

    def test_every_correction_axis_is_strictly_increasing(self):
        """Hermite slopes are meaningless on a non-monotonic axis."""
        for name in ("MNU_NODES", "W0_NODES", "WA_NODES"):
            ax = getattr(grid, name)
            assert np.all(np.diff(ax) > 0), name

    def test_the_lambdacdm_massless_corner_is_on_the_grid(self):
        """It is where both ratios are exactly 1, which is what lets the
        correction be applied with no branch on a traced value."""
        assert grid.MNU_NODES[0] == 0.0
        assert np.any(np.isclose(grid.W0_NODES, -1.0))
        assert np.any(np.isclose(grid.WA_NODES, 0.0))


class TestTheDensityConventions:
    def test_omega_nu_follows_the_93_14_convention(self):
        h, mnu = 0.6736, 0.06
        assert cosmo.omega_nu(mnu, h) == pytest.approx(mnu / (93.14 * h * h))

    def test_omega_nu_is_zero_for_massless_neutrinos(self):
        assert cosmo.omega_nu(0.0, 0.7) == 0.0

    def test_f_nu_is_the_fraction_of_omega_m(self):
        h, mnu, om = 0.6736, 0.1, 0.31
        assert cosmo.f_nu(mnu, h, om) == pytest.approx(
            cosmo.omega_nu(mnu, h) / om)

    def test_the_neutrino_split_matches_class_to_camb(self):
        """`N_ur = N_eff - 3 * 1.0132` is the split that lines CLASS up with
        CAMB; giving the single non-cold species the full degeneracy roughly
        doubles their disagreement."""
        p = cosmo.class_params(h=0.6736, omega_b=0.02237, omega_cdm=0.12,
                               n_s=0.9649, ln10A_s=3.044, sum_mnu=0.06)
        assert p["N_ncdm"] == 1
        assert p["deg_ncdm"] == pytest.approx(3.0)
        assert p["m_ncdm"] == pytest.approx(0.02)
        assert p["N_ur"] == pytest.approx(cosmo.N_EFF - 3 * 1.0132)

    def test_massless_uses_the_full_effective_number(self):
        p = cosmo.class_params(h=0.6736, omega_b=0.02237, omega_cdm=0.12,
                               n_s=0.9649, ln10A_s=3.044, sum_mnu=0.0)
        assert "N_ncdm" not in p
        assert p["N_ur"] == pytest.approx(cosmo.N_EFF)

    def test_the_pivot_is_stated_not_defaulted(self):
        """It appears in the inference path now, so it cannot be a default of
        whatever CLASS happens to ship."""
        p = cosmo.class_params(h=0.7, omega_b=0.022, omega_cdm=0.12,
                               n_s=0.96, ln10A_s=3.0)
        assert p["k_pivot"] == pytest.approx(cosmo.K_PIVOT)

    def test_flatness_and_linearity_are_explicit(self):
        p = cosmo.class_params(h=0.7, omega_b=0.022, omega_cdm=0.12,
                               n_s=0.96, ln10A_s=3.0)
        assert p["Omega_k"] == 0.0
        assert p["non linear"] == "none"

    def test_cpl_switches_on_ppf(self):
        """The box contains w(a) crossing -1, where the fluid
        parameterisation is singular without PPF."""
        p = cosmo.class_params(h=0.7, omega_b=0.022, omega_cdm=0.12,
                               n_s=0.96, ln10A_s=3.0, w0=-0.9, wa=0.2)
        assert p["use_ppf"] == "yes"
        assert p["Omega_Lambda"] == 0.0
        lcdm = cosmo.class_params(h=0.7, omega_b=0.022, omega_cdm=0.12,
                                  n_s=0.96, ln10A_s=3.0)
        assert "w0_fld" not in lcdm, "LambdaCDM should not go through the fluid"


class TestTheBoxAndTheGridsAgree:
    def test_the_emulator_redshift_range_is_the_declared_one(self):
        assert grid.Z_NODES_EMU.min() == grid.Z_MIN
        assert grid.Z_NODES_EMU.max() == pytest.approx(grid.Z_MAX)

    def test_the_design_stays_inside_the_box(self):
        d = box.sample(64, seed=7)
        for row in d:
            assert not box.inside(row), box.inside(row)

    def test_the_design_rejects_early_dark_energy_domination(self):
        d = box.sample(256, seed=11)
        w0 = d[:, box.PARAMS.index("w0")]
        wa = d[:, box.PARAMS.index("wa")]
        assert np.all(w0 + wa < 0.0)
