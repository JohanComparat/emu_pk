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
        CAMB.  It removes what three species would have contributed had they
        stayed relativistic, and there are still three of them whatever their
        masses -- so this number does not move when the masses stop being
        equal."""
        p = cosmo.class_params(h=0.6736, omega_b=0.02237, omega_cdm=0.12,
                               n_s=0.9649, ln10A_s=3.044, sum_mnu=0.06)
        assert p["N_ncdm"] == 3
        assert p["N_ur"] == pytest.approx(cosmo.N_EFF - 3 * 1.0132)
        # Three *separate* species now, not one carrying a degeneracy of three.
        assert "deg_ncdm" not in p
        assert [float(x) for x in p["m_ncdm"].split(",")] == pytest.approx(
            [0.02, 0.02, 0.02])

    def test_n_ncdm_is_an_int_so_the_p_cb_guard_still_works(self):
        """`generate.solve` decides whether to ask CLASS for `pk_cb_lin` by the
        truthiness of `N_ncdm`, and the string "0" is truthy.  An int here is
        what keeps the massless branch reachable."""
        p = cosmo.class_params(h=0.6736, omega_b=0.02237, omega_cdm=0.12,
                               n_s=0.9649, ln10A_s=3.044, sum_mnu=0.06)
        assert isinstance(p["N_ncdm"], int)

    def test_the_default_ratios_are_the_degenerate_convention(self):
        """A caller that names no ratios gets the degenerate convention, so
        the three-species path is the only path and cannot rot."""
        base = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.12,
                    n_s=0.9649, ln10A_s=3.044, sum_mnu=0.12)
        m = [float(x) for x in cosmo.class_params(**base)["m_ncdm"].split(",")]
        assert m == pytest.approx([0.04, 0.04, 0.04])

    @pytest.mark.parametrize("r1,r2,want", [
        (0.2196, 0.2357, "normal"),      # the NH locus at sum = 0.101 eV
        (0.0149, 0.4888, "inverted"),    # the IH locus at the same sum
    ])
    def test_the_ratios_reach_class_and_sum_to_the_mass(self, r1, r2, want):
        """The masses are the ratios times the sum, in CLASS's own comma form,
        and they add back up to what the caller asked for."""
        S = 0.101
        p = cosmo.class_params(h=0.6736, omega_b=0.02237, omega_cdm=0.12,
                               n_s=0.9649, ln10A_s=3.044, sum_mnu=S,
                               nu_r1=r1, nu_r2=r2)
        m = [float(x) for x in p["m_ncdm"].split(",")]
        assert len(m) == 3
        assert sum(m) == pytest.approx(S)
        assert m[0] == pytest.approx(r1 * S)
        assert m[1] == pytest.approx(r2 * S)

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

    def test_curvature_and_linearity_are_explicit(self):
        """`Omega_k` is stated rather than defaulted, and it is *sampled*.

        A default here is a cosmology nobody chose, which is reason enough to
        write it down whether or not it is a parameter.  Being one adds the
        second requirement: the default is only the default, and the value has
        to travel.
        """
        p = cosmo.class_params(h=0.7, omega_b=0.022, omega_cdm=0.12,
                               n_s=0.96, ln10A_s=3.0)
        assert p["Omega_k"] == 0.0
        assert p["non linear"] == "none"

    @pytest.mark.parametrize("ok", [-0.15, -0.02, 0.02, 0.15])
    def test_curvature_reaches_class(self, ok):
        """The half that matters.  Stating a constant zero and passing the
        parameter through look identical from the flat call alone."""
        p = cosmo.class_params(h=0.7, omega_b=0.022, omega_cdm=0.12,
                               n_s=0.96, ln10A_s=3.0, Omega_k=ok)
        assert p["Omega_k"] == pytest.approx(ok)

    def test_the_box_bounds_are_the_ones_class_survives(self):
        """Measured, not chosen.  CLASS solves the whole box at |Omega_k| <=
        0.15; on the closed side it starts refusing near -0.275 at the
        low-density corner.  If the bound ever widens, that measurement is
        what has to be redone."""
        lo, hi = box.BOX["Omega_k"]
        assert (lo, hi) == (-0.15, 0.15)
        assert lo < 0.0 < hi, "flat has to stay inside the box"

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
