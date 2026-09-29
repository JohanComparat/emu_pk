"""The 2.1.0 training truth: CAMB at ggah_mod's precision, times CLASS's heating.

Three things have to hold for a training set built this way to mean what the
CHANGELOG says it means, and each is tested here rather than trusted:

* the CAMB input is ggah_mod 0.9.8's ``camb_input`` for the same point, to the
  last bit -- the neutrinos especially, where a mass share read as a density
  share put a state 20 per cent light;
* a shard says which truth wrote it, and ``assemble`` refuses a directory that
  holds two;
* the heating ratio is a small-scale correction and nothing else.
"""
import json

import numpy as np
import pytest

from emu_pk import assemble, box, cosmo, generate, grid, heating

FID = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.12, n_s=0.9649,
           ln10A_s=3.044, sum_mnu=0.06, w0=-1.0, wa=0.0, Omega_k=0.0,
           nu_r1=1.0 / 3.0, nu_r2=1.0 / 3.0)


class TestTheCambInputIsGgahMods:
    @pytest.mark.parametrize("hierarchy", ["normal", "inverted", "degenerate"])
    def test_same_neutrinos_as_ggah_mod(self, hierarchy):
        pytest.importorskip("camb")
        power = pytest.importorskip("ggah_mod.cosmology.power")
        params = pytest.importorskip("ggah_mod.cosmology.parameters")
        if not hasattr(power, "camb_input"):
            pytest.skip("ggah_mod predates camb_input (0.9.8)")
        c = params.Cosmology.create(sum_mnu=0.1, nu_hierarchy=hierarchy)
        h = float(c.h)
        m = np.asarray(c.nu_masses, dtype=float)
        r = m / m.sum()
        theirs = power.camb_input(c, redshifts=[0.0])
        ours = cosmo.camb_params(h=h, omega_b=float(c.Omega_b) * h * h,
                                 omega_cdm=float(c.Omega_cdm) * h * h,
                                 n_s=float(c.n_s), ln10A_s=float(c.ln10A_s),
                                 sum_mnu=0.1, nu_r1=r[0], nu_r2=r[1],
                                 redshifts=[0.0])
        for field in ("omnuh2", "num_nu_massless", "ombh2", "omch2", "H0"):
            assert getattr(ours, field) == pytest.approx(getattr(theirs, field),
                                                         rel=1e-12), field
        np.testing.assert_allclose(ours.nu_mass_fractions,
                                   theirs.nu_mass_fractions, rtol=1e-12)
        np.testing.assert_allclose(ours.nu_mass_degeneracies,
                                   theirs.nu_mass_degeneracies, rtol=1e-12)

    def test_the_precision_is_ggah_mods(self):
        power = pytest.importorskip("ggah_mod.cosmology.power")
        if not hasattr(power, "CAMB_PRECISION"):
            pytest.skip("ggah_mod predates CAMB_PRECISION (0.9.8)")
        assert cosmo.CAMB_PRECISION == power.CAMB_PRECISION

    def test_constants_are_ggah_mods(self):
        const = pytest.importorskip("ggah_mod.cosmology.constants")
        if not hasattr(const, "N_MASSIVE_EFF"):
            pytest.skip("ggah_mod predates the 0.9.8 neutrino convention")
        assert cosmo.N_MASSIVE_EFF == const.N_MASSIVE_EFF
        assert cosmo.N_UR_REMAINDER == const.N_UR_REMAINDER
        assert cosmo.OMEGA_GAMMA_H2 == const.OMEGA_GAMMA_H2

    def test_the_energy_factor_has_both_limits(self):
        assert cosmo.nu_energy_factor(0.0) == 1.0
        kappa = 180 * 1.2020569031595942 / (7 * np.pi ** 4)
        y = 400.0
        assert cosmo.nu_energy_factor(y) / (kappa * y) == pytest.approx(1.0, rel=1e-4)

    def test_an_unknown_precision_key_raises(self):
        pytest.importorskip("camb")
        with pytest.raises(KeyError):
            cosmo.camb_params(**FID, precision={"NotAKnob": 1.0})


def _shard(path, idx, z, lnk, truth=None):
    theta = box.sample(max(idx) + 1, seed=20260827)[list(idx)]
    pm = np.exp(np.ones((len(idx), len(z), len(lnk))))
    np.savez_compressed(path, idx=np.array(idx, dtype=np.int64), theta=theta,
                        z=z, lnk=lnk, pm=pm, pcb=pm,
                        failed_idx=np.zeros(0, dtype=np.int64),
                        failed_why=np.zeros(0, dtype="U200"),
                        params=np.array(box.PARAMS, dtype="U16"),
                        **(truth or {}))


class TestAShardSaysWhichTruthWroteIt:
    def test_the_stamp_names_the_solver_and_its_settings(self):
        s = generate.stamp()
        assert str(s["solver"]) == generate.SOLVER == "camb"
        assert json.loads(str(s["precision"])) == cosmo.CAMB_PRECISION
        assert json.loads(str(s["heating"])) == heating.SETTINGS

    def test_a_mixed_directory_is_refused(self, tmp_path):
        z, lnk = grid.Z_NODES_EMU[:3], grid.lnk_grid(5)
        _shard(tmp_path / "emu_00000_0000000.npz", [0, 1], z, lnk)  # 2.0: no stamp
        _shard(tmp_path / "emu_00000_0000002.npz", [2, 3], z, lnk,
               truth=generate.stamp("camb"))
        with pytest.raises(ValueError, match="solver/precision/heating"):
            assemble.build_training_set(tmp_path, tmp_path / "ds.npz", parts=1)

    def test_the_stamp_reaches_the_dataset(self, tmp_path):
        z, lnk = grid.Z_NODES_EMU[:3], grid.lnk_grid(5)
        for i, lo in enumerate((0, 2)):
            _shard(tmp_path / f"emu_00000_000000{lo}.npz", [lo, lo + 1], z, lnk,
                   truth=generate.stamp("camb"))
        out = assemble.build_training_set(tmp_path, tmp_path / "ds.npz", parts=1)
        t = assemble.dataset_truth(out)
        assert t["solver"] == "camb"
        assert json.loads(t["precision"]) == cosmo.CAMB_PRECISION

    def test_an_unstamped_dataset_reads_as_2_0(self, tmp_path):
        z, lnk = grid.Z_NODES_EMU[:3], grid.lnk_grid(5)
        _shard(tmp_path / "emu_00000_0000000.npz", [0, 1], z, lnk)
        out = assemble.build_training_set(tmp_path, tmp_path / "ds.npz", parts=1)
        assert assemble.dataset_truth(out) == {"solver": "class",
                                               "precision": "{}",
                                               "heating": "null"}


@pytest.mark.slow
class TestTheHeatingIsASmallScaleCorrection:
    def test_unity_on_large_scales_suppression_on_small(self):
        pytest.importorskip("classy")
        k = np.array([1e-3, 1e-2, 0.1, 50.0, 200.0])
        z = np.array([0.0, 5.0])
        heat = heating.pair(FID, z, k)
        np.testing.assert_allclose(heat.r_m[:, :3], 1.0, atol=1e-4)
        # Suppressed today at k = 200 by a few per cent, and not at z = 5,
        # before reionization finished heating anything.
        assert -0.06 < heat.r_m[0, -1] - 1.0 < -0.01
        assert abs(heat.r_m[1, -1] - 1.0) < 2e-3
