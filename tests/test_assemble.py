"""Shards in, one training set out -- and the ways that can go wrong quietly.

`assemble` is the step that would corrupt a three-thousand-shard reassembly
without saying so: a permuted row, a dropped part, two sweeps merged.  None
of that needs CLASS to test, and all of it needs testing, because the failure
mode is a training set that trains perfectly well and is wrong.
"""
import numpy as np
import pytest

from emu_pk import assemble, box, grid


def _shard(path, idx, z, lnk, seed=0, failed=()):
    """One `emu_*.npz` as `generate.emu_shard` writes it."""
    rng = np.random.default_rng(seed)
    n = len(idx)
    theta = box.sample(max(idx) + 1, seed=20260827)[list(idx)]
    # A spectrum that encodes its own identity, so a permuted row is visible.
    pm = np.array([[[float(i) + 0.001 * j + 1e-6 * m for m in range(len(lnk))]
                    for j in range(len(z))] for i in idx], dtype=np.float32)
    np.savez_compressed(
        path, idx=np.array(idx, dtype=np.int64), theta=theta, z=z, lnk=lnk,
        pm=np.exp(pm / 1e3), pcb=np.exp(pm / 1e3) * 0.99,
        failed_idx=np.array(failed, dtype=np.int64),
        failed_why=np.array(["x"] * len(failed), dtype="U200"))
    return path


@pytest.fixture
def shards(tmp_path):
    z, lnk = grid.Z_NODES_EMU[:4], grid.lnk_grid(6)
    _shard(tmp_path / "emu_00000_0000000.npz", [0, 1, 2], z, lnk, seed=0)
    _shard(tmp_path / "emu_00000_0000003.npz", [3, 4], z, lnk, seed=1,
           failed=[5])
    return tmp_path, z, lnk


class TestTheRoundTrip:
    def test_what_goes_in_comes_back(self, shards):
        d, z, lnk = shards
        assemble.build_training_set(d, d / "ds.npz", parts=2, workers=2)
        X, Ym, Ycb, out_lnk = assemble.load_training_set(d / "ds.npz", workers=2)
        assert len(X) == 5 * len(z)
        assert X.shape[1] == len(box.PARAMS) + 1
        assert Ym.shape == (5 * len(z), len(lnk))
        assert np.array_equal(out_lnk, lnk)
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(Ym))

    def test_rows_are_cosmology_major_and_not_shuffled(self, shards):
        """Row order is the concatenation of parts in index order, so the
        design indices still line up.  A silently permuted row is the error
        that trains perfectly well."""
        d, z, lnk = shards
        assemble.build_training_set(d, d / "ds.npz", parts=2, workers=2)
        X, Ym, _, _ = assemble.load_training_set(d / "ds.npz", workers=2)
        # Each block of len(z) rows shares one theta and steps through z.
        for c in range(5):
            blk = X[c * len(z):(c + 1) * len(z)]
            assert np.allclose(blk[:, :-1], blk[0, :-1]), "theta varies within a cosmology"
            assert np.allclose(blk[:, -1], z), "z column is not the grid, in order"

    def test_the_redshift_column_is_last(self, shards):
        """`train.COLS` reads it as last; if `assemble` ever put it elsewhere,
        the reduced target would divide the wrong column out.

        Compared with a tolerance, not by equality: the design matrix is stored
        as float32, so the redshift the network sees carries float32 precision
        (about 1e-7 relative).  That is far below anything here and worth
        knowing rather than tripping over.
        """
        from emu_pk import train as T

        d, z, _ = shards
        assemble.build_training_set(d, d / "ds.npz", parts=1, workers=1)
        X, *_ = assemble.load_training_set(d / "ds.npz", workers=1)
        assert T.COLS[-1] == "z"
        assert X.dtype == np.float32
        got = np.unique(X[:, -1])
        assert len(got) == len(z)
        assert np.allclose(np.sort(got), np.sort(z), rtol=1e-6)


class TestItRefusesWhatItCannotMerge:
    def test_shards_on_different_k_grids_are_rejected(self, tmp_path):
        """Two sweeps cannot be merged, and the failure has to be loud."""
        z = grid.Z_NODES_EMU[:3]
        _shard(tmp_path / "emu_00000_0000000.npz", [0, 1], z, grid.lnk_grid(6))
        _shard(tmp_path / "emu_00000_0000002.npz", [2, 3], z, grid.lnk_grid(8))
        with pytest.raises(ValueError, match="different grid"):
            assemble.build_training_set(tmp_path, tmp_path / "ds.npz", parts=1)

    def test_an_empty_directory_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no emu_"):
            assemble.build_training_set(tmp_path, tmp_path / "ds.npz")

    def test_a_missing_part_is_caught_not_silently_short(self, shards):
        """A training set that is quietly missing a slab trains fine."""
        d, _, _ = shards
        assemble.build_training_set(d, d / "ds.npz", parts=2, workers=2)
        (d / "ds.part001.npz").unlink()
        with pytest.raises(Exception):
            assemble.load_training_set(d / "ds.npz", workers=1)

    def test_a_truncated_part_is_caught(self, shards):
        """The manifest records the row count precisely so this is an error
        rather than a shorter training set."""
        d, z, lnk = shards
        assemble.build_training_set(d, d / "ds.npz", parts=2, workers=2)
        with np.load(d / "ds.part000.npz") as p:
            half = {k: p[k][: max(1, len(p[k]) // 2)] for k in p.files}
        np.savez(d / "ds.part000.npz", **half)
        with pytest.raises(ValueError, match="rows"):
            assemble.load_training_set(d / "ds.npz", workers=1)


class TestGapsAreReportedNotFilled:
    def test_class_failures_are_recorded_in_the_manifest(self, shards):
        """A set with silent gaps is wrong exactly where CLASS refused, which
        is the part of the box a forecast is most likely to wander into."""
        d, _, _ = shards
        assemble.build_training_set(d, d / "ds.npz", parts=1, workers=1)
        with np.load(d / "ds.npz") as m:
            assert 5 in m["failed_idx"].tolist()
            assert sorted(m["idx"].tolist()) == [0, 1, 2, 3, 4]

    def test_parts_never_outnumber_rows(self, tmp_path):
        """A smoke run with six rows should not write thirty-two files, thirty
        of them empty."""
        z, lnk = grid.Z_NODES_EMU[:2], grid.lnk_grid(4)
        _shard(tmp_path / "emu_00000_0000000.npz", [0], z, lnk)
        assemble.build_training_set(tmp_path, tmp_path / "ds.npz", parts=32)
        with np.load(tmp_path / "ds.npz") as m:
            names = [str(n) for n in m["parts"]]
        assert len(names) <= 2, names
        assert all((tmp_path / n).exists() for n in names)


class TestTheCorrectionTable:
    """`build_ratio` measures the factorisation it does not use.

    The number it prints and stores as `resid_max` is what justifies shipping
    the full four-axis cube instead of two cheap factors, so it has to be the
    real cross term and not a placeholder.
    """

    @staticmethod
    def _ratio_shards(tmp_path, cross=0.0, n_z=2, n_k=3):
        """Every node of the correction grid, with a controllable cross term.

        `ln r = a(mnu) + b(w0, wa) + cross * a * b`, so at `cross = 0` the
        factorisation is exact and `resid_max` must be ~0.
        """
        from emu_pk import cosmo, grid

        z = grid.Z_NODES_RATIO[:n_z]
        lnk = grid.lnk_grid(n_k)
        fid = cosmo.PLANCK18
        rows, pm, pcb = [], [], []
        for mnu in grid.MNU_NODES:
            for w0 in grid.W0_NODES:
                for wa in grid.WA_NODES:
                    a = -0.5 * float(mnu)
                    b = 0.3 * (float(w0) + 1.0) + 0.2 * float(wa)
                    lnr = a + b + cross * a * b
                    block = np.full((n_z, n_k), np.exp(lnr))
                    rows.append((mnu, w0, wa,
                                 cosmo.f_nu(float(mnu), fid["h"], fid["Omega_m"])))
                    pm.append(block)
                    pcb.append(block * 1.0)
        np.savez_compressed(
            tmp_path / "ratio_00000.npz", theta=np.array(rows), z=z, lnk=lnk,
            pm=np.array(pm), pcb=np.array(pcb),
            h_fid=fid["h"], Omega_m_fid=fid["Omega_m"])
        return tmp_path

    def test_an_exactly_factorisable_grid_reports_no_cross_term(self, tmp_path):
        d = self._ratio_shards(tmp_path, cross=0.0)
        out = assemble.build_ratio(d, tmp_path / "tab.npz", verbose=False)
        with np.load(out) as t:
            assert float(t["resid_max"]) < 1e-9

    def test_a_cross_term_is_measured_not_assumed_away(self, tmp_path):
        d = self._ratio_shards(tmp_path, cross=0.5)
        out = assemble.build_ratio(d, tmp_path / "tab.npz", verbose=False)
        with np.load(out) as t:
            assert float(t["resid_max"]) > 1e-3

    def test_the_table_is_exactly_one_at_the_lambdacdm_massless_corner(
            self, tmp_path):
        """The stored log-ratio is exactly zero there, which is what lets the
        correction be applied with no Python branch on a traced value."""
        from emu_pk import grid

        d = self._ratio_shards(tmp_path, cross=0.3)
        out = assemble.build_ratio(d, tmp_path / "tab.npz", verbose=False)
        with np.load(out) as t:
            i = int(np.argmin(np.abs(t["mnu"])))
            a = int(np.argmin(np.abs(t["w0"] + 1.0)))
            b = int(np.argmin(np.abs(t["wa"])))
            assert np.allclose(t["lnr_m"][i, a, b], 0.0, atol=1e-12)

    def test_a_partial_sweep_is_refused(self, tmp_path):
        """A table cannot be assembled from a grid with holes, and the failure
        has to name what is missing rather than interpolating over it."""
        from emu_pk import cosmo, grid

        z, lnk = grid.Z_NODES_RATIO[:2], grid.lnk_grid(3)
        fid = cosmo.PLANCK18
        np.savez_compressed(
            tmp_path / "ratio_00000.npz",
            theta=np.array([(0.0, -1.0, 0.0, 0.0)]), z=z, lnk=lnk,
            pm=np.ones((1, 2, 3)), pcb=np.ones((1, 2, 3)),
            h_fid=fid["h"], Omega_m_fid=fid["Omega_m"])
        with pytest.raises(KeyError, match="missing"):
            assemble.build_ratio(tmp_path, tmp_path / "tab.npz", verbose=False)

    def test_no_shards_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no ratio_"):
            assemble.build_ratio(tmp_path, tmp_path / "tab.npz", verbose=False)
