"""The shard writer, without running CLASS.

`generate.solve` needs classy and minutes; everything *around* it -- the chunk
naming, the atomic rename, the skip-if-exists resume -- needs neither, and is
where the failures that cost a production run actually live.
"""
import numpy as np
import pytest

from emu_pk import assemble, box, generate, grid


@pytest.fixture
def fake_solve(monkeypatch):
    """Replace CLASS with something instant and deterministic."""
    def solve(params, z_nodes, k_h):
        n_z, n_k = len(np.atleast_1d(z_nodes)), len(k_h)
        pm = np.full((n_z, n_k), params["h"], dtype=float)
        return pm, pm * 0.99
    monkeypatch.setattr(generate, "solve", solve)
    return solve


def test_chunk_files_are_named_and_readable(tmp_path, fake_solve):
    """The name says which cosmologies are inside without opening the file."""
    generate.emu_shard(0, 20, tmp_path, n_total=100, chunk=10)
    names = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert names == ["emu_00000_0000000.npz", "emu_00000_0000010.npz"]
    # And no leftovers: a `.part` that never got renamed is the failure mode
    # that reached the cluster once already.
    assert list(tmp_path.glob("*.part*")) == []
    with np.load(tmp_path / names[0]) as d:
        assert d["pm"].shape == (10, len(grid.Z_NODES_EMU), grid.N_K)
        assert d["theta"].shape == (10, len(box.PARAMS))
        assert list(d["idx"]) == list(range(10))


def test_a_landed_chunk_is_skipped(tmp_path, fake_solve, capsys):
    """What makes a besteffort restart resume rather than redo."""
    generate.emu_shard(0, 10, tmp_path, n_total=100, chunk=10)
    before = (tmp_path / "emu_00000_0000000.npz").stat().st_mtime_ns
    capsys.readouterr()
    generate.emu_shard(0, 10, tmp_path, n_total=100, chunk=10)
    assert "skipped" in capsys.readouterr().out
    assert (tmp_path / "emu_00000_0000000.npz").stat().st_mtime_ns == before


def test_a_shard_past_the_end_of_the_design_does_nothing(tmp_path, fake_solve):
    assert generate.emu_shard(99, 500, tmp_path, n_total=100) == []
    assert list(tmp_path.glob("*.npz")) == []


def test_a_failing_solve_is_recorded_not_fatal(tmp_path, monkeypatch):
    """CLASS refuses some corners of any wide box.  A run that dies on the
    first one reports nothing; the index and the reason have to survive."""
    calls = {"n": 0}

    def flaky(params, z_nodes, k_h):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("CLASS said no")
        n_z, n_k = len(np.atleast_1d(z_nodes)), len(k_h)
        pm = np.ones((n_z, n_k))
        return pm, pm
    monkeypatch.setattr(generate, "solve", flaky)

    generate.emu_shard(0, 5, tmp_path, n_total=100, chunk=5)
    with np.load(tmp_path / "emu_00000_0000000.npz") as d:
        assert len(d["idx"]) == 4
        assert list(d["failed_idx"]) == [2]
        assert "CLASS said no" in str(d["failed_why"][0])


class TestTheTrainingSetIsAssembledInOrder:
    """`build_training_set` reads its shards in parallel; order is not.

    A 3.0 MB shard reads in 3.8 s on Dahu -- 0.8 MB/s against /bettik, three
    hours for a 3000-shard sweep if the reads are serialised.  Threads recover
    that, because the cost is filesystem latency rather than zlib.  But `as_completed` returns them in whatever order they finish, so the
    thing to test is that the rows still line up with the design indices: a
    training set silently permuted against its own `idx` would train perfectly
    well and be wrong everywhere.
    """

    @staticmethod
    def _shard(path, i0, n, n_z, n_k, seed):
        rng = np.random.default_rng(seed)
        theta = rng.random((n, 8)).astype(np.float32) + i0
        pm = np.exp(rng.random((n, n_z, n_k)).astype(np.float32))
        np.savez_compressed(
            path, theta=theta, z=np.linspace(0.0, 3.0, n_z),
            lnk=np.linspace(-9.0, 5.0, n_k),
            idx=np.arange(i0, i0 + n), failed_idx=np.array([], dtype=np.int64),
            pm=pm, pcb=pm * 0.9)
        return theta, pm

    def test_rows_follow_the_shard_order_not_the_completion_order(self, tmp_path):
        n, n_z, n_k = 4, 3, 5
        want_theta, want_pm = [], []
        for s in range(6):
            th, pm = self._shard(tmp_path / f"emu_{s:05d}.npz", s * n, n, n_z, n_k, s)
            want_theta.append(th)
            want_pm.append(pm)
        out = tmp_path / "training_set.npz"
        assemble.build_training_set(tmp_path, out, workers=4, parts=5)

        # Read it back the way training does -- through `load_training_set`,
        # which reassembles the parts.  Reading the manifest directly would
        # test the writer against itself and miss a part-ordering error, which
        # is the only interesting way this can fail.
        X, ln_pm, _, _ = assemble.load_training_set(out, workers=3)
        with np.load(out) as d:
            idx = d["idx"]
        assert X.shape == (6 * n * n_z, 9)          # 8 parameters plus z
        np.testing.assert_array_equal(idx, np.arange(6 * n))
        # Row (shard s, cosmology c, redshift j) must carry shard s's theta_c
        # and shard s's spectrum -- which is exactly what a permutation breaks.
        for s in range(6):
            for c in range(n):
                r = (s * n + c) * n_z
                np.testing.assert_allclose(
                    X[r:r + n_z, :8],
                    np.broadcast_to(want_theta[s][c], (n_z, 8)), rtol=1e-6)
                np.testing.assert_allclose(np.exp(ln_pm[r:r + n_z]),
                                           want_pm[s][c], rtol=1e-5)

    def test_a_missing_part_is_refused_rather_than_silently_short(self, tmp_path):
        """The row count is recorded in the manifest and checked on load.

        A part that failed to write, or was cleaned up, would otherwise give a
        training set that is simply shorter than it should be -- which trains
        perfectly well.
        """
        for s in range(3):
            self._shard(tmp_path / f"emu_{s:05d}.npz", s * 2, 2, 3, 5, s)
        out = tmp_path / "o.npz"
        assemble.build_training_set(tmp_path, out, workers=2, parts=3)
        (tmp_path / "o.part001.npz").unlink()
        with pytest.raises(Exception):
            assemble.load_training_set(out, workers=2)

    def test_a_shard_on_a_different_grid_is_refused(self, tmp_path):
        self._shard(tmp_path / "emu_00000.npz", 0, 2, 3, 5, 0)
        self._shard(tmp_path / "emu_00001.npz", 2, 2, 4, 5, 1)   # n_z differs
        with pytest.raises(ValueError, match="different grid"):
            assemble.build_training_set(tmp_path, tmp_path / "o.npz", workers=2)

    def test_a_failed_shard_contributes_no_rows_but_is_recorded(self, tmp_path):
        self._shard(tmp_path / "emu_00000.npz", 0, 2, 3, 5, 0)
        np.savez_compressed(
            tmp_path / "emu_00001.npz",
            theta=np.zeros((0, 8), dtype=np.float32),
            z=np.linspace(0.0, 3.0, 3), lnk=np.linspace(-9.0, 5.0, 5),
            idx=np.array([], dtype=np.int64), failed_idx=np.array([2, 3]),
            pm=np.zeros((0, 3, 5), dtype=np.float32),
            pcb=np.zeros((0, 3, 5), dtype=np.float32))
        out = tmp_path / "o.npz"
        assemble.build_training_set(tmp_path, out, workers=2, parts=2)
        X, _, _, _ = assemble.load_training_set(out, workers=2)
        assert X.shape[0] == 2 * 3
        with np.load(out) as d:
            np.testing.assert_array_equal(d["failed_idx"], [2, 3])


@pytest.mark.slow
class TestThePrimordialSplitAgainstCLASS:
    r"""The one assumption the reduced target cannot check without the solver.

    ``tests/test_model.py`` asserts the *algebra* -- that subtracting
    :math:`\ln10A_s + (n_s-1)\ln(kh/k_*)` removes all dependence on the two.
    That is a statement about arithmetic and it would pass whether or not CLASS
    agrees.  This is the statement about physics: that ``pk_lin`` really is a
    power-law primordial spectrum times a transfer function that does not know
    what :math:`A_s` and :math:`n_s` are.

    If this fails, `emu_pk.model.primordial_ln_pk` is wrong and every spectrum
    the reduced network predicts is wrong by a smooth power law -- finite,
    plausible, and invisible to everything else in this suite.
    """

    Z = np.array([0.0, 1.0])
    K = np.logspace(-3, 1, 48)                       # h/Mpc
    BASE = dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649,
                ln10A_s=3.044, sum_mnu=0.06, w0=-1.0, wa=0.0,
                k_max_h=20.0, z_max=1.0)

    @pytest.fixture
    def ref(self):
        from emu_pk import cosmo
        pytest.importorskip("classy")
        return generate.solve(cosmo.class_params(**self.BASE), self.Z, self.K)[0]

    def test_p_is_linear_in_the_amplitude(self, ref):
        from emu_pk import cosmo
        d = dict(self.BASE, ln10A_s=self.BASE["ln10A_s"] + 0.3)
        got = generate.solve(cosmo.class_params(**d), self.Z, self.K)[0]
        assert np.allclose(got / ref, np.exp(0.3), rtol=1e-6)

    def test_the_tilt_pivots_where_k_pivot_says(self, ref):
        """And in CLASS's units, 1/Mpc, against this package's h/Mpc."""
        from emu_pk import cosmo
        dn = 0.05
        d = dict(self.BASE, n_s=self.BASE["n_s"] + dn)
        got = generate.solve(cosmo.class_params(**d), self.Z, self.K)[0]
        want = (self.K * self.BASE["h"] / cosmo.K_PIVOT) ** dn
        assert np.allclose(got / ref, want[None, :], rtol=1e-6)

    def test_the_reduced_target_is_flat_in_both(self, ref):
        """The two together, through the function training actually calls."""
        from emu_pk import box, cosmo
        from emu_pk import train as T

        lnk = np.log(self.K)
        rows, spectra = [], []
        for a, n in ((1.61, 0.84), (3.044, 0.9649), (4.0, 1.10)):
            d = dict(self.BASE, ln10A_s=a, n_s=n)
            pm = generate.solve(cosmo.class_params(**d), self.Z, self.K)[0]
            for j, zz in enumerate(self.Z):
                rows.append([d["omega_b"], d["omega_cdm"], d["h"], n, a,
                             d["sum_mnu"], d["w0"], d["wa"], zz])
                spectra.append(np.log(pm[j]))
        Y = np.array(spectra)
        T.reduce_target(Y, np.array(rows), lnk)
        # Same redshift, three very different (A_s, n_s): one curve.
        for j in range(len(self.Z)):
            same = Y[j::len(self.Z)]
            spread = float(np.abs(same - same[0]).max())
            assert spread < 1e-6, (
                f"at z={self.Z[j]} the reduced target still moves by "
                f"{spread:.2e} across the amplitude and tilt range")


class TestTheCommandLine:
    """`generate.main` is the cluster entry point; its dispatch is testable
    without a Boltzmann solver even though everything it dispatches to is not.
    """

    def test_emu_mode_reaches_the_shard_writer(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(generate, "emu_shard",
                            lambda *a, **kw: seen.update(args=a, kw=kw) or [])
        generate.main(["--mode", "emu", "--shard", "3", "--n-per-shard", "7",
                       "--n-total", "70", "--out", str(tmp_path)])
        assert seen["args"][0] == 3 and seen["args"][1] == 7
        assert seen["args"][3] == 70

    def test_ratio_mode_reaches_the_ratio_writer(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(generate, "ratio_shard",
                            lambda *a, **kw: seen.update(args=a) or tmp_path)
        generate.main(["--mode", "ratio", "--shard", "2", "--n-per-shard", "25",
                       "--out", str(tmp_path)])
        assert seen["args"][0] == 2 and seen["args"][1] == 25

    def test_an_unknown_mode_is_refused_by_the_parser(self):
        with pytest.raises(SystemExit):
            generate.main(["--mode", "nonsense"])


def test_a_solve_without_massive_neutrinos_copies_p_m_into_p_cb(monkeypatch):
    """With no massive species the cold field and the total field are the same
    field, and `pk_cb_lin` is not defined there -- so `P_cb` is a copy rather
    than a second CLASS call."""
    class FakeClass:
        def set(self, p): self.p = p
        def compute(self): pass
        def pk_lin(self, k, z): return 1.0 / (1.0 + k)
        def pk_cb_lin(self, k, z): raise AssertionError("must not be called")
        def struct_cleanup(self): pass
        def empty(self): pass

    import sys
    import types
    mod = types.ModuleType("classy")
    mod.Class = FakeClass
    monkeypatch.setitem(sys.modules, "classy", mod)

    pm, pcb = generate.solve({"h": 0.7}, [0.0], np.array([0.1, 1.0]))
    assert np.array_equal(pm, pcb)
    assert pm.shape == (1, 2)
