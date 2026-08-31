"""The entry points, and the failures they are required to produce.

Everything here is reachable from a command line or from the documented API,
and none of it needs CLASS: `generate.solve` is the only part that does, and it
is replaced.  These are the paths a user or a job script actually takes, which
makes them the ones worth pinning -- a `main` that raises `AttributeError` on a
flag is found by a queue slot otherwise, and a missing-file message is the whole
of the user experience when the file is missing.
"""
import pathlib

import numpy as np
import pytest

from emu_pk import assemble, box, cosmo, generate, grid, ratio
from emu_pk.model import PkEmulator


@pytest.fixture
def fake_solve(monkeypatch):
    """CLASS, replaced by something instant, deterministic and identifiable.

    The spectrum encodes the cosmology it came from, so a shard written for one
    node and read back as another is visible rather than merely plausible.
    """
    def solve(params, z_nodes, k_h):
        n_z, n_k = len(np.atleast_1d(z_nodes)), len(k_h)
        pm = np.full((n_z, n_k), 1.0 + params["h"], dtype=float)
        return pm, pm * 0.99
    monkeypatch.setattr(generate, "solve", solve)
    return solve


# ==========================================================================
# generate: the correction-grid shard, and the calibration
# ==========================================================================
class TestTheRatioShard:
    """`ratio_shard` has the same three behaviours `emu_shard` is tested for.

    They are what makes an array job restartable, and they were only pinned on
    the `emu` side: the two functions are separate code and a fix to one does
    not reach the other.
    """

    def test_it_writes_the_nodes_it_was_given(self, tmp_path, fake_solve):
        out = generate.ratio_shard(0, 4, tmp_path)
        assert out.name == "ratio_00000.npz"
        with np.load(out) as d:
            assert d["theta"].shape == (4, 4)          # mnu, w0, wa, f_nu
            assert d["pm"].shape == (4, len(grid.Z_NODES_RATIO), grid.N_K)
            assert d["pcb"].shape == d["pm"].shape
            # f_nu is derived from the mass, not carried alongside it, so a
            # table indexed on the fraction cannot disagree with its own mass.
            for mnu, _, _, f_nu in d["theta"]:
                assert f_nu == pytest.approx(
                    cosmo.f_nu(float(mnu), cosmo.PLANCK18["h"],
                               cosmo.PLANCK18["Omega_m"]))

    def test_a_landed_shard_is_skipped(self, tmp_path, fake_solve, capsys):
        """What makes a besteffort restart resume rather than redo."""
        generate.ratio_shard(0, 2, tmp_path)
        before = (tmp_path / "ratio_00000.npz").stat().st_mtime_ns
        capsys.readouterr()
        generate.ratio_shard(0, 2, tmp_path)
        assert "skipped" in capsys.readouterr().out
        assert (tmp_path / "ratio_00000.npz").stat().st_mtime_ns == before

    def test_a_shard_past_the_end_writes_nothing(self, tmp_path, fake_solve,
                                                 capsys):
        """An array sized above the design must not write an empty shard.

        `assemble` refuses a partial grid by counting nodes, so a zero-node
        file would be a hole that reports as a file.
        """
        n = len(generate._ratio_design())
        generate.ratio_shard(n, 25, tmp_path)
        assert "past the end" in capsys.readouterr().out
        assert list(tmp_path.glob("*.npz")) == []


class TestTheCalibration:
    """Gate 1 of a production run: seconds per solve, measured on the design.

    It is what the shard count and the walltimes are computed from, so a
    calibration that crashes or reports nothing costs an allocation.
    """

    def test_it_reports_a_rate_over_the_real_design(self, tmp_path, fake_solve,
                                                    capsys):
        generate._time_calibration(n=3)
        out = capsys.readouterr().out
        assert "=== CLASS calibration ===" in out
        assert "3 ok, 0 failed" in out
        assert "seconds/solve" in out
        assert "core-hours per 1e5 solves" in out
        assert f"{grid.N_K} modes to {grid.K_MAX}" in out
        assert f"{len(grid.Z_NODES_EMU)} per solve" in out

    def test_a_refused_corner_is_counted_not_fatal(self, tmp_path, monkeypatch,
                                                   capsys):
        """CLASS refuses some corners of any wide box.

        A calibration that aborts on the first one reports no rate at all, and
        the rate is the entire point of running it.
        """
        calls = {"n": 0}

        def flaky(params, z_nodes, k_h):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("CosmoComputationError")
            n_z, n_k = len(np.atleast_1d(z_nodes)), len(k_h)
            pm = np.ones((n_z, n_k))
            return pm, pm
        monkeypatch.setattr(generate, "solve", flaky)

        generate._time_calibration(n=3)
        out = capsys.readouterr().out
        assert "2 ok, 1 failed" in out
        assert "FAILED RuntimeError" in out
        assert "seconds/solve" in out, "one refusal must not suppress the rate"


class TestTheGenerateCommandLine:
    def test_ratio_mode_writes_a_shard(self, tmp_path, fake_solve):
        generate.main(["--mode", "ratio", "--shard", "0", "--n-per-shard", "2",
                       "--out", str(tmp_path)])
        assert (tmp_path / "ratio_00000.npz").exists()

    def test_time_mode_runs_the_calibration(self, tmp_path, fake_solve, capsys):
        generate.main(["--mode", "time", "--n-per-shard", "2"])
        assert "=== CLASS calibration ===" in capsys.readouterr().out


# ==========================================================================
# assemble: the ratio side of the merge guard, the report, and the CLI
# ==========================================================================
def _ratio_nodes(z, lnk):
    """Every node of the correction grid, exactly factorisable."""
    fid = cosmo.PLANCK18
    rows, pm = [], []
    for mnu in grid.MNU_NODES:
        for w0 in grid.W0_NODES:
            for wa in grid.WA_NODES:
                lnr = -0.5 * float(mnu) + 0.3 * (float(w0) + 1.0) + 0.2 * float(wa)
                rows.append((mnu, w0, wa,
                             cosmo.f_nu(float(mnu), fid["h"], fid["Omega_m"])))
                pm.append(np.full((len(z), len(lnk)), np.exp(lnr)))
    return np.array(rows), np.array(pm)


def _ratio_grid(tmp_path, name="ratio_00000.npz", n_z=2, n_k=3, z=None,
                lnk=None, n_shards=1):
    """The correction grid on disk, optionally split across shards.

    `n_shards` mirrors production, where the 300 nodes arrive as an OAR array
    and `build_ratio` has to stitch them back together.
    """
    z = grid.Z_NODES_RATIO[:n_z] if z is None else z
    lnk = grid.lnk_grid(n_k) if lnk is None else lnk
    fid = cosmo.PLANCK18
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows, pm = _ratio_nodes(z, lnk)
    bounds = np.array_split(np.arange(len(rows)), n_shards)
    for s, idx in enumerate(bounds):
        nm = name if n_shards == 1 else f"ratio_{s:05d}.npz"
        np.savez_compressed(tmp_path / nm, theta=rows[idx], z=z, lnk=lnk,
                            pm=pm[idx], pcb=pm[idx],
                            h_fid=fid["h"], Omega_m_fid=fid["Omega_m"])
    return tmp_path


class TestTheRatioShardsAreMergedOrRefused:
    def test_the_grid_is_stitched_back_from_many_shards(self, tmp_path):
        """Production splits the 300 nodes across an OAR array, so the normal
        case is many shards, not one.

        The table that comes out must not depend on how the nodes were
        distributed -- a merge that dropped or double-counted a node would give
        a table that is wrong only at the nodes it mishandled, which is the
        hardest kind of wrong to see downstream.
        """
        one = assemble.build_ratio(_ratio_grid(tmp_path / "a", n_shards=1),
                                   tmp_path / "one.npz", verbose=False)
        many = assemble.build_ratio(_ratio_grid(tmp_path / "b", n_shards=7),
                                    tmp_path / "many.npz", verbose=False)
        with np.load(one) as p, np.load(many) as q:
            assert sorted(p.files) == sorted(q.files)
            for key in p.files:
                assert np.array_equal(p[key], q[key]), key

    def test_ratio_shards_on_different_grids_are_refused(self, tmp_path):
        """The same guard the training-set reader has, on the other reader.

        Two shards from sweeps with different k grids describe two different
        tables; interpolating across them would be silent and wrong at every
        node, so it has to raise.
        """
        _ratio_grid(tmp_path, "ratio_00000.npz", n_k=3)
        _ratio_grid(tmp_path, "ratio_00001.npz", n_k=4)
        with pytest.raises(ValueError, match="different grid"):
            assemble.build_ratio(tmp_path, tmp_path / "t.npz", verbose=False)

    def test_the_report_names_the_cross_term_and_where_it_is_worst(
            self, tmp_path, capsys):
        """`resid_max` is what justifies the memory the full cube costs.

        It is printed as well as stored, because the number is the argument and
        a stored number nobody prints is a number nobody checks.
        """
        d = _ratio_grid(tmp_path)
        out = assemble.build_ratio(d, tmp_path / "t.npz", verbose=True)
        text = capsys.readouterr().out
        assert "factorisation this table does not use" in text
        assert "max |r_full/(r_nu . r_de) - 1|" in text
        assert "mnu=" in text and "w0=" in text and "wa=" in text
        assert f"wrote {out}" in text
        with np.load(out) as t:
            assert float(t["resid_max"]) == pytest.approx(0.0, abs=1e-9)


class TestTheTrainingSetLoaderAcceptsBothLayouts:
    def test_a_single_file_dataset_still_loads(self, tmp_path):
        """The documented fallback.

        `build_training_set` writes parts, but a dataset that predates that or
        was written by hand has `X` at the top level and no `parts` key, and
        the loader is documented as reading it.  Nothing else exercises that
        branch, so a refactor of the parts path could silently remove it.
        """
        n, nk = 7, 5
        X = np.arange(n * 9, dtype=np.float32).reshape(n, 9)
        Y = np.arange(n * nk, dtype=np.float32).reshape(n, nk)
        lnk = np.linspace(-9.0, 5.0, nk)
        np.savez(tmp_path / "one.npz", X=X, ln_pm=Y, ln_pcb=Y * 0.9, lnk=lnk)

        gX, gYm, gYcb, glnk = assemble.load_training_set(tmp_path / "one.npz")
        assert np.array_equal(gX, X)
        assert np.array_equal(gYm, Y)
        assert np.array_equal(gYcb, Y * 0.9)
        assert np.array_equal(glnk, lnk)


class TestTheAssembleCommandLine:
    def test_ratio_mode(self, tmp_path):
        d = _ratio_grid(tmp_path)
        assemble.main(["--mode", "ratio", "--shards", str(d),
                       "--out", str(tmp_path / "t.npz")])
        assert (tmp_path / "t.npz").exists()

    def test_emu_mode(self, tmp_path, fake_solve):
        generate.emu_shard(0, 4, tmp_path, n_total=4, chunk=4)
        assemble.main(["--mode", "emu", "--shards", str(tmp_path),
                       "--out", str(tmp_path / "ds.npz"),
                       "--parts", "1", "--workers", "2"])
        X, *_ = assemble.load_training_set(tmp_path / "ds.npz", workers=2)
        assert len(X) == 4 * len(grid.Z_NODES_EMU)


# ==========================================================================
# The documented failure paths of the inference API
# ==========================================================================
class TestAMissingFileSaysWhichAndWhatToDo:
    """Both shipped artefacts can be absent -- a source checkout without LFS,
    a hand-built wheel, a `--weights` pointed at the wrong path.  The message
    is the whole user experience at that moment, so it names the file *and* the
    command that makes it.
    """

    def test_missing_weights(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="emu_pk.train") as e:
            PkEmulator(tmp_path / "absent.npz")
        assert "absent.npz" in str(e.value)

    def test_missing_correction_table(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="emu_pk.assemble") as e:
            ratio.load(tmp_path / "absent.npz")
        assert "absent.npz" in str(e.value)
        assert "classy" in str(e.value)
        # The command has to be one that does something.  `python -m
        # emu_pk.ratio` has no __main__: it exits 0 having built nothing, which
        # reads as success.
        assert "emu_pk.ratio`" not in str(e.value)


class TestTheTrainedRedshiftRangeIsEnforced:
    """The box check covers the eight parameters; `z` is a ninth input and is
    not in `box.BOX`, so it is checked separately or not at all.

    Outside [0, 5] the network extrapolates exactly as it does outside the box
    -- finite, smooth and unwarranted -- and the README quotes the range as a
    validity limit, which makes it a promise.
    """

    @staticmethod
    def _emu():
        return PkEmulator()

    @pytest.mark.parametrize("z", [-0.1, 5.5, 10.0])
    def test_outside_the_range_raises(self, z):
        emu = self._emu()
        k = np.logspace(-3, 0, 8)
        theta = np.array([0.02237, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
        with pytest.raises(ValueError, match="outside the trained range"):
            emu.pk(k, z, theta)

    @pytest.mark.parametrize("z", [grid.Z_MIN, 2.5, grid.Z_MAX])
    def test_the_endpoints_are_inside(self, z):
        """Closed interval: z = 0 and z = 5 are training nodes, not exclusive
        bounds, and rejecting them would reject sigma_8's own redshift."""
        emu = self._emu()
        k = np.logspace(-3, 0, 8)
        theta = np.array([0.02237, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
        assert np.all(np.isfinite(np.asarray(emu.pk(k, z, theta))))


# ==========================================================================
# train: the guard that a dataset means what the trainer assumes
# ==========================================================================
def _dataset(tmp_path, n=24, nz=3, nk=10, ncol=9, seed=3):
    """A parts-layout training set with a controllable design width."""
    rng = np.random.default_rng(seed)
    X = rng.random((n * nz, ncol)).astype(np.float32)
    Y = rng.random((n * nz, nk)).astype(np.float32)
    np.savez(tmp_path / "ds.part000.npz", X=X, ln_pm=Y, ln_pcb=Y * 0.9)
    np.savez(tmp_path / "ds.npz", z=np.linspace(0, 3, nz),
             lnk=np.linspace(-9, 5, nk),
             parts=np.array(["ds.part000.npz"]), n_rows=np.array(len(X)),
             idx=np.arange(n), failed_idx=np.array([], dtype=np.int64))
    return tmp_path / "ds.npz"


class TestADatasetFromAnotherBoxIsRefused:
    def test_the_column_count_has_to_match_the_box(self, tmp_path):
        """A design assembled against a different box has columns that do not
        mean what `COLS` says they mean.

        It trains perfectly well and predicts nonsense, which is the failure
        mode nothing downstream can see -- so it is caught at load, by count,
        before an hour of queue time goes into it.
        """
        from emu_pk import train as T
        ds = _dataset(tmp_path, ncol=7)
        with pytest.raises(ValueError, match="design columns") as e:
            T.train(ds, tmp_path / "w.npz", n_comp=2, hidden=(4,), epochs=1,
                    batch=8, resume=False)
        assert "different box" in str(e.value)


class TestTheAblationFlagsRun:
    """Each switch replaces something in the default path, so each has to be
    exercised: a flag that raises is found by a 24-hour queue slot otherwise.
    """

    def test_unweighted_loss(self, tmp_path, capsys):
        from emu_pk import train as T
        T.train(_dataset(tmp_path), tmp_path / "w.npz", n_comp=2, hidden=(4,),
                epochs=2, batch=8, resume=False, val_frac=0.25, weighted=False)
        assert "unweighted MSE on whitened coefficients" in capsys.readouterr().out

    def test_constant_learning_rate(self, tmp_path, capsys):
        from emu_pk import train as T
        T.train(_dataset(tmp_path), tmp_path / "w.npz", n_comp=2, hidden=(4,),
                epochs=2, batch=8, resume=False, val_frac=0.25, schedule=False)
        assert "constant (--no-schedule)" in capsys.readouterr().out


class TestTheTrainCommandLine:
    def test_it_trains_and_writes_both_files(self, tmp_path):
        """`main` is what the job script calls; the keyword names it passes
        through are not checked by anything else."""
        from emu_pk import train as T
        ds = _dataset(tmp_path)
        out = tmp_path / "w.npz"
        T.main(["--dataset", str(ds), "--out", str(out), "--n-comp", "2",
                "--hidden", "4", "--epochs", "2", "--batch", "8",
                "--no-resume"])
        assert out.exists()
        assert (tmp_path / "w.resume.npz").exists()
        with np.load(out) as w:
            assert int(w["epoch"]) >= 1
            assert str(w["target_form"]) == "reduced"


# ==========================================================================
# What happens when CLASS refuses, and when a checkpoint is damaged
# ==========================================================================
class TestValidateSurvivesARefusedCorner:
    """CLASS refuses ~0.02 % of solves, all in the extreme-quintessence corner.

    `validate` runs for hours over a Latin hypercube that deliberately reaches
    the walls, so it meets them.  A refusal has to cost one point, not the run
    -- otherwise the score of the whole box depends on whether the sampler
    happened to draw a corner.
    """

    @staticmethod
    def _refuse(monkeypatch):
        from emu_pk import validate as V

        def boom(*a, **kw):
            raise RuntimeError("CosmoComputationError")
        monkeypatch.setattr(V, "_class_pk", boom)
        return V

    def test_shape_error_reports_nothing_scored_rather_than_raising(
            self, monkeypatch, capsys):
        V = self._refuse(monkeypatch)
        out = V.shape_error(_FakeEmu(), n=2, z_nodes=(0.0, 1.0), verbose=True)
        text = capsys.readouterr().out
        assert "CLASS refused a validation point" in text
        assert "shape error vs CLASS" in text, "the header still prints"
        for zz in ("0", "1"):
            assert not out["m"][zz].get("n_scored")

    def test_the_parameter_derivative_skips_the_point(self, monkeypatch):
        V = self._refuse(monkeypatch)
        out = V.derivative_error(_FakeEmu(), n=2, z_nodes=(0.0,),
                                 verbose=False)
        for p in box.PARAMS:
            assert not out["0"][p].get("n_scored")

    def test_the_redshift_derivative_skips_the_point(self, monkeypatch):
        V = self._refuse(monkeypatch)
        out = V.redshift_derivative_error(_FakeEmu(), n=2, z_nodes=(0.0, 1.0),
                                          verbose=False)
        for zz in ("0", "1"):
            assert not out[zz].get("n_scored")


class _FakeEmu:
    """The smallest thing `validate` will accept: a smooth power law in k.

    Its numbers are never scored in these tests -- every CLASS call is refused
    -- so what it returns only has to be finite and differentiable.
    """

    def predict(self, k, z, theta, which="m"):
        import jax.numpy as jnp
        k = jnp.asarray(k)
        z = jnp.atleast_1d(jnp.asarray(z, dtype=float))
        return jnp.exp(-3.0 * jnp.log(k)[None, :] - 2.0 * jnp.log1p(z)[:, None])

    def pk(self, k, z, params):
        return self.predict(k, z, params, "m")


class TestAResumeRefusesADamagedCheckpoint:
    """`besteffort` means restarts are routine, so `resume` runs constantly.

    Every one of these is a way a resume can silently make a run worse than
    starting over, which is why the trainer checks rather than trusting.
    """

    @staticmethod
    def _trained(tmp_path, **kw):
        from emu_pk import train as T
        T.train(_dataset(tmp_path), tmp_path / "w.npz", n_comp=2, hidden=(4,),
                epochs=2, batch=8, resume=False, val_frac=0.25, **kw)
        return tmp_path / "w.resume.npz"

    @staticmethod
    def _rewrite(path, **changes):
        with np.load(path) as d:
            keep = {k: d[k] for k in d.files}
        keep.update(changes)
        np.savez(path, **keep)

    def test_nan_weights_are_not_resumed_from(self, tmp_path, capsys):
        """A run that hits a NaN gradient keeps checkpointing.  Resuming from
        one of those trains a full run of NaN from a clean start."""
        from emu_pk import train as T
        r = self._trained(tmp_path)
        with np.load(r) as d:
            self._rewrite(r, W0=np.full_like(d["W0"], np.nan))
        capsys.readouterr()
        T.train(_dataset(tmp_path), tmp_path / "w.npz", n_comp=2, hidden=(4,),
                epochs=1, batch=8, resume=True, val_frac=0.25)
        out = capsys.readouterr().out
        assert "is not finite" in out
        assert "starting from scratch rather than resuming it" in out

    def test_a_missing_optimiser_array_falls_back_to_the_weights(
            self, tmp_path, capsys):
        """A checkpoint whose optimiser state is incomplete is still a good set
        of *weights*; the trainer keeps them and re-initialises Adam, out loud
        rather than silently."""
        from emu_pk import train as T
        r = self._trained(tmp_path)
        with np.load(r) as d:
            keep = {k: d[k] for k in d.files if k != "opt000"}
        np.savez(r, **keep)
        capsys.readouterr()
        T.train(_dataset(tmp_path), tmp_path / "w.npz", n_comp=2, hidden=(4,),
                epochs=1, batch=8, resume=True, val_frac=0.25)
        out = capsys.readouterr().out
        assert "no usable optimiser state in the checkpoint" in out
        assert "the weights still resume" in out
        assert "resuming from epoch" in out

    def test_a_stage_index_past_the_plan_is_clamped(self, tmp_path, capsys):
        """A staged checkpoint read against a shorter plan -- a resume after
        `STAGES` was edited, or a `--staged` run resumed without the flag.

        Indexing the plan with it would raise; clamping continues at the
        slowest rate, which is the safe direction.
        """
        from emu_pk import train as T
        r = self._trained(tmp_path, staged=False)
        self._rewrite(r, stage=np.int64(99), stage_epoch=np.int64(0),
                      stage_since=np.int64(0))
        capsys.readouterr()
        T.train(_dataset(tmp_path), tmp_path / "w.npz", n_comp=2, hidden=(4,),
                epochs=2, batch=8, resume=True, val_frac=0.25)
        assert (tmp_path / "w.npz").exists()


class TestANonFiniteLossStopsTheRun:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_it_raises_rather_than_checkpointing_nan(self, tmp_path):
        """Sixty epochs of NaN look exactly like sixty epochs of training in a
        queue log.  The run stops at the first one, and nothing is written for
        that epoch, because a NaN checkpoint also poisons the next resume.

        Fed through the `direct` head deliberately: the PCA path refuses a
        non-finite target earlier, in `fit_pca`'s SVD, which is a different
        guard.  `direct` has no SVD, so the value reaches the loss -- which is
        the guard this pins.
        """
        from emu_pk import train as T
        ds = _dataset(tmp_path)
        with np.load(ds.with_name("ds.part000.npz")) as d:
            X, Ym, Ycb = d["X"], d["ln_pm"].copy(), d["ln_pcb"]
        Ym[0, 0] = np.inf
        np.savez(ds.with_name("ds.part000.npz"), X=X, ln_pm=Ym, ln_pcb=Ycb)
        with pytest.raises(FloatingPointError, match="loss is not finite") as e:
            T.train(ds, tmp_path / "w.npz", n_comp=2, hidden=(4,), epochs=2,
                    batch=8, resume=False, val_frac=0.25, direct=True)
        assert "Nothing has been checkpointed" in str(e.value)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_the_pca_path_refuses_it_earlier(self, tmp_path):
        """The same bad data on the default path, so the two guards are not
        confused for each other: `fit_pca` cannot decompose a matrix with a
        non-finite entry and says so where it happens."""
        import numpy.linalg as LA
        from emu_pk import train as T
        ds = _dataset(tmp_path)
        with np.load(ds.with_name("ds.part000.npz")) as d:
            X, Ym, Ycb = d["X"], d["ln_pm"].copy(), d["ln_pcb"]
        Ym[0, 0] = np.inf
        np.savez(ds.with_name("ds.part000.npz"), X=X, ln_pm=Ym, ln_pcb=Ycb)
        with pytest.raises(LA.LinAlgError):
            T.train(ds, tmp_path / "w.npz", n_comp=2, hidden=(4,), epochs=2,
                    batch=8, resume=False, val_frac=0.25)


class TestValidateSolvesTheCosmologyItWasAskedFor:
    """`_class_pk` is the seam between an array and CLASS's keyword arguments.

    Every number in `validation.json` is a comparison against what this
    function solved.  If the mapping from `box.PARAMS` order onto
    `cosmo.class_params` were permuted, every score would be a comparison
    against a *different* cosmology -- self-consistently, smoothly, and with
    nothing else in the suite able to see it, because both sides of the
    comparison would still be spectra.
    """

    def test_the_parameter_vector_reaches_class_unpermuted(self, monkeypatch):
        from emu_pk import validate as V

        seen = {}

        def spy(params, z_nodes, k_h):
            seen.update(params=params, z=np.asarray(z_nodes), k=np.asarray(k_h))
            n_z, n_k = len(np.atleast_1d(z_nodes)), len(k_h)
            pm = np.ones((n_z, n_k))
            return pm, pm * 0.5
        monkeypatch.setattr(V.generate, "solve", spy)

        theta = box.sample(1, seed=4)[0]
        k = np.logspace(-3, 0, 6)
        pm, pcb = V._class_pk(theta, [0.0, 1.0], k)

        d = dict(zip(box.PARAMS, theta))
        expected = cosmo.class_params(
            h=d["h"], omega_b=d["omega_b"], omega_cdm=d["omega_cdm"],
            n_s=d["n_s"], ln10A_s=d["ln10A_s"], sum_mnu=d["sum_mnu"],
            w0=d["w0"], wa=d["wa"], k_max_h=grid.K_MAX, z_max=grid.Z_MAX)
        assert seen["params"] == expected
        # And both spectra come back from the one solve, at every z asked for.
        assert pm.shape == (2, len(k)) and pcb.shape == (2, len(k))
        assert np.array_equal(seen["z"], np.array([0.0, 1.0]))

    def test_a_scalar_redshift_is_still_one_solve_of_one_row(self, monkeypatch):
        from emu_pk import validate as V

        def spy(params, z_nodes, k_h):
            pm = np.ones((len(np.atleast_1d(z_nodes)), len(k_h)))
            return pm, pm
        monkeypatch.setattr(V.generate, "solve", spy)
        pm, _ = V._class_pk(box.sample(1, seed=4)[0], 0.5, np.logspace(-3, 0, 4))
        assert pm.shape == (1, 4)


class TestACalibrationThatRefusedEverything:
    def test_it_reports_the_failures_and_no_rate(self, monkeypatch, capsys):
        """Every solve refused is a real outcome -- a broken CLASS build, a
        wrong environment -- and the honest report is `0 ok`.  Printing a mean
        over an empty list would be a crash at the top of a production run.
        """
        def always_refuse(*a, **kw):
            raise RuntimeError("CosmoComputationError")
        monkeypatch.setattr(generate, "solve", always_refuse)

        generate._time_calibration(n=2)
        out = capsys.readouterr().out
        assert "0 ok, 2 failed" in out
        assert "seconds/solve" not in out, "no rate can be quoted from no solves"
        assert "peak RSS" in out, "the rest of the report still prints"


# ==========================================================================
# The declared public surface
# ==========================================================================
class TestEveryDeclaredNameExists:
    """`__all__` is the API promise, and nothing was checking it resolved.

    The docs say "the supported surface is what `emu_pk.__all__` and each
    module's own `__all__` list", and `README`/`CHANGELOG` make that a 1.0
    guarantee. A name in `__all__` that the module does not define is a broken
    promise that is invisible until someone writes `import *` -- which is the
    one form of import nothing in this repository uses, so it never fired.
    """

    MODULES = ["emu_pk", "emu_pk.model", "emu_pk.ratio", "emu_pk.box",
               "emu_pk.grid", "emu_pk.cosmo", "emu_pk.interp",
               "emu_pk.assemble", "emu_pk.generate", "emu_pk.train",
               "emu_pk.validate"]

    @pytest.mark.parametrize("name", MODULES)
    def test_all_resolves(self, name):
        import importlib
        mod = importlib.import_module(name)
        declared = getattr(mod, "__all__", None)
        assert declared, f"{name} declares no __all__"
        missing = [n for n in declared if not hasattr(mod, n)]
        assert not missing, f"{name}.__all__ names {missing}, which do not exist"

    @pytest.mark.parametrize("name", MODULES)
    def test_star_import_works(self, name):
        """The only way the gap above becomes visible to a user."""
        ns = {}
        exec(f"from {name} import *", ns)          # noqa: S102 - that is the test

    def test_the_top_level_reexports_are_the_real_objects(self):
        """`emu_pk.PkEmulator` and `emu_pk.model.PkEmulator` must be one object,
        not two: the docs give the short path and autodoc documents the long
        one, and a reader following either has to land on the same thing."""
        import emu_pk
        from emu_pk import model
        assert emu_pk.PkEmulator is model.PkEmulator
        assert emu_pk.primordial_ln_pk is model.primordial_ln_pk
