"""The predictor, exercised on a synthetic network.

No trained weights are needed to test the machinery around the network: the
unit conversion, the box guard, the redshift vmap and -- the one that matters --
what happens outside the k grid.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

from emu_pk import box, cosmo, grid  # noqa: E402
from emu_pk.model import PkEmulator  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from emu_pk import model as M  # noqa: E402


@pytest.fixture
def toy(tmp_path):
    """A tiny network with the right shapes and arbitrary weights."""
    rng = np.random.default_rng(0)
    n_in, n_comp, n_k = len(box.PARAMS) + 1, 4, 64
    lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), n_k))
    hidden = 8
    d = {
        "W0": rng.normal(size=(n_in, hidden)) * 0.1, "b0": np.zeros(hidden),
        "beta0": np.ones(hidden), "gamma0": np.zeros(hidden),
        "W1": rng.normal(size=(hidden, 2 * n_comp)) * 0.1, "b1": np.zeros(2 * n_comp),
        "n_layers": np.int64(2), "epoch": np.int64(1), "val_loss": np.float64(0.0),
        "x_mean": np.zeros(n_in), "x_std": np.ones(n_in), "lnk": lnk,
    }
    for tag in ("m", "cb"):
        # A basis whose first component is a falling power law, so the toy
        # spectrum has a realistic slope at the top of the grid.
        basis = rng.normal(size=(n_comp, n_k)) * 0.01
        basis[0] = -3.0 * (lnk - lnk.mean()) / (lnk[-1] - lnk[0])
        d[f"pca_{tag}"] = basis
        d[f"pca_mean_{tag}"] = 10.0 - 2.0 * (lnk - lnk[0])
        d[f"coeff_mean_{tag}"] = np.zeros(n_comp)
        d[f"coeff_std_{tag}"] = np.ones(n_comp)
    p = tmp_path / "toy.npz"
    np.savez(p, **d)
    return PkEmulator(p, check_box=False)


def _theta():
    return np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])


def test_shapes_and_redshift_vmap(toy):
    k = np.logspace(-3, 1, 25)
    assert np.asarray(toy.pk(k, 0.0, _theta())).shape == (25,)
    assert np.asarray(toy.pk(k, [0.0, 0.5, 1.0], _theta())).shape == (3, 25)
    assert np.asarray(toy.pk_cb(k, 0.0, _theta())).shape == (25,)


def test_does_not_flatline_above_the_grid(toy):
    r"""The clamp, pinned.

    `jnp.interp` clamps at the edges, so a spectrum interpolated with it is
    *flat* above the last mode instead of falling -- silently, and above
    14.6 h/Mpc for a network that stops there while sigma(M) quadratures to 200.
    Here the continuation is a power law, so the log-slope on either side of the
    top mode is the same.
    """
    khi = float(np.exp(toy.lnk[-1]))
    k = np.array([khi * 0.5, khi * 0.95, khi * 2.0, khi * 8.0])
    p = np.asarray(toy.pk(k, 0.0, _theta()))
    slopes = np.diff(np.log(p)) / np.diff(np.log(k))
    assert np.all(slopes < -0.5), f"spectrum flattens outside the grid: {slopes}"
    assert slopes[-1] == pytest.approx(slopes[-2], rel=1e-6)


def test_gradient_flows_to_every_parameter(toy):
    k = np.logspace(-3, 0, 8)
    J = np.asarray(jax.jacfwd(lambda t: jnp.log(toy.pk(k, 0.3, t)))(_theta()))
    assert J.shape == (8, len(box.PARAMS))
    assert np.all(np.isfinite(J))
    # w0 and wa are the two that were structurally absent before this package.
    for p in ("w0", "wa", "sum_mnu"):
        j = box.PARAMS.index(p)
        assert np.any(J[:, j] != 0.0), f"no response to {p}"


def test_box_guard_raises_outside_and_is_skipped_when_tracing(tmp_path, toy):
    toy._check_box = True
    bad = _theta().copy()
    bad[box.PARAMS.index("h")] = 0.2
    with pytest.raises(ValueError, match="outside"):
        toy.pk(np.array([0.1]), 0.0, bad)
    # Under jit the values are tracers; the guard must be skipped, not raise.
    jax.jit(lambda t: toy.pk(np.array([0.1]), 0.0, t))(bad)


class TestTheActivationIsDifferentiableWhereItIsEvaluated:
    r"""The value is right either way.  The gradient is not, and that is worse.

    ``(gamma + (1-gamma)/(1+exp(-beta x))) x`` and the same thing through
    :func:`jax.nn.sigmoid` agree to the last bit in value.  Their reverse-mode
    derivatives do not: written out, the first carries
    :math:`e^{-\beta x}/(1+e^{-\beta x})^2`, and once :math:`\beta x \lesssim -88`
    the exponential overflows to ``inf`` in float32 -- so the value is a correct
    ``0`` and the gradient is ``inf``/``inf`` = ``NaN``.

    This cost a full training run: 150,000 cosmologies, a PCA residual of
    7.8e-6, no NaN anywhere in the data, and ``train nan  val nan`` from the end
    of epoch 1, because the weights grew until one pre-activation crossed -88
    and a single NaN gradient propagates to every parameter through Adam.
    """

    XS = np.array([-1e4, -700.0, -300.0, -100.0, -89.0, -88.0, -1.0,
                   0.0, 1.0, 88.0, 100.0, 1e4], dtype=np.float32)

    def test_the_gradient_is_finite_over_the_whole_range(self):
        g = jax.vmap(jax.grad(lambda t: M.activation(t, 1.0, 0.0)))(self.XS)
        assert np.all(np.isfinite(np.asarray(g))), np.asarray(g)

    def test_it_is_finite_for_a_learned_beta_and_gamma_too(self):
        """`beta` and `gamma` are trained, so neither stays at its initial
        value -- a large `beta` reaches the overflow at a small `x`."""
        for beta, gamma in ((30.0, 0.0), (1.0, 0.5), (100.0, 0.2), (-5.0, 0.0)):
            g = jax.vmap(jax.grad(lambda t: M.activation(t, beta, gamma)))(self.XS)
            assert np.all(np.isfinite(np.asarray(g))), (beta, gamma, np.asarray(g))

    def test_the_value_is_unchanged_by_the_stable_form(self):
        """The fix must not move the network: the two forms agree in value."""
        def naive(x, beta, gamma):
            return (gamma + (1.0 - gamma) / (1.0 + jnp.exp(-beta * x))) * x
        xs = np.linspace(-30.0, 30.0, 601, dtype=np.float64)
        for beta, gamma in ((1.0, 0.0), (2.5, 0.3)):
            np.testing.assert_allclose(
                np.asarray(M.activation(xs, beta, gamma)),
                np.asarray(naive(xs, beta, gamma)), rtol=1e-12, atol=1e-300)

    def test_the_saturated_limits_are_the_right_ones(self):
        """far negative -> gamma*x, far positive -> x."""
        for gamma in (0.0, 0.4):
            assert float(M.activation(-1e3, 1.0, gamma)) == pytest.approx(
                gamma * -1e3, rel=1e-12)
            assert float(M.activation(1e3, 1.0, gamma)) == pytest.approx(
                1e3, rel=1e-12)


class TestTrainingShipsItsBestEpoch:
    """The file that ships is the best epoch; the file that resumes is the last.

    The validation curve on the real training set is noisy at the tens-of-per-
    cent level around a descending trend, so the last epoch is not reliably the
    best one -- and every epoch overwriting one file meant shipping whichever
    epoch happened to be last.  Two files, because they answer two questions:
    the optimiser trajectory a restart needs is a property of where training
    actually is, not of where it was best.
    """

    @staticmethod
    def _dataset(tmp_path, n=40, nz=3, nk=12, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.random((n * nz, 9)).astype(np.float32)
        Y = rng.random((n * nz, nk)).astype(np.float32)
        np.savez(tmp_path / "ds.part000.npz", X=X, ln_pm=Y, ln_pcb=Y * 0.9)
        np.savez(tmp_path / "ds.npz", z=np.linspace(0, 3, nz),
                 lnk=np.linspace(-9, 5, nk),
                 parts=np.array(["ds.part000.npz"]), n_rows=np.array(len(X)),
                 idx=np.arange(n), failed_idx=np.array([], dtype=np.int64))
        return tmp_path / "ds.npz"

    def test_the_two_files_hold_different_epochs(self, tmp_path):
        from emu_pk import train as T
        out = tmp_path / "w.npz"
        T.train(self._dataset(tmp_path), out, n_comp=4, hidden=(8, 8),
                epochs=6, batch=16, resume=False, val_frac=0.25)
        with np.load(out) as w, np.load(tmp_path / "w.resume.npz") as r:
            assert int(r["epoch"]) == 6, "resume must hold the last epoch"
            assert float(w["val_loss"]) <= float(r["val_loss"]) + 1e-12, \
                "the shipped file is not the best epoch"
            assert float(r["best_val"]) == pytest.approx(float(w["val_loss"]))

    def test_a_restart_continues_from_the_last_not_the_best(self, tmp_path):
        """Otherwise a preempted run would silently rewind to its best epoch
        and redo everything after it, which on a besteffort queue is a loop."""
        from emu_pk import train as T
        ds = self._dataset(tmp_path)
        out = tmp_path / "w.npz"
        T.train(ds, out, n_comp=4, hidden=(8, 8), epochs=4, batch=16,
                resume=False, val_frac=0.25)
        with np.load(tmp_path / "w.resume.npz") as r:
            last = int(r["epoch"])
        T.train(ds, out, n_comp=4, hidden=(8, 8), epochs=6, batch=16,
                resume=True, val_frac=0.25)
        with np.load(tmp_path / "w.resume.npz") as r:
            assert int(r["epoch"]) == 6
        assert last == 4


class TestTheShippedWeightsAreTheOnesValidated:
    """The file and the claim about it must not drift apart.

    `data/validation.json` is what the paper quotes.  It was produced by
    running `validate.py` against a particular weights file, and nothing but
    this test ties the two together -- reship the weights without revalidating
    and the numbers keep looking current while describing a network that is no
    longer there.
    """

    @staticmethod
    def _both():
        import json
        import pathlib

        d = pathlib.Path(__file__).resolve().parent.parent / "emu_pk" / "data"
        with np.load(d / "emu_pk_mlp.npz") as w:
            weights = {k: w[k] for k in ("epoch", "val_loss", "best_val")}
            weights["loss_form"] = (str(w["loss_form"])
                                    if "loss_form" in w.files else "whitened_mse")
        return weights, json.loads((d / "validation.json").read_text())

    @staticmethod
    def _shape_m(v):
        """The z=0 total-matter summary, from either schema.

        `validate.py` grew a redshift axis, so what was ``{shape_m: summary}``
        is now ``{shape: {m: {z: summary}}}``.  A file in the older layout still
        describes real weights honestly, so both are read rather than one being
        declared invalid -- what must not happen is the numbers outliving the
        weights, and that is what the assertions below are for.
        """
        return v["shape_m"] if "shape_m" in v else v["shape"]["m"]["0"]

    def test_the_validation_describes_the_shipped_file(self):
        w, v = self._both()
        assert v["weights"] in ("shipped", None), (
            f"validation.json was produced against {v['weights']!r} rather "
            "than the shipped file; re-run `python -m emu_pk.validate --json "
            "emu_pk/data/validation.json` with no --weights")
        # Every held-out point asked for was scored; a median over a subset is
        # a different claim and the file records which.
        s = self._shape_m(v)
        assert s["n_scored"] == s["n_requested"]

    def test_the_validation_agrees_about_what_the_network_predicts(self):
        """A `reduced` file scored by a run that thought it was `raw` would
        report a shape error wrong by a power law and look merely bad."""
        _, v = self._both()
        if "target_form" not in v:
            pytest.skip("validation.json predates the target-form record")
        assert v["target_form"] == (
            "reduced" if PkEmulator(check_box=False)._reduced else "raw")

    def test_the_shipped_file_is_the_best_epoch_not_the_last(self):
        w, _ = self._both()
        assert float(w["val_loss"]) == pytest.approx(float(w["best_val"]))

    def test_it_is_better_than_what_it_replaced(self):
        """A guard against reshipping a worse checkpoint by accident.

        Easy to do when the trainer writes two files and one of them is the
        last epoch rather than the best.

        Guarded on `loss_form`, because it changes what `val_loss` *is*:
        unweighted it is a mean square over whitened PCA coefficients, weighted
        it is the mean squared error in ln P, and they differ by orders of
        magnitude in the direction that looks like success.  Under the weighted
        loss the comparison is against the *shape* error, which means the same
        thing in both regimes.

        The bar is a median shape error of 0.470 %.  The shipped network is at
        0.111 %, so the margin is wide -- which is the point of setting a bar
        well below what happens to be current.
        """
        w, v = self._both()
        if w["loss_form"] == "whitened_mse":
            assert float(w["val_loss"]) < 0.001481      # the 57-epoch weights
        else:
            # sqrt of the weighted loss is the RMS fractional error in P.
            assert float(np.sqrt(w["val_loss"])) < 0.0047
            assert self._shape_m(v)["median"] < 0.0047, (
                "the shipped weights are worse in shape than the 0.470 % "
                "median this bar is set at")

    def test_it_beats_the_emulator_it_exists_to_replace(self):
        """CosmoPower reproduces the CLASS shape to 0.159 %.

        That is the number this package has to beat to justify existing, so it
        is the threshold worth pinning -- on a box that is wider in every shared
        axis and carries `sum_mnu`, `w0` and `wa`, which CosmoPower does not.
        """
        _, v = self._both()
        assert self._shape_m(v)["median"] < 0.00159

    def test_the_two_analytic_derivatives_are_still_analytic(self):
        """`ln10A_s` and `n_s` are not learned; they are added back in closed
        form.  If either drifts off zero the primordial split has broken."""
        _, v = self._both()
        deriv = v.get("derivative")
        if deriv is None or "0" not in deriv:
            pytest.skip("validation.json predates the redshift sweep")
        for p in ("ln10A_s", "n_s"):
            err = deriv["0"][p]
            err = err["err"] if isinstance(err, dict) else err
            assert err < 1e-4, f"dlnP/d{p} is {err:.2e}, not analytic"


class TestThePrimordialSplitIsExact:
    r"""The claim the reduced target rests on, and the two derivatives it buys.

    :math:`P = P_\mathcal{R}(k)\,T^2(k)` with a power-law primordial spectrum,
    so ``ln10A_s`` and ``n_s`` enter ``ln P`` in closed form and the transfer
    function does not know they exist.  If that is true the reduced target is
    *exactly* independent of both, and the network's derivatives with respect to
    them are exactly right rather than fitted.  If it is false the whole
    construction is wrong by a smooth power law that no test without CLASS could
    see -- so it is asserted here on the algebra, and against CLASS itself in
    ``test_generate.py``.
    """

    @staticmethod
    def _lnk(n=64):
        return np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), n))

    def test_the_reduced_target_does_not_depend_on_amplitude_or_tilt(self):
        from emu_pk import train as T

        lnk = self._lnk()
        i = {c: T.COLS.index(c) for c in T.COLS}
        rows = np.tile(np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044,
                                 0.06, -1.0, 0.0, 0.5]), (5, 1))
        # The whole box on both axes, not a wiggle around the fiducial.
        rows[:, i["ln10A_s"]] = np.linspace(*box.BOX["ln10A_s"], 5)
        rows[:, i["n_s"]] = np.linspace(*box.BOX["n_s"], 5)

        transfer = -3.0 * lnk + 8.0 * 0.12 - 2.0 * 0.5 * np.tanh(lnk)
        Y = np.asarray(M.primordial_ln_pk(
            lnk[None, :], rows[:, i["h"]][:, None], rows[:, i["n_s"]][:, None],
            rows[:, i["ln10A_s"]][:, None])) + transfer[None, :]

        before = float((Y.max(axis=0) - Y.min(axis=0)).max())
        T.reduce_target(Y, rows, lnk)
        after = float((Y.max(axis=0) - Y.min(axis=0)).max())
        assert before > 1.0, "the test rows have to actually span something"
        assert after < 1e-12, (
            f"the reduced target still moves by {after:.2e} across the "
            "amplitude and tilt range; the split is not exact")

    def test_reduce_target_overwrites_rather_than_copies(self):
        """It is a few GB on the cluster, and the docstring promises in place."""
        from emu_pk import train as T

        lnk = self._lnk(8)
        rows = np.tile(np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044,
                                 0.06, -1.0, 0.0, 0.5]), (3, 1))
        Y = np.zeros((3, 8))
        assert T.reduce_target(Y, rows, lnk) is Y
        assert np.any(Y != 0.0)

    def test_the_analytic_derivatives_are_exact(self, tmp_path):
        r"""``dlnP/dln10A_s = 1`` and ``dlnP/dn_s = ln(kh/k_*)``, whatever the
        weights are.

        The network is random here on purpose: these two derivatives are not
        supposed to be *learned* well, they are supposed to not come from the
        network at all.  A run of this against CLASS scored 0.31 % and 1.02 %.
        """
        lnk = self._lnk()
        emu = _toy_weights(tmp_path, lnk, reduced=True)
        th = np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
        k = np.exp(lnk)

        J = np.asarray(jax.jacfwd(
            lambda t: jnp.log(emu.pk(k, 0.7, t)))(jnp.asarray(th)))
        h, n_s = th[box.PARAMS.index("h")], th[box.PARAMS.index("n_s")]

        d_as = J[:, box.PARAMS.index("ln10A_s")]
        assert np.allclose(d_as, 1.0, atol=1e-10), \
            f"dlnP/dln10A_s departs from 1 by {np.abs(d_as - 1).max():.2e}"

        d_ns = J[:, box.PARAMS.index("n_s")]
        want = lnk + np.log(h) - np.log(cosmo.K_PIVOT)
        assert np.allclose(d_ns, want, atol=1e-9), \
            f"dlnP/dn_s departs from ln(kh/k*) by {np.abs(d_ns - want).max():.2e}"

    def test_amplitude_scales_the_spectrum_exactly(self, tmp_path):
        """P is linear in A_s, so doubling ln10A_s by ln 2 doubles P."""
        lnk = self._lnk()
        emu = _toy_weights(tmp_path, lnk, reduced=True)
        k = np.exp(lnk)
        th = np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
        hi = th.copy()
        hi[box.PARAMS.index("ln10A_s")] += np.log(2.0)
        r = np.asarray(emu.pk(k, 0.7, hi)) / np.asarray(emu.pk(k, 0.7, th))
        assert np.allclose(r, 2.0, rtol=1e-10)


def _legacy_weights(tmp_path, lnk):
    """A checkpoint that declares none of its own format.

    Nine inputs, PCA output, plain z, and *no* `target_form`, `output_form` or
    `z_var` keys at all.  A file that omits them has to read back as whole
    `ln P` from nine inputs on plain z, which is what every default is set to.
    """
    emu = _toy_weights(tmp_path, lnk, reduced=False)
    d = {k: v for k, v in emu.w.items()
         if k not in ("target_form", "output_form", "z_var", "k_pivot")}
    p = tmp_path / "legacy.npz"
    np.savez(p, **d)
    return p


def _toy_weights(tmp_path, lnk, reduced: bool):
    """A random network in either target form, sharing one construction.

    The point of the reduced-form tests is what happens *around* the network,
    so the weights are arbitrary and the two forms differ only in what the file
    declares and how many inputs it takes.
    """
    rng = np.random.default_rng(1)
    keep = [p for p in box.PARAMS if not reduced or p not in ("ln10A_s", "n_s")]
    n_in, n_comp, n_k, hidden = len(keep) + 1, 4, len(lnk), 8
    d = {
        "W0": rng.normal(size=(n_in, hidden)) * 0.1, "b0": np.zeros(hidden),
        "beta0": np.ones(hidden), "gamma0": np.zeros(hidden),
        "W1": rng.normal(size=(hidden, 2 * n_comp)) * 0.1,
        "b1": np.zeros(2 * n_comp),
        "n_layers": np.int64(2), "epoch": np.int64(1),
        "val_loss": np.float64(0.0), "best_val": np.float64(0.0),
        "x_mean": np.zeros(n_in), "x_std": np.ones(n_in), "lnk": lnk,
        "params_order": np.array(keep + ["z"], dtype="U16"),
        "target_form": "reduced" if reduced else "raw",
        "k_pivot": np.float64(cosmo.K_PIVOT),
    }
    for tag in ("m", "cb"):
        basis = rng.normal(size=(n_comp, n_k)) * 0.01
        basis[0] = -3.0 * (lnk - lnk.mean()) / (lnk[-1] - lnk[0])
        d[f"pca_{tag}"] = basis
        d[f"pca_mean_{tag}"] = 10.0 - 2.0 * (lnk - lnk[0])
        d[f"coeff_mean_{tag}"] = np.zeros(n_comp)
        d[f"coeff_std_{tag}"] = np.ones(n_comp)
    p = tmp_path / f"toy_{'reduced' if reduced else 'raw'}.npz"
    np.savez(p, **d)
    return PkEmulator(p, check_box=False)


class TestACheckpointDeclaresItsOwnForm:
    """A file trained one way and read the other is wrong by a power law.

    Finite everywhere, smooth in k, and invisible to every test that does not
    involve CLASS -- the same shape of defect as the `jnp.interp` clamp this
    package already carries a test for.  So the form is written into the `.npz`
    and read back, never inferred from the shapes.
    """

    def test_a_checkpoint_predating_the_split_is_read_as_raw(self, tmp_path):
        """A file with no `target_form` at all must read back as whole `ln P`
        from nine inputs."""
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 32))
        emu = PkEmulator(_legacy_weights(tmp_path, lnk), check_box=False)
        assert emu._reduced is False
        assert len(emu.w["x_mean"]) == len(box.PARAMS) + 1

    def test_a_raw_file_is_unaffected_by_the_split(self, tmp_path):
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 32))
        emu = _toy_weights(tmp_path, lnk, reduced=False)
        k = np.exp(lnk)
        th = np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
        hi = th.copy()
        hi[box.PARAMS.index("ln10A_s")] += np.log(2.0)
        # A raw network learned the amplitude, so it does *not* respond exactly;
        # if it did, the primordial term is being added to a file that already
        # contains it.
        r = np.asarray(emu.pk(k, 0.7, hi)) / np.asarray(emu.pk(k, 0.7, th))
        assert not np.allclose(r, 2.0, rtol=1e-6)

    def test_a_file_that_does_not_describe_itself_is_refused(self, tmp_path):
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 32))
        emu = _toy_weights(tmp_path, lnk, reduced=True)
        src = dict(emu.w)

        bad = dict(src, params_order=np.array(["omega_b", "z", "h"], dtype="U16"))
        np.savez(tmp_path / "b1.npz", **bad)
        with pytest.raises(ValueError, match="last input has to be z"):
            PkEmulator(tmp_path / "b1.npz", check_box=False)

        bad = dict(src, params_order=np.array(["omega_b", "sigma8", "z"], dtype="U16"))
        np.savez(tmp_path / "b2.npz", **bad)
        with pytest.raises(ValueError, match="different box"):
            PkEmulator(tmp_path / "b2.npz", check_box=False)

        # Names and normalisation disagreeing is how a silently permuted or
        # dropped column would show up.
        bad = dict(src, x_mean=np.zeros(3), x_std=np.ones(3))
        np.savez(tmp_path / "b3.npz", **bad)
        with pytest.raises(ValueError, match="inconsistent with itself"):
            PkEmulator(tmp_path / "b3.npz", check_box=False)


class TestTheLossIsTheMetric:
    r"""Weighting by ``cstd`` is what makes the objective the reported number.

    The SVD basis is orthonormal, so a coefficient error ``dc`` is an error of
    exactly ``|dc|`` in ``ln P``.  The target is whitened, ``dc = dT * cstd``,
    and ``cstd`` spans orders of magnitude across 64 components -- so an
    unweighted MSE over the whitened target fits the components that contribute
    least to the spectrum.  This is the identity that says so.
    """

    def test_the_weighted_residual_is_the_ln_p_residual(self):
        from emu_pk import train as T

        rng = np.random.default_rng(3)
        n_k, n_comp = 80, 12
        # A smooth family, so that a handful of components really do capture it
        # -- an orthonormal basis of noise would pass this test trivially.
        lnk = np.linspace(-9, 5, n_k)
        Y = np.array([a * lnk ** 2 + b * np.tanh(lnk - c) + d
                      for a, b, c, d in rng.normal(size=(500, 4))])
        mean, basis, cmean, cstd = T.fit_pca(Y, n_comp, seed=0)

        assert np.allclose(basis @ basis.T, np.eye(n_comp), atol=1e-10), \
            "the whole weighting argument rests on the basis being orthonormal"

        dT = rng.normal(size=(40, n_comp))
        d_lnp = (dT * cstd) @ basis                    # the error in ln P
        lhs = np.mean(np.sum((dT * cstd) ** 2, axis=-1)) / n_k
        rhs = np.mean(d_lnp ** 2)
        assert lhs == pytest.approx(rhs, rel=1e-10), \
            "the weighted loss is not the mean squared error in ln P"

        # And the objective this replaced cannot tell these two apart: the
        # same whitened error, once in the component that carries the spectrum
        # and once in the component that carries almost none of it.
        big = np.zeros((1, n_comp)); big[0, 0] = 1.0
        small = np.zeros((1, n_comp)); small[0, -1] = 1.0
        assert np.mean(big ** 2) == pytest.approx(np.mean(small ** 2)), \
            "the unweighted loss is supposed to be blind to which component"
        ratio = (np.sum((big * cstd) ** 2) / np.sum((small * cstd) ** 2))
        assert ratio > 1e3, (
            f"the leading component only outweighs the last by {ratio:.3g}; "
            "on a real training set it is orders of magnitude and that is the "
            "whole reason for weighting")

    def test_sqrt_of_the_reported_loss_is_the_rms_error_in_ln_p(self, tmp_path):
        """End to end, including the factor of two for the two heads.

        A missing ``1/n_k`` or a dropped head would leave the identity above
        intact and the reported number wrong by a constant, which is exactly the
        kind of thing that survives until someone quotes it in a paper.
        """
        from emu_pk import train as T

        ds, X, lnP, lnk = _toy_dataset(tmp_path, n_cos=300, nz=4, nk=24)
        out = tmp_path / "w.npz"
        T.train(ds, out, n_comp=8, hidden=(64, 64), epochs=250, batch=128,
                resume=False, val_frac=0.2, lr=3e-3)

        emu = PkEmulator(out, check_box=False)
        k = np.exp(lnk)
        got = np.array([np.log(np.asarray(emu.pk(k, row[-1], row[:-1])))
                        for row in X[::37]])
        rms = float(np.sqrt(np.mean((got - lnP[::37]) ** 2)))
        reported = float(np.sqrt(np.load(out)["val_loss"]))
        assert reported == pytest.approx(rms, rel=0.35), (
            f"the log reports an RMS ln P error of {reported:.4%} and the "
            f"spectra are actually off by {rms:.4%}")


def _toy_dataset(tmp_path, n_cos=300, nz=4, nk=24, seed=7):
    """A dataset that is exactly primordial x transfer, so the split is exact."""
    from emu_pk import train as T

    lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), nk))
    i = {c: T.COLS.index(c) for c in T.COLS}
    theta = box.sample(n_cos, seed=seed)
    z = np.linspace(grid.Z_MIN, grid.Z_MAX, nz)
    X = np.concatenate([np.repeat(theta, nz, axis=0),
                        np.tile(z, n_cos)[:, None]], axis=1)
    transfer = (-3.0 * lnk[None, :]
                + 8.0 * X[:, i["omega_cdm"]][:, None]
                - 2.0 * X[:, i["z"]][:, None] * np.tanh(lnk)[None, :]
                + 0.7 * X[:, i["w0"]][:, None])
    lnP = (np.asarray(M.primordial_ln_pk(
        lnk[None, :], X[:, i["h"]][:, None], X[:, i["n_s"]][:, None],
        X[:, i["ln10A_s"]][:, None])) + transfer)
    np.savez(tmp_path / "ds.part000.npz", X=X.astype(np.float32),
             ln_pm=lnP.astype(np.float32), ln_pcb=lnP.astype(np.float32))
    np.savez(tmp_path / "ds.npz", z=z, lnk=lnk,
             parts=np.array(["ds.part000.npz"]), n_rows=np.array(len(X)),
             idx=np.arange(n_cos), failed_idx=np.array([], dtype=np.int64))
    return tmp_path / "ds.npz", X, lnP, lnk


class TestARestartContinuesTheOptimiser:
    """Weights were resumed and the optimiser was not, on a besteffort queue.

    `opt.init(params)` on every restart threw away Adam's moments -- wasteful
    but survivable while the learning rate was constant.  It is not survivable
    with a schedule, because the schedule's position lives in the same state:
    a job preempted every few epochs would restart at peak learning rate every
    time and never decay at all, and nothing in the log would say so.
    """

    @staticmethod
    def _steps(path):
        """The optimiser's step counters, whatever optax calls them today."""
        with np.load(path) as d:
            n = int(d["opt_n"])
            return [int(d[f"opt{i:03d}"]) for i in range(n)
                    if d[f"opt{i:03d}"].shape == ()
                    and np.issubdtype(d[f"opt{i:03d}"].dtype, np.integer)]

    def test_the_step_counter_continues_rather_than_rewinding(self, tmp_path):
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=120, nz=3, nk=16)
        out = tmp_path / "w.npz"
        kw = dict(n_comp=4, hidden=(16, 16), batch=32, val_frac=0.25)

        T.train(ds, out, epochs=3, resume=False, **kw)
        first = self._steps(tmp_path / "w.resume.npz")
        assert first and all(s > 0 for s in first), \
            "no optimiser step counter was checkpointed at all"

        T.train(ds, out, epochs=6, resume=True, **kw)
        second = self._steps(tmp_path / "w.resume.npz")
        assert second == [2 * s for s in first], (
            f"steps went {first} -> {second}; six epochs must leave twice the "
            "counter of three, or the restart rewound the schedule")

    def test_a_checkpoint_without_optimiser_state_still_resumes_the_weights(
            self, tmp_path):
        """Older checkpoints exist, and weights are the valuable half."""
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=120, nz=3, nk=16)
        out = tmp_path / "w.npz"
        kw = dict(n_comp=4, hidden=(16, 16), batch=32, val_frac=0.25)
        T.train(ds, out, epochs=3, resume=False, **kw)

        rp = tmp_path / "w.resume.npz"
        with np.load(rp) as d:
            np.savez(rp, **{k: d[k] for k in d.files
                            if not k.startswith("opt")})
        T.train(ds, out, epochs=5, resume=True, **kw)
        with np.load(rp) as d:
            assert int(d["epoch"]) == 5

    def test_a_non_finite_moment_is_refused(self, tmp_path):
        """A NaN in an Adam moment poisons every later step and `val_loss`
        cannot see it."""
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=120, nz=3, nk=16)
        out = tmp_path / "w.npz"
        kw = dict(n_comp=4, hidden=(16, 16), batch=32, val_frac=0.25)
        T.train(ds, out, epochs=3, resume=False, **kw)

        rp = tmp_path / "w.resume.npz"
        with np.load(rp) as d:
            poisoned = {k: d[k] for k in d.files}
        victim = next(k for k in poisoned
                      if k.startswith("opt") and poisoned[k].ndim >= 1)
        poisoned[victim] = np.full_like(poisoned[victim], np.nan)
        np.savez(rp, **poisoned)

        T.train(ds, out, epochs=5, resume=True, **kw)
        with np.load(rp) as d:
            assert int(d["epoch"]) == 5
            assert all(np.all(np.isfinite(d[f"opt{i:03d}"]))
                       for i in range(int(d["opt_n"])))


class TestAResumeChecksShapesNotJustNames:
    """A checkpoint of one shape can sit where a run of another shape writes.

    Narrowing the input from nine columns to seven leaves `n_layers` identical
    and every parameter *name* identical, and changes only `W0` from (9, h) to
    (7, h).  A resume that checks names and layer count accepts that and dies at
    the first matmul -- in a queue slot, after the training set has loaded.
    """

    def test_a_raw_checkpoint_is_refused_by_a_reduced_run(self, tmp_path, capsys):
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=120, nz=3, nk=16)
        out = tmp_path / "w.npz"
        kw = dict(n_comp=4, hidden=(16, 16), batch=32, val_frac=0.25)

        T.train(ds, out, epochs=2, resume=False, reduced=False, **kw)
        with np.load(out) as d:
            assert d["W0"].shape[0] == len(box.PARAMS) + 1     # nine inputs
            assert str(d["target_form"]) == "raw"

        T.train(ds, out, epochs=2, resume=True, reduced=True, **kw)
        assert "different architecture" in capsys.readouterr().out
        with np.load(out) as d:
            assert d["W0"].shape[0] == len(box.PARAMS) - 1     # seven
            assert str(d["target_form"]) == "reduced"

    def test_a_matching_checkpoint_is_still_resumed(self, tmp_path, capsys):
        """The guard must not refuse everything, which would pass the test above
        and quietly turn every preemption into a restart from scratch."""
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=120, nz=3, nk=16)
        out = tmp_path / "w.npz"
        kw = dict(n_comp=4, hidden=(16, 16), batch=32, val_frac=0.25)
        T.train(ds, out, epochs=2, resume=False, **kw)
        T.train(ds, out, epochs=4, resume=True, **kw)
        assert "resuming from epoch 2" in capsys.readouterr().out


class TestTheScheduleSurvivesAShortRun:
    """optax gives the cosine whatever is left after the warmup.

    A warmup at least as long as the whole run leaves it zero steps and optax
    raises -- and a two-epoch smoke against a two-epoch default warmup is
    exactly that, which is how it was found, in a submitted job.
    """

    @pytest.mark.parametrize("epochs", [1, 2, 3])
    def test_a_run_shorter_than_the_warmup_still_trains(self, tmp_path, epochs):
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=120, nz=3, nk=16)
        out = tmp_path / f"w{epochs}.npz"
        T.train(ds, out, epochs=epochs, resume=False, n_comp=4, hidden=(16, 16),
                batch=32, val_frac=0.25, warmup_epochs=5)
        with np.load(out) as d:
            assert int(d["epoch"]) <= epochs
            assert np.isfinite(float(d["val_loss"]))


class TestTheOutputRepresentationIsDeclared:
    """PCA or direct, written into the file and read back.

    CosmoPower's choice for this quantity is the direct form: their released
    linear-matter model is `PKLIN_NN`, a `cosmopower_NN`, and they report having
    tested PCA against it and found the direct form more accurate -- reserving
    PCA for spectra with negative values, where the logarithm is unavailable.
    `ln P` is positive.

    Both are kept, so the difference is measurable rather than argued, which
    means a checkpoint has to say which one it is.
    """

    def test_direct_and_pca_both_round_trip(self, tmp_path):
        from emu_pk import train as T

        ds, X, lnP, lnk = _toy_dataset(tmp_path, n_cos=200, nz=3, nk=24)
        k = np.exp(lnk)
        for direct in (False, True):
            out = tmp_path / f"w_{int(direct)}.npz"
            T.train(ds, out, epochs=120, resume=False, n_comp=8, hidden=(32, 32),
                    batch=64, val_frac=0.2, lr=3e-3, direct=direct)
            with np.load(out) as d:
                assert str(d["output_form"]) == ("direct" if direct else "pca")
                n_out = d[f"W{int(d['n_layers']) - 1}"].shape[1]
                assert n_out == (2 * len(lnk) if direct else 16)
                have = set(d.files)
            # Each form carries its own decode arrays and not the other's.
            if direct:
                assert {"feat_mean_m", "feat_std_m"} <= have
                assert "pca_m" not in have
            else:
                assert {"pca_m", "coeff_std_m"} <= have
                assert "feat_mean_m" not in have

            emu = PkEmulator(out, check_box=False)
            got = np.log(np.asarray(emu.pk(k, X[0][-1], X[0][:-1])))
            assert np.all(np.isfinite(got))
            assert got.shape == (len(lnk),)

    def test_an_unknown_output_form_is_refused(self, tmp_path):
        """Rather than decoded as whichever branch happens to be the default."""
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 24))
        emu = _toy_weights(tmp_path, lnk, reduced=True)
        bad = dict(emu.w, output_form="wavelets")
        np.savez(tmp_path / "bad.npz", **bad)
        with pytest.raises(ValueError, match="output_form"):
            PkEmulator(tmp_path / "bad.npz", check_box=False)

    def test_a_file_without_output_form_is_read_as_pca(self, tmp_path):
        """Every checkpoint written before this key existed is a PCA one."""
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 32))
        p = _legacy_weights(tmp_path, lnk)
        with np.load(p) as d:
            assert "output_form" not in d.files
        assert PkEmulator(p, check_box=False)._output_form == "pca"


class TestTheStagedScheduleActuallyChangesTheLearningRate:
    """CosmoPower's schedule is five learning rates, and it has to be five.

    `jax.jit` traces a function once and caches on the argument signature, so a
    Python value captured from the enclosing scope is baked into the trace and
    rebinding it afterwards does not retrace.  A single jitted `step` reading
    `opt` from the enclosing scope therefore keeps the optimiser it was first
    traced with -- every stage runs at the first stage's learning rate, the
    schedule does nothing, and *nothing* about the loss curve says so.
    """

    def test_a_frozen_stage_is_frozen(self, tmp_path, capsys):
        """The lever this test pulls: a stage at 1e-12 cannot move weights."""
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=200, nz=3, nk=16)
        T.train(ds, tmp_path / "w.npz", resume=False, n_comp=4, hidden=(16, 16),
                batch=32, val_frac=0.25, staged=True,
                stages=((1e-3, 99, 4), (1e-12, 99, 4)))
        lines = [l for l in capsys.readouterr().out.splitlines()
                 if l.strip().startswith("epoch 2.")]
        vals = [float(l.split("val")[1].split()[0]) for l in lines]
        assert len(vals) == 4
        assert len(set(vals)) == 1, (
            f"validation moved during a 1e-12 stage ({vals}); the stage's "
            "learning rate is not reaching the update")

    def test_early_stopping_ends_a_stage(self, tmp_path, capsys):
        """A stage runs until its rate has nothing left, not for a fixed count.

        Not tested by freezing the learning rate: at 1e-12 the *biases* still
        move, because they start at exactly 0 and 1e-12 is perfectly
        representable there, so a strict `<` keeps finding a new best by a
        fifteenth decimal place forever.  A realistic stage that converges and
        then bounces is the honest lever.
        """
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=200, nz=3, nk=16)
        T.train(ds, tmp_path / "w.npz", resume=False, n_comp=4, hidden=(16, 16),
                batch=32, val_frac=0.25, staged=True, lr=3e-3,
                stages=((3e-3, 3, 400),))
        out = capsys.readouterr().out
        assert "stopped at epoch" in out, out[-800:]
        ran = sum(l.strip().startswith("epoch 1.") for l in out.splitlines())
        assert ran < 400, "it used the whole budget; early stopping did nothing"

    def test_a_preempted_staged_run_resumes_into_its_stage(self, tmp_path):
        """Otherwise a restart rewinds to 1e-2 and undoes every decade of
        learning rate the run had bought -- on a queue where preemption is
        routine rather than exceptional."""
        from emu_pk import train as T

        ds, *_ = _toy_dataset(tmp_path, n_cos=200, nz=3, nk=16)
        out = tmp_path / "w.npz"
        kw = dict(n_comp=4, hidden=(16, 16), batch=32, val_frac=0.25,
                  staged=True)
        # Run the first two stages, then "preempt" by simply stopping.
        T.train(ds, out, resume=False,
                stages=((1e-3, 99, 3), (1e-4, 99, 3)), **kw)
        with np.load(tmp_path / "w.resume.npz") as d:
            assert int(d["stage"]) == 1, "should have finished in stage 2"
            assert int(d["epoch"]) == 6

        # Resume against the full three-stage plan: it must continue in stage 2,
        # not restart at stage 1.
        T.train(ds, out, resume=True,
                stages=((1e-3, 99, 3), (1e-4, 99, 3), (1e-5, 99, 3)), **kw)
        with np.load(tmp_path / "w.resume.npz") as d:
            assert int(d["stage"]) == 2
            assert int(d["epoch"]) == 9, "epochs should accumulate across stages"


class TestTheRedshiftVariable:
    r"""`ln P` is nearly linear in `log10(1+z)`, and that is where the z=0 hook
    comes from.

    A network can fit values superbly (0.18 % shape error) and get `dlnP/dz`
    wrong by 16 % **at z=0 only** -- 0.02 % at z=0.5, 0.01 % at z=1.  z=0 is a
    training node in value and an endpoint in slope: nothing on the z<0 side
    constrains it, so the variable has to leave the network enough freedom to
    bend.

    How much freedom depends on the variable.  Against CLASS over z in [0, 5],
    departure from a straight line is 0.196 in `log10(1+z)`, 0.359 in `z` and
    0.782 in `a`; from a cubic, 2.6e-3 against 9.0e-3 and 8.3e-2.  The reason is
    exact: :math:`d\ln P/d\log_{10}(1+z) = -2\ln(10)\,f(z)`, and the growth rate
    `f` is bounded in roughly [0.5, 1].
    """

    def test_the_transform_is_what_it_says(self):
        for z in (0.0, 0.5, 5.0):
            assert float(M.Z_VARS["log10_1pz"](jnp.asarray(z))) == \
                pytest.approx(np.log10(1.0 + z))
            assert float(M.Z_VARS["z"](jnp.asarray(z))) == pytest.approx(z)

    def test_the_public_interface_is_still_z(self, tmp_path):
        """The transform is internal.  `pk(k, z, ...)` takes a redshift and
        `jax.grad` of it is still d/dz, by the chain rule."""
        from emu_pk import train as T

        ds, X, lnP, lnk = _toy_dataset(tmp_path, n_cos=200, nz=4, nk=24)
        out = tmp_path / "w.npz"
        T.train(ds, out, epochs=120, resume=False, n_comp=8, hidden=(32, 32),
                batch=64, val_frac=0.2, lr=3e-3, z_var="log10_1pz")
        with np.load(out) as d:
            assert str(d["z_var"]) == "log10_1pz"
        emu = PkEmulator(out, check_box=False)
        k = np.exp(lnk)
        th = np.array([0.0224, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
        # A finite difference in z must match autodiff in z.
        h = 1e-4
        fd = (np.log(np.asarray(emu.pk(k, 0.7 + h, th)))
              - np.log(np.asarray(emu.pk(k, 0.7 - h, th)))) / (2 * h)
        ad = np.asarray(jax.jacfwd(
            lambda s: jnp.log(emu.pk(k, s, th)))(0.7))
        assert np.allclose(ad, fd, rtol=1e-4, atol=1e-6)

    def test_an_unknown_variable_is_refused(self, tmp_path):
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 24))
        emu = _toy_weights(tmp_path, lnk, reduced=True)
        np.savez(tmp_path / "bad.npz", **dict(emu.w, z_var="conformal_time"))
        with pytest.raises(ValueError, match="z_var"):
            PkEmulator(tmp_path / "bad.npz", check_box=False)

    def test_files_without_the_key_are_plain_z(self, tmp_path):
        lnk = np.log(np.logspace(np.log10(grid.K_MIN), np.log10(grid.K_MAX), 32))
        p = _legacy_weights(tmp_path, lnk)
        with np.load(p) as d:
            assert "z_var" not in d.files
        assert PkEmulator(p, check_box=False)._z_var == "z"


def test_reloading_a_rewritten_weights_file_sees_the_new_weights(tmp_path):
    """`load_weights` is cached, and caching on the path alone is wrong.

    Retraining to the same filename and reloading returned the previous
    network, silently.  It cost a pilot comparison: two configurations wrote to
    one output name, the second scored the first one's weights, and the two
    identical rows read as a genuine null result.
    """
    p = tmp_path / "w.npz"
    np.savez(p, marker=np.array([1.0]))
    assert M.load_weights(p)["marker"][0] == 1.0
    np.savez(p, marker=np.array([2.0]))
    assert M.load_weights(p)["marker"][0] == 2.0


def test_the_validation_file_names_every_choice_that_changes_the_network():
    """`validation.json` is what gets quoted; it has to say what it scored.

    A record that omits `output_form` or `z_var` reads as the default, which
    makes a network trained `direct` on `log10(1+z)` indistinguishable from a
    PCA network on plain z in its own validation file.
    """
    import json
    import pathlib

    d = pathlib.Path(__file__).resolve().parent.parent / "emu_pk" / "data"
    v = json.loads((d / "validation.json").read_text())
    emu = PkEmulator(check_box=False)
    for key, got in (("target_form", "reduced" if emu._reduced else "raw"),
                     ("output_form", emu._output_form),
                     ("z_var", emu._z_var)):
        assert key in v, f"validation.json does not record {key}"
        assert v[key] == got, f"validation.json says {key}={v[key]!r}, weights say {got!r}"


class TestTheColdAndTotalSpectraAreConsistent:
    r"""Physics the network is never told, and should satisfy anyway.

    With massive neutrinos, :math:`\delta_m = (1-f_\nu)\delta_{cb} +
    f_\nu\delta_\nu`.  On large scales the neutrinos cluster with everything
    else and the two spectra converge; on small scales they free-stream,
    :math:`\delta_\nu \to 0`, and :math:`P_m \to (1-f_\nu)^2 P_{cb}`.

    Both come from one network with two heads, which is what stops them
    drifting apart -- but nothing in the loss enforces either limit, so they
    are worth asserting on the shipped weights.
    """

    K = np.logspace(-4, 1, 300)
    TH = np.array([0.02237, 0.1200, 0.6736, 0.9649, 3.044, 0.30, -1.0, 0.0])

    @staticmethod
    def _f_nu(theta):
        from emu_pk import cosmo
        h = theta[box.PARAMS.index("h")]
        om = ((theta[box.PARAMS.index("omega_b")]
               + theta[box.PARAMS.index("omega_cdm")]) / h ** 2
              + cosmo.omega_nu(theta[box.PARAMS.index("sum_mnu")], h))
        return cosmo.f_nu(theta[box.PARAMS.index("sum_mnu")], h, om)

    def test_the_cold_field_has_more_power_than_the_total(self):
        emu = PkEmulator(check_box=False)
        pm = np.asarray(emu.pk(self.K, 0.0, self.TH))
        pcb = np.asarray(emu.pk_cb(self.K, 0.0, self.TH))
        assert np.all(pcb >= pm * (1 - 1e-6)), "P_cb dips below P_m"

    def test_they_converge_on_large_scales(self):
        emu = PkEmulator(check_box=False)
        r = (np.asarray(emu.pk_cb(self.K, 0.0, self.TH))
             / np.asarray(emu.pk(self.K, 0.0, self.TH)))
        assert r[0] == pytest.approx(1.0, abs=2e-3)

    def test_the_small_scale_limit_is_the_free_streaming_one(self):
        emu = PkEmulator(check_box=False)
        r = (np.asarray(emu.pk_cb(self.K, 0.0, self.TH))
             / np.asarray(emu.pk(self.K, 0.0, self.TH)))
        want = 1.0 / (1.0 - self._f_nu(self.TH)) ** 2
        assert r[-1] == pytest.approx(want, rel=0.02), (
            f"P_cb/P_m -> {r[-1]:.4f}, free streaming says {want:.4f}")

    def test_without_massive_neutrinos_they_are_the_same_field(self):
        """`generate.solve` writes P_cb as a *copy* of P_m at zero mass, so the
        network sees identical targets -- but the two heads decode
        independently and nothing forces them equal, so the residual is a
        property of the fit rather than of the data."""
        emu = PkEmulator(check_box=False)
        th = self.TH.copy()
        th[box.PARAMS.index("sum_mnu")] = 0.0
        r = (np.asarray(emu.pk_cb(self.K, 0.0, th))
             / np.asarray(emu.pk(self.K, 0.0, th)))
        assert np.abs(r - 1.0).max() < 5e-3
