"""The scoring, exercised against an emulator whose answer is known.

`validate.py` produces every number this package claims, and had no tests at
all.  It cannot be tested against CLASS in CI -- that is the `[gen]` install and
minutes per solve -- but it does not need to be: what has to be right is the
*arithmetic* of scoring, and that is checkable against an analytic spectrum
whose shape error and derivatives are known in closed form.

The solver is replaced by a spectrum that is smooth, positive and differentiable
in every argument:

    ln P(k, z; theta) = A + n ln k + sum_j c_j theta_j - 2 ln(1+z)

so `dlnP/dtheta_j = c_j` exactly, `dlnP/dz = -2/(1+z)` exactly, and an emulator
that returns the same expression scores zero.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

from emu_pk import box  # noqa: E402
from emu_pk import validate as V  # noqa: E402

# Coefficients of the synthetic spectrum, one per box parameter.  Chosen
# non-zero and of different magnitudes so a permuted column would show.
COEF = np.array([3.0, -2.0, 1.5, 0.7, 1.0, -4.0, 0.5, -0.25])
AMPL, TILT = 10.0, -1.5


def _ln_p(k, z, theta):
    k = jnp.asarray(k)
    return (AMPL + TILT * jnp.log(k)
            + jnp.dot(jnp.asarray(theta), jnp.asarray(COEF))
            - 2.0 * jnp.log1p(jnp.asarray(z)))


class FakeEmulator:
    """Reproduces the synthetic spectrum, optionally with a known defect.

    ``amp`` scales it (a pure amplitude error, which the shape metric must
    *not* see) and ``tilt`` adds a slope in ln k (which it must).
    """

    def __init__(self, amp=1.0, tilt=0.0, dz=0.0):
        self.amp, self.tilt, self.dz = amp, tilt, dz
        self._reduced, self._output_form, self._z_var = True, "direct", "z"
        self.w = {}

    def predict(self, k, z, params, which="m"):
        k = jnp.asarray(k)
        scalar = np.ndim(z) == 0
        zz = jnp.atleast_1d(jnp.asarray(z, dtype=float))
        out = jax.vmap(lambda s: jnp.exp(
            _ln_p(k, s, params)
            + jnp.log(self.amp)
            + self.tilt * jnp.log(k)
            + self.dz * s))(zz)
        return out[0] if scalar else out


@pytest.fixture
def fake_class(monkeypatch):
    """Replace the solver with the same analytic spectrum."""
    def _pk(theta, z, k):
        zz = np.atleast_1d(np.asarray(z, dtype=float))
        pm = np.array([np.asarray(jnp.exp(_ln_p(k, s, theta))) for s in zz])
        # P_cb sits *above* P_m with massive neutrinos, not below: on small
        # scales the neutrinos do not cluster, so delta_m ~ (1-f_nu) delta_cb
        # and P_m ~ (1-f_nu)^2 P_cb.  The factor here is arbitrary; the sign is
        # not.
        return pm, pm * 1.01
    monkeypatch.setattr(V, "_class_pk", _pk)
    return _pk


# ==========================================================================
# Shape
# ==========================================================================
class TestShapeError:
    def test_a_perfect_emulator_scores_zero(self, fake_class):
        out = V.shape_error(FakeEmulator(), n=4, z_nodes=(0.0, 1.0),
                            which=("m",), verbose=False)
        for z in ("0", "1"):
            assert out["m"][z]["median"] < 1e-10
            assert out["m"][z]["max"] < 1e-10

    def test_a_pure_amplitude_error_is_not_a_shape_error(self, fake_class):
        """The metric renormalises, so a spectrum right in shape and wrong in
        amplitude must not be scored as wrong in both."""
        out = V.shape_error(FakeEmulator(amp=1.5), n=4, z_nodes=(0.0,),
                            which=("m",), verbose=False)
        assert out["m"]["0"]["max"] < 1e-10

    def test_but_that_amplitude_error_is_still_reported(self, fake_class):
        """...and it must not vanish entirely.

        The renormalisation is what makes the shape number meaningful, but on
        its own it would let a spectrum wrong by a constant factor score
        perfectly. The discarded factor is reported beside the shape, so the
        two together are an accuracy claim about `P(k)` rather than about its
        shape alone.
        """
        out = V.shape_error(FakeEmulator(amp=1.5), n=4, z_nodes=(0.0,),
                            which=("m",), verbose=False)["m"]["0"]
        assert out["max"] < 1e-10, "the shape must still be perfect"
        assert out["amplitude"]["median"] == pytest.approx(0.5, rel=1e-9)
        assert out["amplitude"]["max"] == pytest.approx(0.5, rel=1e-9)

    def test_a_perfect_emulator_has_no_amplitude_error_either(self, fake_class):
        out = V.shape_error(FakeEmulator(), n=4, z_nodes=(0.0, 1.0),
                            which=("m",), verbose=False)
        for z in ("0", "1"):
            assert out["m"][z]["amplitude"]["max"] < 1e-10

    def test_the_amplitude_is_the_offset_at_k_norm_exactly(self, fake_class):
        """`FakeEmulator`'s tilt is `k**d`, pivoted at k = 1 rather than at
        K_NORM, so it moves the value at K_NORM as well as the shape.

        That makes it a closed-form check on the amplitude number: the reported
        figure has to be `|k_norm**d - 1|` and nothing else, which pins that it
        is read at K_NORM and is not picking up any of the shape.
        """
        d = 1e-3
        k = np.logspace(np.log10(V.K_TRUSTED[0]), np.log10(V.K_TRUSTED[1]), 300)
        k0 = k[int(np.argmin(abs(k - V.K_NORM)))]      # the grid point used
        out = V.shape_error(FakeEmulator(tilt=d), n=2, z_nodes=(0.0,),
                            which=("m",), verbose=False)["m"]["0"]
        assert out["amplitude"]["max"] == pytest.approx(abs(k0 ** d - 1),
                                                        rel=1e-9)
        assert out["max"] > 1e-6, "the tilt is still a shape error too"

    def test_a_tilt_is_a_shape_error_of_the_right_size(self, fake_class):
        r"""A tilt `d` multiplies the renormalised ratio by
        `(k/k_norm)^d`, so the max error over the scored range is
        `|(k_max/k_norm)^d - 1|` at the ends of the range."""
        d = 1e-3
        out = V.shape_error(FakeEmulator(tilt=d), n=2, z_nodes=(0.0,),
                            which=("m",), verbose=False)
        edge = max(abs((V.K_TRUSTED[1] / V.K_NORM) ** d - 1),
                   abs((V.K_TRUSTED[0] / V.K_NORM) ** d - 1))
        assert out["m"]["0"]["max"] == pytest.approx(edge, rel=1e-3)

    def test_both_spectra_come_from_one_solve(self, fake_class, monkeypatch):
        """Scoring P_m and P_cb separately would double the only expensive
        part of this for no new information."""
        calls = []
        orig = V._class_pk
        monkeypatch.setattr(V, "_class_pk",
                            lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1])
        V.shape_error(FakeEmulator(), n=5, z_nodes=(0.0, 1.0),
                      which=("m", "cb"), verbose=False)
        assert len(calls) == 5, f"{len(calls)} solves for 5 cosmologies x 2 spectra"

    def test_a_refused_point_is_recorded_not_hidden(self, monkeypatch):
        """A median over 3 of 4 is a different claim from a median over 4."""
        def _pk(theta, z, k):
            if theta[0] > box.BOX["omega_b"][0] + 0.001:
                raise RuntimeError("CosmoComputationError")
            zz = np.atleast_1d(np.asarray(z, dtype=float))
            pm = np.array([np.asarray(jnp.exp(_ln_p(k, s, theta))) for s in zz])
            return pm, pm
        monkeypatch.setattr(V, "_class_pk", _pk)
        out = V.shape_error(FakeEmulator(), n=8, z_nodes=(0.0,),
                            which=("m",), verbose=False)
        s = out["m"]["0"]
        assert s["n_requested"] == 8
        assert s["n_scored"] < 8


# ==========================================================================
# Where in the box
# ==========================================================================
class TestWhereInBox:
    def test_the_centre_is_not_an_edge(self):
        mid = np.array([(lo + hi) / 2 for lo, hi in
                        (box.BOX[p] for p in box.PARAMS)])
        assert V.where_in_box(mid)["edge"] == pytest.approx(0.5)

    def test_a_point_on_a_wall_has_edge_zero(self):
        th = np.array([box.BOX[p][0] for p in box.PARAMS])
        assert V.where_in_box(th)["edge"] == pytest.approx(0.0)

    def test_the_quintessence_corner_is_reported(self):
        th = np.array([(lo + hi) / 2 for lo, hi in
                       (box.BOX[p] for p in box.PARAMS)])
        th[box.PARAMS.index("w0")] = -0.6
        th[box.PARAMS.index("wa")] = 0.5
        assert V.where_in_box(th)["w0_plus_wa"] == pytest.approx(-0.1)
        assert V.where_in_box(th)["w0_plus_wa"] > V.QUINTESSENCE_CORNER

    def test_summary_splits_edge_from_interior(self):
        where = [{"edge": 0.01, "w0_plus_wa": -1.0},     # edge
                 {"edge": 0.40, "w0_plus_wa": -1.0},     # interior
                 {"edge": 0.30, "w0_plus_wa": -0.05}]    # interior, corner
        s = V._summary([1.0, 2.0, 3.0], where, 3)
        assert s["n_scored"] == 3 and s["edge"]["n"] == 1
        assert s["interior"]["n"] == 2
        assert s["quintessence_corner"]["n"] == 1
        assert s["edge"]["median"] == pytest.approx(1.0)

    def test_summary_of_nothing_says_so(self):
        s = V._summary([], [], 5)
        assert s["n_scored"] == 0 and s["n_requested"] == 5


# ==========================================================================
# Derivatives
# ==========================================================================
class TestDerivativeError:
    def test_a_perfect_emulator_scores_zero(self, fake_class):
        out = V.derivative_error(FakeEmulator(), n=3, z_nodes=(0.0, 1.0),
                                 convergence=False, verbose=False)
        for z in ("0", "1"):
            for p in box.PARAMS:
                assert out[z][p]["err"] < 1e-6, p

    def test_the_floor_is_reported_when_asked(self, fake_class):
        out = V.derivative_error(FakeEmulator(), n=2, z_nodes=(0.0,),
                                 convergence=True, verbose=False)
        for p in box.PARAMS:
            assert out["0"][p]["floor"] is not None
            # The synthetic spectrum is exactly linear in theta, so its own
            # central difference is exact and the two step sizes agree.
            assert out["0"][p]["floor"] < 1e-6

    def test_without_convergence_there_is_no_floor(self, fake_class):
        out = V.derivative_error(FakeEmulator(), n=2, z_nodes=(0.0,),
                                 convergence=False, verbose=False)
        assert out["0"]["h"]["floor"] is None

    def test_a_step_leaving_the_box_is_skipped(self, fake_class):
        """`_class_dlnp` returns None rather than asking CLASS for a cosmology
        outside the box the design was drawn from."""
        k = np.array([0.1])
        th = np.array([box.BOX[p][0] for p in box.PARAMS])   # on the low wall
        assert V._class_dlnp(th, 0, 1.0, np.array([0.0]), k, "m") is None


class TestRedshiftDerivative:
    def test_a_perfect_emulator_scores_the_stencil_and_the_floor_says_so(
            self, fake_class):
        """A perfect emulator does **not** score zero here, and should not.

        The metric compares exact autodiff against a *finite difference* of the
        reference, so a network that is right scores the stencil's own
        truncation error.  That is the whole reason `floor` is reported, and
        the test worth having is that the two agree: if the score of a perfect
        emulator were far above its floor, the floor would not be measuring the
        floor.
        """
        out = V.redshift_derivative_error(FakeEmulator(), n=3,
                                          z_nodes=(0.0, 0.5, 1.0), verbose=False)
        for z in ("0", "0.5", "1"):
            err, floor = out[z]["err"], out[z]["floor"]
            assert err < 1e-2, z
            assert floor is not None and floor > 0, z
            # Truncation goes as the square of the step, so halving it removes
            # three quarters: floor ~ 0.75 x err for a perfect emulator.
            assert err < 5 * floor, f"z={z}: err {err:.2e} vs floor {floor:.2e}"

    def test_z_zero_uses_a_forward_stencil(self, fake_class):
        """A central difference at z=0 would ask for z<0, which is not a
        redshift.  The forward stencil is second order, so its floor should sit
        within an order of magnitude of the central ones' rather than blowing
        up."""
        out = V.redshift_derivative_error(FakeEmulator(), n=2,
                                          z_nodes=(0.0, 1.0), verbose=False)
        assert out["0"]["floor"] is not None and out["1"]["floor"] is not None
        assert out["0"]["floor"] < 20 * out["1"]["floor"]

    def test_a_known_z_error_is_measured(self, fake_class):
        """dlnP/dz of the truth is -2/(1+z), which is -1 at z=1; an emulator
        carrying an extra `dz*z` term is wrong by `dz` in absolute terms, so
        by `dz` relative at that node."""
        dz = 1e-2
        out = V.redshift_derivative_error(FakeEmulator(dz=dz), n=2,
                                          z_nodes=(1.0,), verbose=False)
        assert out["1"]["err"] == pytest.approx(dz, rel=0.1)


def test_k_norm_is_not_the_primordial_pivot():
    """Two different quantities that happen to share a number.

    `K_NORM` is where the shape comparison renormalises, in **h/Mpc**;
    `cosmo.K_PIVOT` is the primordial pivot, in **1/Mpc**.  The reduced target
    makes the second one load-bearing, so the names stay distinct: `validate`
    must not define a second `K_PIVOT` for a reader to conflate with the
    first.
    """
    assert "K_PIVOT" not in vars(V), "validate re-defines K_PIVOT"
    assert V.K_NORM == 0.05


class TestTheCommandLine:
    """`validate.main` is what writes the file the README quotes."""

    def test_it_writes_a_json_that_describes_what_it_scored(
            self, fake_class, tmp_path, monkeypatch):
        monkeypatch.setattr(V, "PkEmulator", lambda *a, **kw: FakeEmulator())
        out = tmp_path / "v.json"
        res = V.main(["--n-shape", "3", "--n-deriv", "2",
                      "--z", "0.0", "1.0", "--no-convergence",
                      "--json", str(out)])
        import json
        on_disk = json.loads(out.read_text())
        assert on_disk == json.loads(json.dumps(res, sort_keys=True))

        # Everything a reader needs to know which network produced this.
        for key in ("target_form", "output_form", "z_var", "loss_form",
                    "weights", "z_nodes", "k_trusted", "n_shape", "n_deriv"):
            assert key in on_disk, key
        assert on_disk["shape"]["m"]["0"]["n_requested"] == 3
        assert on_disk["z_nodes"] == [0.0, 1.0]

    def test_it_scores_both_spectra(self, fake_class, tmp_path, monkeypatch):
        monkeypatch.setattr(V, "PkEmulator", lambda *a, **kw: FakeEmulator())
        res = V.main(["--n-shape", "2", "--n-deriv", "1", "--z", "0.0",
                      "--no-convergence"])
        assert set(res["shape"]) == {"m", "cb"}

    def test_the_verbose_tables_print(self, fake_class, capsys):
        """The terminal output is what a run is watched through; a formatting
        error there is only found by printing it."""
        V.shape_error(FakeEmulator(), n=2, z_nodes=(0.0, 1.0), verbose=True)
        V.derivative_error(FakeEmulator(), n=1, z_nodes=(0.0,),
                           convergence=True, verbose=True)
        V.redshift_derivative_error(FakeEmulator(), n=1, z_nodes=(0.0, 1.0),
                                    verbose=True)
        out = capsys.readouterr().out
        assert "shape error vs CLASS" in out
        assert "derivative error vs CLASS" in out
        assert "dlnP/dz vs CLASS" in out
        assert "floor" in out, "the metric's own noise floor must be shown"


class TestTheTotalIsReportedToo:
    """`amplitude` and `shape` are separate statements; `total` is the one
    number that covers `P(k)` itself, so it is measured rather than inferred
    from the other two."""

    def test_a_pure_amplitude_error_is_the_total(self, fake_class):
        """With the shape exact, the total is the amplitude and nothing else."""
        out = V.shape_error(FakeEmulator(amp=1.5), n=4, z_nodes=(0.0,),
                            which=("m",), verbose=False)["m"]["0"]
        assert out["max"] < 1e-10
        assert out["total"]["max"] == pytest.approx(0.5, rel=1e-9)

    def test_a_perfect_emulator_has_no_total_error(self, fake_class):
        out = V.shape_error(FakeEmulator(), n=4, z_nodes=(0.0,),
                            which=("m",), verbose=False)["m"]["0"]
        assert out["total"]["max"] < 1e-10

    def test_the_total_is_never_below_the_amplitude(self, fake_class):
        """The amplitude is the error at one k; the total is the max over all
        of them, so the total bounds it from above by construction."""
        for emu in (FakeEmulator(amp=1.02), FakeEmulator(tilt=1e-3),
                    FakeEmulator(amp=1.02, tilt=1e-3)):
            s = V.shape_error(emu, n=3, z_nodes=(0.0,), which=("m",),
                              verbose=False)["m"]["0"]
            assert s["total"]["max"] >= s["amplitude"]["max"] - 1e-12
