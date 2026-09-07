"""The sampling box."""
import numpy as np
import pytest

from emu_pk import box


def test_design_is_inside_the_box_and_reproducible():
    a = box.sample(500)
    b = box.sample(500)
    assert np.array_equal(a, b), "the design must follow from the seed alone"
    for j, p in enumerate(box.PARAMS):
        lo, hi = box.BOX[p]
        assert a[:, j].min() >= lo and a[:, j].max() <= hi


def test_design_excludes_early_dark_energy_domination():
    d = box.sample(1000)
    w0 = d[:, box.PARAMS.index("w0")]
    wa = d[:, box.PARAMS.index("wa")]
    assert np.all(w0 + wa < 0.0)


def test_box_is_wider_than_cosmopower_where_it_matters():
    """The two bounds that make an external emulator unusable here."""
    assert box.BOX["h"][0] < 0.64, "a wide h prior must not leave the box"
    assert "w0" in box.BOX and "wa" in box.BOX, "CPL must be a response, not a hole"


def test_check_names_every_offending_axis():
    with pytest.raises(ValueError) as e:
        box.check({"h": 0.4, "n_s": 2.0, "omega_b": 0.02})
    msg = str(e.value)
    assert "h" in msg and "n_s" in msg and "omega_b" not in msg


def test_check_ignores_traced_values():
    """`None` stands for a value that is a tracer; the check is skipped, not failed."""
    box.check({"h": None, "n_s": None})


class TestTheBoxCarriesCurvature:
    r"""``Omega_k`` is the ninth axis, and the bound is a measurement.

    CLASS solves the *whole* box at :math:`|\Omega_k| \le 0.15`.  Wider, it
    starts refusing on the closed side where the density is lowest: at
    ``omega_cdm = 0.05``, ``h = 0.85`` (:math:`\Omega_m = 0.108`) it fails from
    :math:`\Omega_k = -0.275`, and at :math:`\Omega_m = 0.261` it survives to
    :math:`-0.40`.  The open side never refuses at all -- not even where the
    closure drives :math:`\Omega_{de}` to :math:`-0.4` -- which is why that side
    needs a stated bound rather than a solver error to find it.
    """

    def test_it_is_the_last_axis(self):
        """Appended, not inserted.  Every other `PARAMS.index` keeps its value,
        so a checkpoint's `_in_idx` and a shard's columns stay comparable."""
        assert box.PARAMS[-1] == "Omega_k"
        assert box.PARAMS[:8] == ("omega_b", "omega_cdm", "h", "n_s",
                                  "ln10A_s", "sum_mnu", "w0", "wa")

    def test_the_bounds_bracket_flat(self):
        lo, hi = box.BOX["Omega_k"]
        assert (lo, hi) == (-0.15, 0.15)
        assert lo < 0.0 < hi

    def test_every_parameter_has_bounds(self):
        assert set(box.BOX) == set(box.PARAMS)

    def test_the_design_spans_the_axis(self):
        d = box.sample(2000, seed=5)
        ok = d[:, box.PARAMS.index("Omega_k")]
        lo, hi = box.BOX["Omega_k"]
        assert lo <= ok.min() and ok.max() <= hi
        # A Latin hypercube fills each axis by construction; if this ever fails
        # the column is constant, which is what a mis-wired `pin` looks like.
        assert ok.min() < lo * 0.9 and ok.max() > hi * 0.9

    def test_a_curved_point_is_inside_and_a_wilder_one_is_not(self):
        assert box.inside(box.sample(1, seed=7)[0]) == {}
        bad = dict(zip(box.PARAMS, box.sample(1, seed=7)[0]))
        bad["Omega_k"] = 0.4
        assert "Omega_k" in box.inside(bad)
        with pytest.raises(ValueError, match="Omega_k"):
            box.check(bad)


class TestPinHoldsAColumnFixed:
    """The flat control the curved design is measured against.

    Without it, "the nine-parameter fit is worse" and "this design is smaller
    than the shipped one" are the same observation.  The pin makes the two arms
    differ in the design and in nothing else -- same box, same seed, same
    network shape.
    """

    def test_the_pinned_column_is_constant(self):
        d = box.sample(200, pin={"Omega_k": 0.0})
        assert np.ptp(d[:, box.PARAMS.index("Omega_k")]) == 0.0

    def test_it_can_pin_to_a_non_zero_value(self):
        d = box.sample(50, pin={"Omega_k": 0.1})
        assert np.allclose(d[:, box.PARAMS.index("Omega_k")], 0.1)

    def test_the_other_columns_are_untouched(self):
        """Pinning is applied after the draw, so the rest of the design is the
        one the seed names -- which is what makes the two arms comparable."""
        free = box.sample(200, seed=11)
        pinned = box.sample(200, seed=11, pin={"Omega_k": 0.0})
        j = box.PARAMS.index("Omega_k")
        assert np.array_equal(np.delete(free, j, axis=1),
                              np.delete(pinned, j, axis=1))

    def test_pinning_an_unknown_name_raises(self):
        with pytest.raises(ValueError, match="cannot pin"):
            box.sample(4, pin={"Omega_lambda": 0.7})

    def test_no_pin_is_the_default(self):
        assert np.array_equal(box.sample(20, seed=3),
                              box.sample(20, seed=3, pin=None))
