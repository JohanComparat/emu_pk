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
