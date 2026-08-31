"""The correction table: the identity it must satisfy, and the continuity."""
import os

import jax
import numpy as np
import pytest

from emu_pk import ratio

_DATA = os.path.join(os.path.dirname(__file__), "..", "emu_pk", "data")
LEGACY = os.path.join(_DATA, "class_nu_ratio_legacy.npz")
FULL = os.path.join(_DATA, "class_pk_ratio.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(LEGACY),
                                reason="legacy table not present")

K = np.logspace(-3, 1, 40)


def test_exactly_one_at_the_lambdacdm_massless_corner():
    """Not 'small': *exactly* one.

    This is what lets the correction be applied unconditionally, with no
    `if sum_mnu > 0` branch -- and a Python branch on the neutrino mass is a
    branch on a value that is a tracer under `jax.grad`, which would break the
    gradient the differentiable path exists to provide.
    """
    for fn in (ratio.suppression_m, ratio.suppression_cb):
        r = np.asarray(fn(K, 0.0, 0.0, table=LEGACY))
        assert np.all(r == 1.0)


def test_reproduces_ggah_mod_bit_for_bit():
    """The move between packages must change no number."""
    ggah = pytest.importorskip("ggah_mod.cosmology.nu_ratio")
    from ggah_mod.cosmology.parameters import Cosmology
    k = np.logspace(-4, np.log10(200.0), 97)
    for mnu in (0.0, 0.06, 0.234, 0.5):
        for z in (0.0, 0.5, 2.4):
            c = Cosmology.create(sum_mnu=mnu)
            for old, new in ((ggah.suppression_m, ratio.suppression_m),
                             (ggah.suppression_cb, ratio.suppression_cb)):
                a = np.asarray(old(k, c, z))
                b = np.asarray(new(k, float(c.f_nu), z, table=LEGACY))
                assert np.array_equal(a, b)


def test_derivative_is_continuous_across_a_node():
    """C1 in f_nu, which linear interpolation is not.

    The fiducial 0.06 eV is a node of the table, so this is the derivative a
    Fisher forecast at the fiducial actually takes.
    """
    with np.load(LEGACY) as d:
        f_nodes = d["f_nu"]
    node = float(f_nodes[3])
    g = jax.grad(lambda f: ratio.suppression_m(np.array([1.0]), f, 0.5,
                                               table=LEGACY)[0])
    eps = 1e-9
    assert float(g(node - eps)) == pytest.approx(float(g(node + eps)), rel=1e-5)


def test_refuses_to_extrapolate():
    with pytest.raises(ValueError, match="outside"):
        ratio.suppression_m(K, 0.9, 0.0, table=LEGACY)
    with pytest.raises(ValueError, match="outside"):
        ratio.suppression_m(K, 0.0, 99.0, table=LEGACY)


def test_survives_jit_and_grad():
    f = jax.jit(lambda fnu, z: ratio.suppression_m(K, fnu, z, table=LEGACY))
    a = np.asarray(f(0.004, 0.3))
    b = np.asarray(ratio.suppression_m(K, 0.004, 0.3, table=LEGACY))
    assert np.allclose(a, b, rtol=1e-12)


# ==========================================================================
# The four-axis table
# ==========================================================================
full_only = pytest.mark.skipif(not os.path.exists(FULL),
                               reason="the four-axis table has not been built")


@full_only
def test_dark_energy_is_a_response_not_a_constant():
    """The gap this table exists to close.

    Through the LambdaCDM-trained emulator alone, dlnP/dw0 is not small -- it is
    identically zero, which a Fisher matrix reads as a flat direction rather
    than as an error.
    """
    k = np.array([1.0])
    tab = ratio.load(FULL)
    f = float(tab["grids"][0][3])
    lo = float(ratio.suppression_m(k, f, 0.5, -1.3, 0.0)[0])
    mid = float(ratio.suppression_m(k, f, 0.5, -1.0, 0.0)[0])
    hi = float(ratio.suppression_m(k, f, 0.5, -0.7, 0.0)[0])
    assert lo > mid > hi
    g = jax.grad(lambda w: ratio.suppression_m(k, f, 0.5, w, 0.0)[0])
    assert abs(float(g(-1.0))) > 1e-3


@full_only
def test_still_exactly_one_at_the_lambdacdm_massless_corner():
    k = np.logspace(-3, 1, 30)
    for fn in (ratio.suppression_m, ratio.suppression_cb):
        assert np.all(np.asarray(fn(k, 0.0, 0.0, -1.0, 0.0, table=FULL)) == 1.0)


@full_only
@pytest.mark.parametrize("axis", ["f_nu", "w0", "wa", "z"])
@pytest.mark.parametrize("others", ["on-node", "off-node"])
def test_derivative_is_continuous_in_every_axis(axis, others):
    """C1 in all four axes, and *between* nodes as well as on them.

    The cheap construction -- one slope array per axis rather than every mixed
    partial -- is C1 at a node in every axis, because the Hermite basis
    coefficients that multiply the carried slopes vanish there.  Between nodes
    it is not: measured, it leaves a 1.6 percent jump in dr/dw0 at an off-node
    w0.  That is why this table carries all 2^N mixed partials, and why this
    test parametrises over `off-node` as well as `on-node`.
    """
    k = np.array([1.0])
    tab = ratio.load(FULL)
    fnode = float(tab["grids"][0][3])
    base = (dict(f=fnode, z=0.5, w=-1.0, a=0.0) if others == "on-node"
            else dict(f=fnode * 1.31, z=0.63, w=-1.07, a=0.11))
    node = {"f_nu": fnode, "w0": -1.0, "wa": 0.0, "z": 0.5}[axis]

    def at(x):
        d = dict(base)
        d[{"f_nu": "f", "w0": "w", "wa": "a", "z": "z"}[axis]] = x
        return ratio.suppression_m(k, d["f"], d["z"], d["w"], d["a"],
                                   table=FULL)[0]

    g = jax.grad(at)
    eps = 1e-7
    left, right = float(g(node - eps)), float(g(node + eps))
    rel = abs(left - right) / max(abs(left), 1e-30)
    assert rel < 1e-4, f"d/d{axis} jumps by {rel:.2e} across its node"


@full_only
def test_only_the_requested_spectrum_is_built():
    """Derivative arrays are 2^N per spectrum; a run that asks for one pays once."""
    ratio.load.__wrapped__ if hasattr(ratio.load, "__wrapped__") else None
    ratio._load_resolved.cache_clear()
    tab = ratio.load(FULL)
    assert tab["_derivs"] == {}
    ratio.suppression_m(np.array([1.0]), 0.0, 0.0, -1.0, 0.0, table=FULL)
    assert "lnr_m" in tab["_derivs"] and "lnr_cb" not in tab["_derivs"]


def test_the_declared_bounds_are_the_shipped_ones():
    """The module says where the table is valid; the table has to agree.

    This drifted once already.  The shipped ``class_nu_ratio.npz`` was built on
    the nodes ``[0, .02, .04, .06, .09, .12, .18, .25, .35, .5]`` in float32
    while ``build()``'s defaults produced ``[0, .02, .05, .06, .1, .15, .2, .3,
    .4, .5]`` in float64 -- two different grids under one filename, neither
    checked against the other, and the refusal boundary quoted in the paper
    belonging to whichever happened to be on disk.

    ``MNU_MAX``, ``W0_RANGE`` and ``WA_RANGE`` are what ``_refuse_outside``
    enforces, so if they and the axes ever part company the package refuses the
    wrong region: either extrapolating past the table in silence, or declining
    cosmologies it can actually serve.
    """
    import numpy as np

    from emu_pk import ratio

    d = np.load(ratio._NPZ)
    assert float(d["mnu"].max()) == pytest.approx(ratio.MNU_MAX)
    assert float(d["mnu"].min()) == 0.0, "the exact-identity corner must be a node"
    assert (float(d["w0"].min()), float(d["w0"].max())) == ratio.W0_RANGE
    assert (float(d["wa"].min()), float(d["wa"].max())) == ratio.WA_RANGE
    # Every axis strictly increasing, or the Hermite slopes are meaningless.
    for ax in ("mnu", "f_nu", "w0", "wa"):
        assert np.all(np.diff(d[ax]) > 0.0), ax
