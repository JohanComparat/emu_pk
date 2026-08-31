"""The gradient-safety properties, tested directly rather than through a table."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from emu_pk.interp import hermite, lin_weights, pchip_slopes


def test_lin_weights_gradient_is_whole_at_a_node():
    """A node query must not halve the gradient.

    `jnp.clip(t, 0, 1)` on the *fraction* puts a query that lands exactly on a
    node on the clip boundary, and JAX splits a `minimum` tie 50/50, which
    returns exactly half the true derivative.  The fiducial neutrino mass is a
    node of the correction table, so a Fisher forecast at the fiducial would
    inherit that with no symptom other than the number.
    """
    grid = jnp.array([0.0, 1.0, 2.0, 3.0])

    def interp_at(x):
        i, t = lin_weights(x, grid)
        y = jnp.array([0.0, 10.0, 20.0, 30.0])
        return y[i] * (1 - t) + y[i + 1] * t

    # On a node, off a node: the slope is 10 everywhere, and must be 10 at both.
    assert float(jax.grad(interp_at)(1.0)) == pytest.approx(10.0)
    assert float(jax.grad(interp_at)(1.5)) == pytest.approx(10.0)
    assert float(jax.grad(interp_at)(2.0)) == pytest.approx(10.0)


def test_lin_weights_clamps_the_query_not_the_fraction():
    grid = jnp.array([1.0, 2.0, 3.0])
    i, t = lin_weights(-5.0, grid)
    assert int(i) == 0 and float(t) == pytest.approx(0.0)
    i, t = lin_weights(99.0, grid)
    assert int(i) == 1 and float(t) == pytest.approx(1.0)


def test_pchip_is_c1_across_a_node():
    """Hermite has one derivative at a node; linear interpolation has two."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.sin(x)[:, None]                       # (n, 1) -- leading axis is the axis
    d = pchip_slopes(y, x)

    def val(q):
        i = jnp.clip(jnp.searchsorted(jnp.asarray(x), q) - 1, 0, len(x) - 2)
        h = x[1] - x[0]
        t = (q - jnp.asarray(x)[i]) / h
        return hermite(jnp.asarray(y)[i, 0], jnp.asarray(y)[i + 1, 0],
                       jnp.asarray(d)[i, 0], jnp.asarray(d)[i + 1, 0], h, t)

    g = jax.grad(val)
    eps = 1e-6
    left, right = float(g(2.0 - eps)), float(g(2.0 + eps))
    assert abs(left - right) < 1e-4, "derivative jumps at a node: not C1"


def test_pchip_handles_a_two_node_axis():
    """A two-node axis has no interior, and must still produce finite slopes."""
    d = pchip_slopes(np.array([[1.0], [3.0]]), np.array([0.0, 2.0]))
    assert np.allclose(d, 1.0)


def test_pchip_handles_a_degenerate_axis():
    d = pchip_slopes(np.array([[7.0]]), np.array([0.5]))
    assert d.shape == (1, 1) and np.all(d == 0.0)


class TestTheRejectedConstruction:
    """`interp_cascaded` is the cheap construction that was not shipped.

    It carries one slope array per axis instead of every mixed partial, which
    is C1 *at a node* in every axis -- the Hermite coefficients multiplying the
    carried slopes vanish there -- and not between nodes.  It stays in the
    module and in `__all__` because it is the evidence for the expensive
    construction that did ship, and evidence nothing exercises is not evidence.
    """

    @staticmethod
    def _cube():
        """A smooth function on a 2-D grid, with its axes."""
        gx = np.array([0.0, 1.0, 2.0, 3.0])
        gy = np.array([0.0, 0.5, 1.5, 3.0])
        f = np.exp(-0.3 * gx[:, None]) * np.sin(0.7 * gy[None, :] + 0.2)
        return f, [gx, gy]

    def test_it_is_exact_on_a_node(self, ):
        """Both constructions reproduce the data where the query sits on a grid
        point; that is why the on-node case cannot tell them apart."""
        from emu_pk.interp import (cascaded_arrays, interp_cascaded,
                                   interp_tensor, tensor_arrays)

        f, grids = self._cube()
        casc = cascaded_arrays(f, grids)
        tens = tensor_arrays(f, grids)
        for i, x in enumerate(grids[0]):
            for j, y in enumerate(grids[1]):
                q = [np.asarray(x), np.asarray(y)]
                assert float(interp_cascaded(casc, grids, q)) == pytest.approx(
                    f[i, j], abs=1e-12)
                assert float(interp_tensor(tens, grids, q)) == pytest.approx(
                    f[i, j], abs=1e-12)

    def test_off_a_node_the_two_constructions_disagree(self):
        """Which is the whole finding: the cheap one is not merely cheaper.

        If this ever stops being true the cheap construction has become
        acceptable and the expensive cube is no longer justified -- so the
        assertion is that they *differ*, and it is meant to be read as the
        reason the tensor arrays ship.
        """
        from emu_pk.interp import (cascaded_arrays, interp_cascaded,
                                   interp_tensor, tensor_arrays)

        f, grids = self._cube()
        casc = cascaded_arrays(f, grids)
        tens = tensor_arrays(f, grids)
        q = [np.asarray(1.37), np.asarray(0.93)]        # off-node in both axes
        a = float(interp_cascaded(casc, grids, q))
        b = float(interp_tensor(tens, grids, q))
        assert a != pytest.approx(b, abs=1e-12)

    def test_the_expensive_construction_carries_every_mixed_partial(self):
        """2^N arrays for N axes, against N+1 for the cascaded one.  That
        count is the memory the decision costs and the reason it was close."""
        from emu_pk.interp import cascaded_arrays, tensor_arrays

        f, grids = self._cube()
        assert len(tensor_arrays(f, grids)) == 2 ** len(grids)
        assert len(cascaded_arrays(f, grids)) == len(grids) + 1
