r"""Inference: a linear-P(k) network evaluated in pure JAX.

Nothing here imports ``optax``, ``classy`` or anything else outside the core
install.  Loading a trained model needs numpy and jax and nothing more, which is
what lets ``ggah_mod`` depend on this package without inheriting a Boltzmann
solver or a training stack.

The network is a dense MLP over CosmoPower's architecture -- four layers of 512
with a learned :math:`(\gamma + (1-\gamma)\sigma(\beta x))x` activation -- whose
output is read one of two ways, declared by the checkpoint's ``output_form``:

``direct``
    standardised :math:`\ln P` at every wavenumber.  This is what CosmoPower's
    own linear-matter model does (``PKLIN_NN`` is a ``cosmopower_NN``), and they
    report testing PCA against it and preferring this.

``pca``
    coefficients on a 64-component basis.  Supported, and not what the shipped
    weights use: a basis makes every coefficient error non-local in ``k``.

Weights are a plain ``.npz``; there is no framework to reconstruct.
"""

from __future__ import annotations

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from . import box, cosmo, grid
from .interp import concrete

__all__ = ["PkEmulator", "load_weights", "activation", "primordial_ln_pk",
           "Z_VARS", "DEFAULT_WEIGHTS"]

DEFAULT_WEIGHTS = pathlib.Path(__file__).resolve().parent / "data" / "emu_pk_mlp.npz"

# Positions in `box.PARAMS` of the three parameters the primordial term needs.
# Read by index rather than by unpacking a dict, because `_forward` runs under
# `jax.grad` and the argument is one array.
_I_H = box.PARAMS.index("h")
_I_NS = box.PARAMS.index("n_s")
_I_LN10AS = box.PARAMS.index("ln10A_s")


#: Redshift variables the network may be fed.  The value is what goes in; the
#: emulator's public interface is always ``z``, and the chain rule through these
#: is what ``jax.grad`` differentiates, so ``dP/dz`` is still ``dP/dz``.
Z_VARS = {
    "z": lambda z: z,
    "log10_1pz": lambda z: jnp.log10(1.0 + z),
}


def primordial_ln_pk(lnk, h, n_s, ln10A_s, k_pivot=cosmo.K_PIVOT):
    r"""The closed-form part of :math:`\ln P`: amplitude and tilt.

    .. math::

        \ln P(k) = \ln10A_s + (n_s - 1)\,\ln\!\big(k h / k_*\big)
                   + \ln T^2(k;\ \text{everything else})

    In linear theory with a power-law primordial spectrum this split is
    *exact*: :math:`P = P_\mathcal{R}(k)\,T^2(k)`, the transfer function does
    not know what :math:`A_s` or :math:`n_s` are, and CLASS's ``pk_lin`` is that
    product.  So the network is given the second term only and this one is added
    back here.

    Two things follow, and they are the reason for the split rather than side
    effects of it:

    * :math:`\partial\ln P/\partial \ln10A_s = 1` and
      :math:`\partial\ln P/\partial n_s = \ln(kh/k_*)` become *exact*, where
      a network that learned them scored 0.31 % and 1.02 %.  A Fisher matrix
      built on this one is exactly right in two of its eight directions.
    * Amplitude and tilt are the two largest variance directions in the target
      over this box -- a factor ~11 in amplitude and ~44 in tilt across a
      wavenumber range of 14.5 e-folds.  Removing them analytically is capacity
      the PCA and the network get back for the transfer function, the BAO and
      the neutrino suppression, which is where the error actually is.

    ``lnk`` is :math:`\ln k` with :math:`k` in **h/Mpc**, this package's
    convention, and ``k_pivot`` is in **1/Mpc**, CLASS's -- hence the explicit
    factor of ``h``.  The two units meeting in one expression is exactly the
    kind of thing that is wrong by :math:`h` and self-consistent everywhere, so
    it happens once, here.

    The constant :math:`-10\ln 10` from :math:`A_s = 10^{-10}e^{\ln10A_s}` is
    deliberately *not* included: it does not depend on the cosmology, so it is
    absorbed into the PCA mean and carrying it would only be one more place for
    train and inference to disagree.

    Arguments broadcast: pass ``lnk`` as ``(n_k,)`` against scalars for one
    spectrum, or against ``(n_rows, 1)`` columns for a whole training set.
    """
    return jnp.asarray(ln10A_s) + (jnp.asarray(n_s) - 1.0) * (
        jnp.asarray(lnk) + jnp.log(jnp.asarray(h)) - jnp.log(k_pivot))


def activation(x, beta, gamma):
    r"""The CosmoPower activation, :math:`[\gamma + (1-\gamma)\,\sigma(\beta x)]\,x`.

    Learned per unit.  It interpolates between linear (:math:`\gamma = 1`) and a
    smooth gated non-linearity, and unlike a ReLU it is :math:`C^\infty` -- which
    matters here because the whole purpose of this network is that something
    differentiates it.  A kinked activation gives a kinked
    :math:`\partial P/\partial\theta`, and a Fisher matrix built on it inherits
    the kink at whatever point the fiducial happens to land.

    **The sigmoid is** :func:`jax.nn.sigmoid` **and not** ``1/(1 + exp(-bx))``.
    The two agree to the last bit in *value*; they do not in *gradient*.  Written
    out, the reverse-mode derivative carries a term
    :math:`e^{-\beta x}/(1+e^{-\beta x})^2`, and once :math:`\beta x \lesssim -88`
    the exponential overflows to ``inf`` in float32: the value is still a correct
    ``0``, and the gradient is ``inf``/``inf`` = ``NaN``.  :func:`jax.nn.sigmoid`
    is the numerically stable form; it differentiates to :math:`s(1-s)`, which is
    ``0`` there.

    This is not hypothetical.  The first full training run on 150,000
    cosmologies returned ``train nan  val nan`` at the end of epoch 1, with a
    PCA residual of :math:`7.8\times10^{-6}` and no NaN anywhere in the data:
    the weights had simply grown until some pre-activation crossed :math:`-88`,
    and one NaN gradient poisons every parameter through Adam.  An activation
    whose whole purpose is that something differentiates it has to be
    differentiable everywhere it is *evaluated*, not merely where it was tested.
    """
    return (gamma + (1.0 - gamma) * jax.nn.sigmoid(beta * x)) * x


@functools.lru_cache(maxsize=4)
def _load_weights(path: str, mtime_ns: int, size: int) -> dict:
    with np.load(path) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_weights(path=None) -> dict:
    """Trained weights, cached, held as **numpy**.

    numpy rather than ``jnp`` for the same reason the correction table is: an
    array cached during a ``jit`` trace carries that trace with it, and the next
    transformation dies with an ``UnexpectedTracerError``.

    **Keyed on the file's mtime and size, not on its path alone.**  Caching on
    the path is right until something rewrites that path, and then it is
    silently wrong: retraining to the same filename and reloading returned the
    *previous* network, with no error and no warning.  That is not hypothetical
    -- it invalidated a pilot comparison here, where two configurations wrote
    to one output name and the second scored the first one's weights, producing
    two identical rows that looked like a real null result.
    """
    p = DEFAULT_WEIGHTS if path is None else pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing.  Train it with `python -m emu_pk.train` on a "
            "generated training set, or point `PkEmulator(weights=...)` at one.")
    st = p.stat()
    return _load_weights(str(p.resolve()), st.st_mtime_ns, st.st_size)


class PkEmulator:
    """Linear :math:`P(k,z)` over the box of :mod:`emu_pk.box`.

    Matches the ``LinearPowerSpectrum`` contract ``ggah_mod`` expects: ``pk``
    and ``pk_cb``, plus ``has_native_z`` and ``differentiable`` declared rather
    than inferred.  Both spectra come from one network with two output heads, so
    they cannot drift apart the way two separately trained models would.
    """

    name = "emu_pk"
    has_native_z = True
    differentiable = True

    def __init__(self, weights=None, check_box: bool = True):
        self.w = load_weights(weights)
        self._check_box = bool(check_box)
        self.lnk = jnp.asarray(self.w["lnk"])

        # -- what this particular file was trained to predict -----------------
        # Read from the checkpoint, never assumed.  A file written before the
        # primordial split predicts whole `ln P` from nine inputs; one written
        # after predicts the reduced target from seven.  Loading either and
        # guessing wrong gives a spectrum that is wrong by a power law and
        # finite everywhere, which is the failure mode this package keeps
        # finding and the reason the form is stored rather than inferred.
        self._reduced = str(self.w.get("target_form", "raw")) == "reduced"
        # Files predating the direct mode carry no `output_form` and are all
        # PCA, so that is the default rather than an error.
        # What the redshift column was fed as.  `log10(1+z)` is the variable
        # `ln P` is nearly linear in -- `dlnP/dlog10(1+z) = -2 ln(10) f(z)`, and
        # the growth rate `f` is bounded in about [0.5, 1] -- so the network has
        # to represent far less curvature along it, and correspondingly less
        # freedom to invent any.  Measured against CLASS over z in [0, 5]:
        # departure from a straight line 0.196 against 0.359 in z and 0.782 in
        # a, and from a cubic 2.6e-3 against 9.0e-3 and 8.3e-2.
        self._z_var = str(self.w.get("z_var", "z"))
        if self._z_var not in Z_VARS:
            raise ValueError(
                f"the checkpoint declares z_var={self._z_var!r}; this version "
                f"knows {sorted(Z_VARS)}.")
        self._output_form = str(self.w.get("output_form", "pca"))
        if self._output_form not in ("pca", "direct"):
            raise ValueError(
                f"the checkpoint declares output_form={self._output_form!r}, "
                "which this version does not know how to decode.")
        self.k_pivot = float(self.w.get("k_pivot", cosmo.K_PIVOT))
        order = [str(p) for p in
                 self.w.get("params_order", np.array(list(box.PARAMS) + ["z"]))]
        if not order or order[-1] != "z":
            raise ValueError(
                f"the checkpoint's params_order is {order!r}; the network's "
                "last input has to be z.")
        unknown = [p for p in order[:-1] if p not in box.PARAMS]
        if unknown:
            raise ValueError(
                f"the checkpoint feeds inputs this package does not name: "
                f"{unknown}.  It was written against a different box.")
        self._in_idx = np.array([box.PARAMS.index(p) for p in order[:-1]],
                                dtype=int)
        n_in = len(self.w["x_mean"])
        if n_in != len(order):
            raise ValueError(
                f"the checkpoint names {len(order)} inputs but normalises "
                f"{n_in}: the file is inconsistent with itself.")

    # -- the box ------------------------------------------------------------
    def _validate(self, params, z):
        """Skipped under tracing, where the values are not available.

        Attempting it there would raise ``ConcretizationTypeError`` and break
        the gradient this whole package exists to provide.  A jitted forward
        model is checked once when it is built, outside the trace.
        """
        if not self._check_box:
            return
        vals = {p: concrete(v) for p, v in zip(box.PARAMS, params)}
        vals = {p: v for p, v in vals.items() if v is not None}
        if vals:
            box.check(vals)
        zc = concrete(jnp.max(jnp.asarray(z)))
        if zc is not None and not grid.Z_MIN <= zc <= grid.Z_MAX:
            raise ValueError(
                f"z = {zc:g} is outside the trained range "
                f"[{grid.Z_MIN:g}, {grid.Z_MAX:g}].")

    # -- the network --------------------------------------------------------
    def _forward(self, params, z, which: str):
        # `params` is always the full box.PARAMS vector; which of it the network
        # eats is the checkpoint's business, not the caller's.  Under the
        # reduced target ln10A_s and n_s are not fed in at all -- they enter
        # analytically at the bottom of this function instead.
        p = jnp.asarray(params, dtype=float).ravel()
        zt = Z_VARS[self._z_var](jnp.asarray(z, dtype=float))
        x = jnp.concatenate([p[self._in_idx], jnp.atleast_1d(zt)])
        x = (x - jnp.asarray(self.w["x_mean"])) / jnp.asarray(self.w["x_std"])
        n_layers = int(self.w["n_layers"])
        for i in range(n_layers - 1):
            x = activation(x @ jnp.asarray(self.w[f"W{i}"]) + jnp.asarray(self.w[f"b{i}"]),
                           jnp.asarray(self.w[f"beta{i}"]),
                           jnp.asarray(self.w[f"gamma{i}"]))
        coeff = x @ jnp.asarray(self.w[f"W{n_layers - 1}"]) + jnp.asarray(self.w[f"b{n_layers - 1}"])
        if self._output_form == "direct":
            # Standardised ln P per wavenumber; no basis to project through.
            scale = jnp.asarray(self.w[f"feat_std_{which}"])
            off = jnp.asarray(self.w[f"feat_mean_{which}"])
            n_k = scale.shape[0]
            head = coeff[..., :n_k] if which == "m" else coeff[..., n_k:]
            lnp = head * scale + off
        else:
            # PCA decode, per head.
            basis = jnp.asarray(self.w[f"pca_{which}"])      # (n_comp, n_k)
            mean = jnp.asarray(self.w[f"pca_mean_{which}"])  # (n_k,)
            scale = jnp.asarray(self.w[f"coeff_std_{which}"])
            off = jnp.asarray(self.w[f"coeff_mean_{which}"])
            head = (coeff[..., : basis.shape[0]] if which == "m"
                    else coeff[..., basis.shape[0]:])
            lnp = (head * scale + off) @ basis + mean        # on self.lnk
        if not self._reduced:
            return lnp
        # `self._reduced` is a Python bool read at load time, so this is a
        # static branch: `jit` traces one side of it and the gradient never
        # sees a conditional.
        return lnp + primordial_ln_pk(self.lnk, p[_I_H], p[_I_NS],
                                      p[_I_LN10AS], self.k_pivot)

    def _interp_lnk(self, lnp, k):
        r"""Interpolate onto ``k``, continuing as a power law outside the grid.

        ``jnp.interp`` clamps at the edges, and a clamped linear spectrum is
        *flat* above the last mode instead of falling as
        :math:`k^{-3}\ln^2 k`.  That is not hypothetical: it is what the
        CosmoPower backend in ``ggah_mod`` did above 14.6 h/Mpc while
        :math:`\sigma(M)` quadratured out to 200, silently and with no test.
        Here the grid reaches 200 h/Mpc so the tail is a safety net rather than
        a load-bearing extrapolation -- but it is a net, not a cliff.
        """
        lnq = jnp.log(jnp.asarray(k))
        inside = jnp.interp(lnq, self.lnk, lnp)
        s_hi = (lnp[-1] - lnp[-2]) / (self.lnk[-1] - self.lnk[-2])
        s_lo = (lnp[1] - lnp[0]) / (self.lnk[1] - self.lnk[0])
        hi = lnp[-1] + s_hi * (lnq - self.lnk[-1])
        lo = lnp[0] + s_lo * (lnq - self.lnk[0])
        out = jnp.where(lnq > self.lnk[-1], hi,
                        jnp.where(lnq < self.lnk[0], lo, inside))
        return jnp.exp(out)

    # -- public -------------------------------------------------------------
    def predict(self, k, z, params, which: str = "m"):
        """:math:`P(k,z)` in (Mpc/h)^3, ``params`` in :data:`emu_pk.box.PARAMS` order."""
        self._validate(params, z)
        # Whether the caller asked for one redshift or many is a property of the
        # argument, not of the array it becomes: `jnp.asarray([0.5])` and
        # `jnp.asarray(0.5)` differ, but only before the conversion.
        scalar = np.ndim(z) == 0
        z_arr = jnp.atleast_1d(jnp.asarray(z, dtype=float))
        out = jax.vmap(
            lambda zz: self._interp_lnk(self._forward(params, zz, which), k))(z_arr)
        return out[0] if scalar else out

    def pk(self, k, z, params):
        r""":math:`P_m(k,z)` -- the total matter spectrum."""
        return self.predict(k, z, params, "m")

    def pk_cb(self, k, z, params):
        r""":math:`P_{cb}(k,z)` -- the cold field haloes form from."""
        return self.predict(k, z, params, "cb")
