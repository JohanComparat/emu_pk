"""Differentiable linear-P(k) emulation for ``ggah_mod``.

Two things live here, in the order they were built:

* :mod:`emu_pk.ratio` -- the CLASS-distilled correction that carries massive
  neutrinos *and* CPL dark energy onto a massless-LambdaCDM spectrum.  It is
  what gives a LambdaCDM-trained emulator a response to ``w0`` and ``wa``.
* :mod:`emu_pk.model` -- a linear-P(k) network trained here, over a box wider
  than any external emulator's, out to the wavenumber its consumer actually
  integrates to.  It subsumes the correction: trained on massive-neutrino
  w0waCDM spectra directly, there is nothing left to correct.

The split between them is the reason both exist: the correction was cheap and
landed first, and it is the validation target the network has to beat.

The **core install is numpy and jax only**.  Generating training data needs
``classy`` (``pip install emu_pk[gen]``) and training needs ``optax``
(``[train]``); neither is required to *use* a shipped model, which is what lets
``ggah_mod`` depend on this package without inheriting a Boltzmann solver.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import box, cosmo, grid, interp, model, ratio
from .model import PkEmulator, primordial_ln_pk

#: The supported surface.  Anything not listed here, and anything not in a
#: module's own ``__all__``, is private and may change without a major version.
__all__ = [
    # the emulator
    "PkEmulator",
    "primordial_ln_pk",
    # the modules, for the things that live in them
    "box",
    "cosmo",
    "grid",
    "interp",
    "model",
    "ratio",
    "__version__",
]
