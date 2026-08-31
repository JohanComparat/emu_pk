r"""The grids, defined once and shared by generation, training and inference.

A training set and the model that reads it must agree on the wavenumber grid to
the last node.  Writing the grid down in one module, imported by the generator
*and* by the predictor, is what makes that agreement structural rather than a
convention two files happen to share.

Wavenumbers are in h/Mpc and spectra in (Mpc/h)^3 throughout, matching
``ggah_mod``'s convention rather than CLASS's 1/Mpc -- the conversion happens
once, in :mod:`emu_pk.generate`, so nothing downstream carries an ``h``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["K_MIN", "K_MAX", "N_K", "k_grid", "lnk_grid",
           "Z_MIN", "Z_MAX", "Z_NODES_RATIO", "Z_NODES_EMU",
           "MNU_NODES", "W0_NODES", "WA_NODES"]

# --------------------------------------------------------------------------
# Wavenumbers
# --------------------------------------------------------------------------
#: h/Mpc.  The top of the range is what the consumer needs.  ``ggah_mod``'s
#: FAST backend quadratures sigma(M) out to k = 200 h/Mpc.  An emulator that stops
#: short of that leaves ``jnp.interp`` clamping silently above its last mode,
#: which returns a flat tail where the spectrum should be falling and puts
#: P(200) orders of magnitude high with nothing raising.  Generating to 200
#: means the emulator covers what its consumer integrates.
K_MIN = 1e-4
K_MAX = 200.0
N_K = 400


def k_grid(n_k: int = N_K) -> np.ndarray:
    """Log-spaced wavenumbers in h/Mpc."""
    return np.logspace(np.log10(K_MIN), np.log10(K_MAX), n_k)


def lnk_grid(n_k: int = N_K) -> np.ndarray:
    """``log`` of :func:`k_grid` -- the axis everything interpolates on."""
    return np.log(k_grid(n_k))


# --------------------------------------------------------------------------
# Redshift
# --------------------------------------------------------------------------
Z_MIN, Z_MAX = 0.0, 5.0

#: Redshift nodes of the correction table.  A CLASS call returns every redshift
#: it is asked for from one solve, so a dense axis here is nearly free.
#:
#: The round numbers stay on nodes deliberately.  They are what a user evaluates
#: at, and a node is where interpolation is *exact*; the Hermite construction is
#: what makes the derivative continuous there as well.
Z_NODES_RATIO = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.7, 1.0, 1.25,
                          1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

#: Redshifts written per training-set cosmology.  ``z`` is a network *input*,
#: so these are rows rather than an interpolation axis, and all of them cost
#: one CLASS solve between them.
#:
#: Twenty uniform nodes over [0, 5] put the first interior node at z = 0.263,
#: which leaves the slope at z = 0 unpinned: z = 0 is a node in *value* and an
#: endpoint in *slope*, nothing on the z < 0 side constrains it, and
#: ``f sigma_8`` is built from that derivative and quoted at low redshift.  The
#: eleven log-spaced nodes below z = 0.5 densify exactly that region.
#:
#: The tail starts at 5e-3 and its nodes are the same for every cosmology.
#: Both were measured rather than assumed: starting it at 1e-4 instead changes
#: ``dlnP/dz(0)`` by less than a third of the run-to-run scatter, and jittering
#: the nodes per cosmology changes it by less again.
#:
#: **Regenerating against this needs a fresh shard directory.**
#: ``generate.emu_shard`` skips a chunk whose output exists, so 20-node shards
#: would be kept and silently mixed; ``assemble`` refuses shards whose z axes
#: disagree, which turns that into an error rather than a bad training set, but
#: the clean move is a new directory.
Z_NODES_EMU = np.unique(np.concatenate([
    np.linspace(Z_MIN, Z_MAX, 20),
    np.logspace(np.log10(5e-3), np.log10(0.5), 11),
]))


# --------------------------------------------------------------------------
# Correction-table axes
# --------------------------------------------------------------------------
#: Sum of neutrino masses [eV].  Twelve nodes to 0.6 eV, against ten to 0.5.
#: Zero is a node and must stay one: it is where both ratios are *exactly* 1,
#: which is what lets the correction be applied unconditionally with no Python
#: branch on a traced value.
MNU_NODES = np.array([0.0, 0.02, 0.04, 0.06, 0.09, 0.12,
                      0.18, 0.25, 0.32, 0.40, 0.50, 0.60])

#: CPL dark energy.  Bounds match ``ggah_mod_benchmark``'s case grid, so the
#: benchmark can exercise the whole box rather than its middle.  ``w0 = -1`` and
#: ``wa = 0`` are nodes: LambdaCDM is where the ratio must reduce to the
#: neutrino-only answer, and a node is where that is exact.
W0_NODES = np.array([-1.30, -1.15, -1.00, -0.85, -0.70])
WA_NODES = np.array([-0.70, -0.35, 0.00, 0.25, 0.50])
