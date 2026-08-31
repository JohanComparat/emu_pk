# emu_pk

Differentiable emulation of the **linear matter power spectrum**, over an
eight-parameter cosmology that includes the summed neutrino mass and CPL dark
energy, out to $k = 200\ h\,\mathrm{Mpc}^{-1}$ and $z = 5$.

It reproduces CLASS's shape to a **median 0.111 %**, and because it is written
in JAX its derivatives with respect to cosmological parameters come from
automatic differentiation rather than finite differences. Two of those
derivatives are *exact*: the primordial power law is divided out of the training
target and restored in closed form, so a Fisher matrix built on this network is
exactly right in two of its eight directions.

```python
import numpy as np
from emu_pk import PkEmulator

emu = PkEmulator()
k = np.logspace(-3, 1, 200)
theta = np.array([0.02237, 0.1200, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])
pk = emu.pk(k, z=0.5, params=theta)
```

## Why this one

Most linear-$P(k)$ emulators are trained on a narrower box and validated on
*values*. `emu_pk` differs in three ways that matter for a forecast:

- **The box carries `sum_mnu`, `w0` and `wa`**, and is wider than CosmoPower's
  in every axis they share. An emulator without those parameters returns
  $\partial P/\partial w_0 = 0$ — not a small response, an absent one, which in
  a Fisher matrix is a flat direction.
- **The derivatives are validated, not assumed.** An emulator can reproduce
  $P(k)$ to a tenth of a percent and still get $\partial\ln P/\partial\theta$
  wrong, because the error surface is smooth in $k$ and rough in $\theta$.
  `emu_pk.validate` measures autodiff against central differences of CLASS,
  per parameter and per redshift, and reports the finite-difference floor of
  its own comparison.
- **It reaches $k = 200\ h\,\mathrm{Mpc}^{-1}$**, which is what a halo-model
  $\sigma(M)$ integral actually needs.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Tutorial

tutorial/01_spectrum
tutorial/02_accuracy
tutorial/03_derivatives
tutorial/04_the_box
tutorial/05_correction
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
design_notes
reproducing
```

## Citing

Please cite `emu_pk` via its `CITATION.cff`, and also
**CosmoPower** (Spurio Mancini et al. 2022, MNRAS 511, 1771), whose network
architecture and learned activation this reproduces, and **CLASS**
(Blas, Lesgourgues & Tram 2011, JCAP 07, 034), which produced every training
spectrum and every validation reference.
