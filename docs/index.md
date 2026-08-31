# emu_pk

Differentiable emulation of the **linear matter power spectrum**, over an
eight-parameter cosmology that includes the summed neutrino mass and CPL dark
energy, out to $k = 200\ h\,\mathrm{Mpc}^{-1}$ and $z = 5$.

It reproduces CLASS's shape to a **median 0.111 %** and its amplitude to
**0.012 %**, so the total error on $P(k)$ itself is **0.112 %** — the shape
metric renormalises, and the amplitude it divides out is accurate to an order
of magnitude better than the shape.

Because it is written in JAX, its derivatives with respect to cosmological
parameters come from automatic differentiation rather than finite differences.
Against central differences of CLASS at $z = 0$:

| `omega_cdm` | `h` | `w0` | `sum_mnu` | `omega_b` | `wa` |
|---|---|---|---|---|---|
| 0.06 % | 0.12 % | 0.16 % | 0.18 % | 0.20 % | 0.41 % |

`ln10A_s` and `n_s` are **exact**, to $2\times10^{-14}$ and $6\times10^{-8}$:
the primordial power law is divided out of the training target and restored in
closed form, so those two are analytic rather than fitted, and a Fisher matrix
built on this network is exact in two of its eight directions.

The derivative with respect to redshift, which $f\sigma_8$ is built from:

| z = 0 | z = 0.5 | z = 1 | z = 2 |
|---|---|---|---|
| 0.155 % | 0.015 % | 0.012 % | 0.008 % |

Away from $z = 0$ that is within about a factor of two of what the comparison
itself can resolve; the per-parameter floors are on the
{doc}`accuracy page <tutorial/02_accuracy>`.


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
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
design_notes
reproducing
```

## Citing

Please cite `emu_pk` itself through its
[`CITATION.cff`](https://github.com/JohanComparat/emu_pk/blob/main/CITATION.cff),
and the two works it is built on.

### CosmoPower

The network architecture — four dense layers of 512 with the learned
$[\gamma + (1-\gamma)\sigma(\beta x)]\,x$ activation — is theirs, and so is
the direct-output choice for this quantity. `emu_pk` implements that
architecture independently in JAX, written from the published description
rather than derived from their source, and differs in the training box, the
wavenumber reach and the training target. CosmoPower itself is not a
dependency: `emu_pk` neither imports nor vendors it.

> Spurio Mancini, A., Piras, D., Alsing, J., Joachimi, B. & Hobson, M. P.,
> *CosmoPower: emulating cosmological power spectra for accelerated Bayesian
> inference from next-generation surveys*,
> **MNRAS 511** (2022) 1771–1788.

- Paper: [doi:10.1093/mnras/stac064](https://doi.org/10.1093/mnras/stac064)
  · [arXiv:2106.03846](https://arxiv.org/abs/2106.03846)
  · [ADS](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.1771S)
- Code: [github.com/alessiospuriomancini/cosmopower](https://github.com/alessiospuriomancini/cosmopower)

### CLASS

Every training spectrum and every validation reference is a CLASS solve, and
the correction table in `emu_pk.ratio` is distilled from CLASS directly. Its
authors ask that any use cite at least the *Approximation schemes* paper.

> Blas, D., Lesgourgues, J. & Tram, T.,
> *The Cosmic Linear Anisotropy Solving System (CLASS). Part II: Approximation
> schemes*, **JCAP 07** (2011) 034.

- Paper: [doi:10.1088/1475-7516/2011/07/034](https://doi.org/10.1088/1475-7516/2011/07/034)
  · [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)
  · [ADS](https://ui.adsabs.harvard.edu/abs/2011JCAP...07..034B)
- Code: [github.com/lesgourg/class_public](https://github.com/lesgourg/class_public)
