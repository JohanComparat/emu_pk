# emu_pk

[![PyPI](https://img.shields.io/pypi/v/emu_pk.svg)](https://pypi.org/project/emu_pk/)
[![Python](https://img.shields.io/pypi/pyversions/emu_pk.svg)](https://pypi.org/project/emu_pk/)
[![Tests](https://github.com/JohanComparat/emu_pk/actions/workflows/tests.yml/badge.svg)](https://github.com/JohanComparat/emu_pk/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/emu-pk/badge/?version=stable)](https://emu-pk.readthedocs.io/en/stable/)
[![Licence: BSD-3-Clause](https://img.shields.io/badge/licence-BSD--3--Clause-blue.svg)](LICENSE)

Differentiable emulation of the **linear matter power spectrum**, over an
eleven-parameter cosmology that includes **three separate neutrino masses**,
CPL dark energy and **spatial curvature**, out to
$k = 300\ h\,\mathrm{Mpc}^{-1}$ and $z = 5$.

It is trained on CAMB at high precision with CLASS's reionization heating, and
reproduces a *converged* CAMB reference to a median 0.063 %. It is written in
JAX, so derivatives with respect to the cosmological parameters come from
automatic differentiation.

```python
import numpy as np
from emu_pk import PkEmulator

emu = PkEmulator()
k = np.logspace(-3, 1, 200)                      # h/Mpc
theta = np.array([
    0.02237,     # omega_b
    0.1200,      # omega_cdm
    0.6736,      # h
    0.9649,      # n_s
    3.044,       # ln10A_s
    0.06,        # sum_mnu [eV] -- still the sum, unchanged in meaning
    -1.0,        # w0
    0.0,         # wa
    0.0,         # Omega_k      -- positive is open
    1 / 3,       # nu_r1  \  how that sum is divided over three species,
    1 / 3,       # nu_r2  /  m_i = r_i * sum_mnu, with r_3 = 1 - r_1 - r_2.
])               #            (1/3, 1/3) is the degenerate convention.

pk = emu.pk(k, z=0.5, params=theta)              # P_m(k, z) in (Mpc/h)^3
pk_cb = emu.pk_cb(k, z=0.5, params=theta)        # cdm + baryons, without neutrinos
```

## Accuracy

Two records ship beside the weights. Both score held-out cosmologies on a Latin
hypercube drawn from a different seed from the training design.

### Against a converged reference

[`emu_pk/data/validation_reference.json`](emu_pk/data/validation_reference.json)
scores the network against CAMB at its converged precision rung, times the same
reionization heating: a truth neither 2.0 nor 2.1 was trained on, and so the one
on which they compare. Total error in $P_m$, 32 cosmologies, at $z = 0$:

| against the converged reference | median | 90th | max |
|---|---|---|---|
| $k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$ | **0.063 %** | 0.130 % | 0.190 % |
| tail, $k \in [10, 300]\ h\,\mathrm{Mpc}^{-1}$ | **0.044 %** | 0.069 % | 0.086 % |
| *2.0.1, $k \in [10^{-3}, 10]$* | *0.351 %* | *0.580 %* | *1.358 %* |

The median stays between 0.051 % and 0.063 % from $z = 0$ to 5, and the cold
spectrum $P_{cb}$ scores the same. 2.0 was trained on CLASS at its defaults,
which a precision scan put 0.41 % from converged; its own validation, against
that same default CLASS, could not see it. 2.1 is trained on CAMB at
`ggah_mod`'s precision, 0.039 % from converged.

### Against its training truth

[`emu_pk/data/validation.json`](emu_pk/data/validation.json), written by
`python -m emu_pk.validate`, scores against the truth the network learnt — CAMB
at the training precision, times the heating — and adds the derivatives, the
lowest decade and the floor of each comparison.

Three quantities describe $P(k)$: its amplitude at
$k = 0.05\ h\,\mathrm{Mpc}^{-1}$, its shape once both spectra are
renormalised there, and the two combined. Shape error is the largest fractional
departure from the truth over $k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$.
Medians are over the held-out cosmologies, at $z = 0$.

<!-- NUMBERS-PENDING -->
| at $z = 0$ | median | 90th | max |
|---|---|---|---|
| amplitude at $k = 0.05$ | … | … | … |
| shape, renormalised | **…** | … | … |
| **total, absolute** | **…** | … | … |
| *the metric's own floor* | *…* | *…* | *…* |

The floor row is the metric's own. The network predicts on a 411-node grid and
the comparison asks the solver at 300 other wavenumbers, so the interpolation
between the nodes is scored as network error; that row is a truth spectrum
pushed through the same path.

A Fisher forecast consumes derivatives rather than spectra. Against central
differences of the training truth, at $z = 0$:

<!-- NUMBERS-PENDING -->
| parameter | error | floor | | parameter | error | floor |
|---|---|---|---|---|---|---|
| `ln10A_s` | **exact** | — | | `sum_mnu` | … | … |
| `n_s` | **exact** | — | | `wa` | … | … |
| `omega_cdm` | … | … | | `Omega_k` | … | … |
| `h` | … | … | | `nu_r1` | … | … |
| `w0` | … | … | | `nu_r2` | … | … |
| `omega_b` | … | … | | | | |

The two mass ratios are the weakest axes. Their effect on $P(k)$ is 0.32 % at
$\Sigma m_\nu = 0.10$ eV and under 0.012 % above 0.25 eV, so there is little
signal to fit; an axis the network ignored would score near 100 %.

And with respect to redshift, which $f\sigma_8$ is built from:

<!-- NUMBERS-PENDING -->
| | z = 0 | z = 0.5 | z = 1 | z = 2 |
|---|---|---|---|---|
| $\partial\ln P/\partial z$ | … | … | … | … |
| *the measurement's own floor* | *…* | *…* | *…* | *…* |

## The box

Outside these bounds the network extrapolates, returning a finite and
unwarranted number, so `PkEmulator` checks the box on every call it can.

| parameter | range |
|---|---|
| `omega_b` | 0.0170 – 0.0280 |
| `omega_cdm` | 0.0500 – 0.3000 |
| `h` | 0.5500 – 0.8500 |
| `n_s` | 0.8400 – 1.1000 |
| `ln10A_s` | 1.6100 – 4.0000 |
| `sum_mnu` [eV] | 0.0000 – 0.6000 |
| `w0` | −1.5000 – −0.5000 |
| `wa` | −1.0000 – 0.6000 |
| `Omega_k` | −0.1500 – 0.1500 |
| `nu_r1` | 0.0000 – 0.3333 |
| `nu_r2` | 0.0000 – 0.5000 |
| $k$ [h/Mpc] | $10^{-4}$ – **300** |
| $z$ | 0 – 5 |

Points with `w0 + wa >= 0` are excluded. CPL dark energy then grows without
bound towards early times and dominates before recombination.

The curvature bound is measured. CLASS solves the whole box at
$|\Omega_k| \le 0.15$; on the closed side it begins refusing at
$\Omega_k = -0.275$, at `omega_cdm = 0.05`, `h = 0.85`. On the open side it
never refuses, including where the closure
$\Omega_{\rm de} = 1 - \Omega_k - \Omega_m - \Omega_r$ turns negative, so
that bound is stated rather than discovered.

Negative $\Omega_{\rm de}$ is kept. It covers 0.4 % of the flat box and 0.9 %
of this one, the solvers handle it, and the spectrum is smooth in the parameters;
`validate` reports it as a stratum.

The neutrino ratios divide $\Sigma m_\nu$ over three species,
$m_i = r_i \Sigma m_\nu$ with $r_3 = 1 - r_1 - r_2$, on the ordered simplex
$0 \le r_1 \le r_2 \le (1-r_1)/2$. A Boltzmann solver sums the species and
cannot distinguish them, so the six permutations of a mass vector are one cosmology
and the ordering removes five of them. Both mass orderings and the degenerate
limit lie inside; the degenerate point $(1/3, 1/3)$ is a vertex, and is scored
as its own stratum.

Against CLASS, the degenerate approximation is wrong by 0.32 % in $P(k)$ at
$\Sigma m_\nu = 0.10$ eV in an inverted ordering, and by under 0.012 % above
0.25 eV. This emulator's own error, against the converged reference, is 0.063 %.

Curvature costs the fit little above $k \approx 10^{-2}$, where its effect is
a $k$-independent growth rescaling flat in $k$ to 0.5 %. Below
$k \approx 10^{-3}$ it does not: the curvature scale $\sqrt{|\Omega_k|}H_0/c$
is $1.3\times10^{-4}\ h\,\mathrm{Mpc}^{-1}$ at the edge of the box, inside
the $k$ grid. That decade is scored separately; see
[`docs/design_notes.md`](docs/design_notes.md).

## Install

```bash
pip install emu_pk                 # inference: numpy + jax, nothing else
pip install 'emu_pk[gen]'          # + CAMB and classy, to generate data or validate
pip install 'emu_pk[train]'        # + optax, to train
```

`import emu_pk` in an environment with no `classy` and no `optax` must work,
so that a package depending on this one does not inherit a Boltzmann solver or
a training stack. The test suite asserts it.

`[gen]` compiles CLASS from source and needs a C compiler. The shipped weights
were trained and validated on **CAMB 1.6.6**: CAMB 2.0.4 differs from it by up
to 4.4e-4 in $\ln P$ near $k = 0.1\ h\,\mathrm{Mpc}^{-1}$, a visible fraction
of the network's own error, so pin it (`pip install camb==1.6.6`) to reproduce
the validation.

### A dedicated environment

[`environment.yml`](environment.yml) is a minimal conda environment — python,
numpy and JAX, and nothing else:

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
mamba activate emu_pk
pip install -e .
```

It pins the **CPU** build of `jaxlib`: 64 MB against 199 MB for the CUDA one,
and left unpinned the build depends on whether the machine that solved the
environment happened to have a driver. The file says how to swap it for a GPU.

The extras are commented blocks in the same file — uncomment the one you need.
`classy` is not on conda-forge, so `[gen]` comes from pip and compiles CLASS
from source; CAMB comes from pip as a wheel.

## What is here

| Module | Needs | What it does |
| --- | --- | --- |
| `emu_pk.model` | numpy, jax | evaluate the network, in pure JAX |
| `emu_pk.ratio` | numpy, jax | the CLASS-distilled massive-ν and CPL correction |
| `emu_pk.box`, `.grid`, `.cosmo`, `.interp` | numpy, jax | the hypercube, the grids, the conventions, the interpolation |
| `emu_pk.generate`, `.assemble` | `[gen]` | run CAMB with CLASS's heating; shards → table or training set |
| `emu_pk.heating` | `[gen]` | the CLASS pair that carries reionization heating |
| `emu_pk.train` | `[train]` | fit the network, on a CPU or a GPU |
| `emu_pk.validate` | `[gen]` | shape error *and* derivative error, against the training truth or a converged reference |

`emu_pk.ratio` is a correction measured from CLASS on a grid in
$(\Sigma m_\nu, w_0, w_a, z, k)$ and applied multiplicatively to a
massless-ΛCDM spectrum. It is exactly 1 at the ΛCDM massless corner, so it applies unconditionally;
a Python branch on the neutrino mass would branch on a tracer and break the
gradient. It gives neutrinos and dark energy to an emulator that has neither.
This package's own network is trained on massive-neutrino w0waCDM spectra and
needs no correction.

## Conventions

$k$ in $h\,\mathrm{Mpc}^{-1}$ and $P$ in $(h^{-1}\mathrm{Mpc})^3$ throughout.
The solvers' units are converted once, in `generate` and `heating`, so nothing
downstream carries an $h$.

$\Omega_m$ **contains the neutrinos**, and
$\Omega_\nu = \Sigma m_\nu/(93.143 h^2)$, the rest mass of three states at
CLASS's default temperature.

## Validating it yourself

With `[gen]` installed (and CAMB 1.6.6, above), the numbers above are two
commands:

```bash
python -m emu_pk.validate --json my_validation.json               # training truth
python -m emu_pk.validate --truth reference --json my_reference.json
```

The first scores shape error and derivative error against the training truth
across the redshift range, reports the floor of its own comparison, and splits
the box edge and the extreme-quintessence corner out from the interior: about
840 solves. The second is the converged reference, half an hour of one core a
point. `--cache DIR` keeps the solves, and a cache filled on one machine
serves another.

## Tests

```bash
pip install 'emu_pk[dev]'
python -m pytest tests/ -q
```

357 tests. <!-- NUMBERS-PENDING: coverage --> Every statement in `emu_pk` is executed by the suite and 99 % of
its branches. `[dev]` installs everything the suite needs, including `optax`,
since the tests that exercise the trainer import it. Neither solver is needed: each is
replaced by a stub wherever a test wants a spectrum rather than a *correct*
one, and the few tests that do want a real solver skip without it. The tests
that compare conventions against the consuming package skip when it is not
importable.

## Documentation

Full documentation, including a tutorial with figures, is at
[emu-pk.readthedocs.io](https://emu-pk.readthedocs.io/en/stable/). Decisions that would
look arbitrary from the source alone are in
[`docs/design_notes.md`](docs/design_notes.md).

## Citing

If you use `emu_pk`, please cite it via [`CITATION.cff`](CITATION.cff), and
also:

- **CosmoPower** — Spurio Mancini et al. (2022), MNRAS 511, 1771. The network
  architecture and the learned activation are theirs.
- **CAMB** — Lewis, Challinor & Lasenby (2000), ApJ 538, 473. Since 2.1 every
  training spectrum and every validation truth is a CAMB solve.
- **CLASS** — Blas, Lesgourgues & Tram (2011), JCAP 07, 034. Every training
  spectrum carries CLASS's reionization heating, and the correction table in
  `emu_pk.ratio` is distilled from CLASS.

## Licence

BSD 3-Clause. See [`LICENSE`](LICENSE).
