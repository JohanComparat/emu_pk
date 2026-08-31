# emu_pk

Differentiable emulation of the **linear matter power spectrum**, over an
eight-parameter cosmology that includes the summed neutrino mass and CPL dark
energy, out to $k = 200\ h\,\mathrm{Mpc}^{-1}$ and $z = 5$.

It is written in JAX, so derivatives with respect to cosmological parameters
come from automatic differentiation rather than finite differences — and two of
them are *exact*, because the primordial power law is divided out of the
training target and restored in closed form.

```python
import numpy as np
from emu_pk import PkEmulator

emu = PkEmulator()
k = np.logspace(-3, 1, 200)                      # h/Mpc
#                 omega_b  omega_cdm    h      n_s  ln10A_s  sum_mnu   w0   wa
theta = np.array([0.02237,    0.1200, 0.6736, 0.9649,  3.044,    0.06, -1.0, 0.0])

pk = emu.pk(k, z=0.5, params=theta)              # P_m(k, z) in (Mpc/h)^3
pk_cb = emu.pk_cb(k, z=0.5, params=theta)        # cdm + baryons, without neutrinos
```

## Accuracy

Against held-out CLASS solves — a Latin hypercube on a different seed from the
training design, so no scored point was trained on. The full record is
[`emu_pk/data/validation.json`](emu_pk/data/validation.json), written by
`python -m emu_pk.validate` and not typed by hand.

Three quantities are scored, and together they describe $P(k)$: its amplitude
at $k = 0.05\ h\,\mathrm{Mpc}^{-1}$, its shape once both spectra are
renormalised there, and the two combined with nothing removed. Medians are over
the held-out cosmologies, at $z = 0$.

| at $z = 0$ | median | 90th | max |
|---|---|---|---|
| amplitude at $k = 0.05$ | 0.012 % | 0.035 % | 0.059 % |
| shape, renormalised | **0.111 %** | 0.224 % | 0.621 % |
| **total, absolute** | **0.112 %** | 0.221 % | 0.603 % |

**0.112 %** is the single number for $P(k)$. CosmoPower's released
linear-matter model reaches 0.159 % on the shape measure, on a box narrower in
four of the five axes the two share and equal on the fifth. Shape error is the
largest fractional departure from CLASS over
$k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$; the amplitude is the factor that
renormalisation divides out, and it holds between 0.005 % and 0.012 % across
the whole redshift range.

Derivatives are what a Fisher forecast actually consumes, and an emulator can
reproduce $P(k)$ to a tenth of a percent and still get
$\partial\ln P/\partial\theta$ wrong. Against central differences of CLASS, at
$z = 0$:

| parameter | error | | parameter | error |
|---|---|---|---|---|
| `ln10A_s` | **exact** | | `sum_mnu` | 0.18 % |
| `n_s` | **exact** | | `w0` | 0.16 % |
| `omega_cdm` | 0.06 % | | `omega_b` | 0.20 % |
| `h` | 0.12 % | | `wa` | 0.41 % |

And with respect to redshift — the derivative $f\sigma_8$ is built from:

| | z = 0 | z = 0.5 | z = 1 | z = 2 |
|---|---|---|---|---|
| $\partial\ln P/\partial z$ | 0.155 % | 0.015 % | 0.012 % | 0.008 % |
| *the measurement's own floor* | *0.049 %* | *0.011 %* | *0.006 %* | *0.005 %* |

Away from $z = 0$ this sits within a factor of about two of what the comparison
itself can resolve, so most of what is quoted there is the ruler rather than the
network. At $z = 0$ the ratio is 3.1: that node is an endpoint in slope, with
nothing on the $z < 0$ side to constrain it.

## The box

Deliberately wider than CosmoPower's `mpk_lin`, and carrying three parameters it
does not have. Outside these bounds the network does not fail, it
*extrapolates* — returning a number that is finite, smooth and unwarranted — so
`PkEmulator` checks the box on every call that can afford to look.

| parameter | CosmoPower | `emu_pk` |
|---|---|---|
| `omega_b` | 0.01875 – 0.02625 | 0.0170 – 0.0280 |
| `omega_cdm` | 0.05 – 0.255 | 0.0500 – 0.3000 |
| `h` | 0.64 – 0.82 | 0.5500 – 0.8500 |
| `n_s` | 0.84 – 1.10 | 0.8400 – 1.1000 |
| `ln10A_s` | 1.61 – 3.91 | 1.6100 – 4.0000 |
| `sum_mnu` [eV] | — | 0.0000 – 0.6000 |
| `w0` | — | −1.5000 – −0.5000 |
| `wa` | — | −1.0000 – 0.6000 |
| $k_{\max}$ [h/Mpc] | 14.56 | **200** |
| $z$ | 0 – 5 | 0 – 5 |

Points with `w0 + wa >= 0` are excluded: CPL dark energy then grows without
bound towards early times and dominates before recombination, which is not a
cosmology anyone means to train on.

## Install

```bash
pip install emu_pk                 # inference: numpy + jax, nothing else
pip install 'emu_pk[gen]'          # + classy, to generate data or validate
pip install 'emu_pk[train]'        # + optax, to train
```

The split is load-bearing. `import emu_pk` in an environment with **no**
`classy` and **no** `optax` must work — that is what lets another package depend
on this one without inheriting a Boltzmann solver or a training stack — and the
test suite asserts it.

`[gen]` compiles CLASS from source and needs a C compiler.

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
from source.

## What is here

| Module | Needs | What it does |
| --- | --- | --- |
| `emu_pk.model` | numpy, jax | evaluate the network, in pure JAX |
| `emu_pk.ratio` | numpy, jax | the CLASS-distilled massive-ν and CPL correction |
| `emu_pk.box`, `.grid`, `.cosmo`, `.interp` | numpy, jax | the hypercube, the grids, the conventions, the interpolation |
| `emu_pk.generate`, `.assemble` | `[gen]` | run CLASS; shards → table or training set |
| `emu_pk.train` | `[train]` | fit the network |
| `emu_pk.validate` | `[gen]` | shape error *and* derivative error against CLASS |

`emu_pk.ratio` is a correction measured from CLASS on a grid in
$(\Sigma m_\nu, w_0, w_a, z, k)$ and applied multiplicatively to a
massless-ΛCDM spectrum. It is exactly 1 at the ΛCDM massless corner, which is
what lets it be applied unconditionally — a Python branch on the neutrino mass
would be a branch on a tracer and would break the gradient. It exists so that an
emulator *without* neutrinos or dark energy can be given both. `emu_pk`'s own
network is trained on massive-neutrino w0waCDM spectra directly and needs no
correction.

## Conventions

$k$ in $h\,\mathrm{Mpc}^{-1}$ and $P$ in $(h^{-1}\mathrm{Mpc})^3$ throughout.
CLASS's $1/\mathrm{Mpc}$ is converted once, in `generate`, so nothing
downstream carries an $h$.

$\Omega_m$ **contains the neutrinos**, and
$\Omega_\nu = \Sigma m_\nu/(93.14 h^2)$.

## Validating it yourself

With `[gen]` installed, the numbers above are one command:

```bash
python -m emu_pk.validate --json my_validation.json
```

It scores shape error and derivative error against CLASS across the redshift
range, reports the finite-difference floor of its own comparison, and splits the
box edge and the extreme-quintessence corner out from the interior.

## Tests

```bash
pip install 'emu_pk[dev]'
python -m pytest tests/ -q
```

225 tests. Every statement in `emu_pk` is executed by the suite and 99 % of
its branches. The suite runs without `classy` or `optax` — CLASS is replaced by
a stub wherever a test needs a spectrum rather than a *correct* spectrum — and
the tests that compare conventions against the consuming package skip when it
is not importable.

## Documentation

Full documentation, including a tutorial with figures, is at
[emu-pk.readthedocs.io](https://emu-pk.readthedocs.io). Decisions that would
look arbitrary from the source alone are in
[`docs/design_notes.md`](docs/design_notes.md).

## Citing

If you use `emu_pk`, please cite it via [`CITATION.cff`](CITATION.cff), and
also:

- **CosmoPower** — Spurio Mancini et al. (2022), MNRAS 511, 1771. The network
  architecture and the learned activation are theirs.
- **CLASS** — Blas, Lesgourgues & Tram (2011), JCAP 07, 034. Every training
  spectrum and every validation reference is a CLASS solve.

## Licence

BSD 3-Clause. See [`LICENSE`](LICENSE).
