# The box, and its edges

A neural emulator is valid inside the box it was trained on and nowhere else.
Outside it the network does not fail — it *extrapolates*, returning a number
that is finite, smooth and unwarranted. So the box is data, checked on every
call that can afford to look, rather than a sentence in a docstring.

The bounds are closed and inclusive, and these are the values themselves:

| parameter | `emu_pk` | CosmoPower `mpk_lin` |
|---|---|---|
| `omega_b` | 0.0170 … 0.0280 | 0.01875 … 0.02625 |
| `omega_cdm` | 0.0500 … 0.3000 | 0.05 … 0.255 |
| `h` | 0.5500 … 0.8500 | 0.64 … 0.82 |
| `n_s` | 0.8400 … 1.1000 | 0.84 … 1.10 |
| `ln10A_s` | 1.6100 … 4.0000 | 1.61 … 3.91 |
| `sum_mnu` [eV] | 0.0000 … 0.6000 | *not a parameter* |
| `w0` | −1.5000 … −0.5000 | *not a parameter* |
| `wa` | −1.0000 … 0.6000 | *not a parameter* |
| $k$ [$h\,\mathrm{Mpc}^{-1}$] | $10^{-4}$ … 200 | up to 14.56 |
| $z$ | 0 … 5 | 0 … 5 |

The `emu_pk` column is `box.BOX`, and the wavenumber and redshift rows are
`grid.K_MIN`/`grid.K_MAX` and `grid.Z_MIN`/`grid.Z_MAX`. Of the five axes the
two share, `emu_pk` is strictly wider in four and matches on `n_s`. It carries
three parameters CosmoPower's `mpk_lin` leaves fixed, and reaches
200 $h\,\mathrm{Mpc}^{-1}$ against 14.56.

Points satisfying `w0 + wa < 0` are the sampled region within those bounds;
the reason is below.

```python
from emu_pk import box

box.PARAMS      # the order everything downstream reads
box.BOX         # {name: (low, high)}, closed and inclusive
```

## Checking a point

```python
import numpy as np
from emu_pk import box

theta = np.array([0.02237, 0.1200, 0.6736, 0.9649, 3.044, 0.06, -0.2, 0.0])

box.inside(theta)
# {'w0': (-0.2, (-1.5, -0.5))}      empty when the point is inside

box.check(theta)
# ValueError: outside the emulator training box, where the network
# extrapolates with no accuracy guarantee: w0 = -0.2 not in [-1.5, -0.5].
```

`PkEmulator` calls `check` for you on every call where the values are concrete.
Under `jax.jit` the values are tracers, so the check is *skipped* rather than
attempted — see {doc}`../design_notes`.

## Drawing a design

```python
design = box.sample(1000, seed=12345)     # (1000, 8), Latin hypercube
```

Deterministic in the seed: a design is reproducible from the seed alone, so a
shard can be regenerated years later without shipping the design matrix, and
two workers can never disagree about which cosmology index *i* means.

Points with `w0 + wa >= 0` are rejected and redrawn. That is not a taste
constraint: with `w0 + wa >= 0` the CPL dark-energy density grows without bound
towards early times, dark energy dominates before recombination, and CLASS
either refuses or returns a spectrum that is not a cosmology anyone means to
train on.

## Where it is thinnest

Two places, and both are reported separately by `emu_pk.validate`:

**The walls.** An eight-dimensional Latin hypercube essentially never samples a
corner. The design's points are stratified, so they cover each axis evenly, but
a point near the wall *in every axis at once* does not occur. A sampler with
wide priors will visit places the training set did not.

**The extreme-quintessence corner.** With `w0` near $-0.5$ and `wa` positive,
`w(a)` climbs toward zero at early times. CLASS refused about 0.02 % of the
training solves there, all of them in that one corner, so the training set has
a small hole exactly where a forecast is most likely to wander. The generator
records failures rather than filling them, because a set with silent gaps
trains perfectly well and is wrong in a place nothing points at.
