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
| `Omega_k` | −0.1500 … 0.1500 | *not a parameter* |
| `nu_r1` | 0.0000 … 0.3333 | *not a parameter* |
| `nu_r2` | 0.0000 … 0.5000 | *not a parameter* |
| $k$ [$h\,\mathrm{Mpc}^{-1}$] | $10^{-4}$ … 200 | up to 14.56 |
| $z$ | 0 … 5 | 0 … 5 |

![The training box against CosmoPower's](../_static/figures/04_the_box.png)

Each bar is one axis, normalised to the `emu_pk` range so the comparison is a
fraction rather than a unit.

The `emu_pk` column is `box.BOX`, and the wavenumber and redshift rows are
`grid.K_MIN`/`grid.K_MAX` and `grid.Z_MIN`/`grid.Z_MAX`. Of the five axes the
two share, `emu_pk` is strictly wider in four and matches on `n_s`. It carries
six parameters CosmoPower's `mpk_lin` leaves fixed, and reaches
200 $h\,\mathrm{Mpc}^{-1}$ against 14.56.

`Omega_k` is positive for an open universe, CLASS's own sign. `nu_r1` and
`nu_r2` divide the neutrino mass over three species, $m_i = r_i \Sigma m_\nu$
with $r_3 = 1 - r_1 - r_2$; `sum_mnu` is still the sum, and what these two add
is how it is split.

Not every point inside these bounds is sampled: `w0 + wa < 0` and the ordered
simplex $0 \le r_1 \le r_2 \le (1-r_1)/2$ carve out the rest. Both reasons are
below.

```python
from emu_pk import box

box.PARAMS      # the order everything downstream reads
box.BOX         # {name: (low, high)}, closed and inclusive
```

## Checking a point

```python
import numpy as np
from emu_pk import box

theta = np.array([0.02237, 0.1200, 0.6736, 0.9649, 3.044, 0.06, -0.2, 0.0,
                  0.0, 1/3, 1/3])

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
design = box.sample(1000, seed=12345)     # (1000, 11), Latin hypercube
flat = box.sample(1000, seed=12345, pin={"Omega_k": 0.0})
```

Deterministic in the seed: a design is reproducible from the seed alone, so a
shard can be regenerated years later without shipping the design matrix, and
two workers can never disagree about which cosmology index *i* means.

`pin` holds named columns at a fixed value after the draw, which is how the
control arms of a campaign are built. A column pinned during *training*
standardises to a constant, so such a network is meaningful only on the slice
it was pinned to.

Two rejections apply, and neither is a matter of taste.

**`w0 + wa >= 0` is redrawn.** The CPL dark-energy density then grows without
bound towards early times, dark energy dominates before recombination, and
CLASS either refuses or returns a spectrum that is not a cosmology anyone means
to train on.

**Mass ratios outside the ordered simplex are redrawn.** CLASS sums the three
species' contributions and cannot tell them apart, so the six permutations of
one mass vector are the same cosmology; ordering them is what stops the network
spending capacity on an exact symmetry instead of the physics. `nu_r1` cannot
exceed 1/3 under that constraint, which is where its bound comes from.

Curvature is *not* rejected anywhere. $\pm0.15$ is the widest interval over
which CLASS solves the whole box: at the low-density corner it begins refusing
near $\Omega_k = -0.275$, and on the open side it never refuses at all. Open
curvature can drive the closure $\Omega_{\rm de} = 1 - \Omega_m - \Omega_r -
\Omega_k$ negative, and those solve without complaint — a strange universe, not
an ill-posed one. `validate` reports the region as its own stratum rather than
carving it out.

## Where it is thinnest

Three places, each reported separately by `emu_pk.validate`:

**The walls.** An eleven-dimensional Latin hypercube essentially never samples a
corner. The design's points are stratified, so they cover each axis evenly, but
a point near the wall *in every axis at once* does not occur. A sampler with
wide priors will visit places the training set did not.

**The degenerate neutrino point.** $r = (1/3, 1/3)$ is a *vertex* of the
sampled simplex, not an interior point — it is where `nu_r1` meets its bound and
the ordering constraint is tight at once, and it cannot be made interior because
a mass spread is non-negative. It is also where every result published before
version 2 sits, so it gets its own stratum.

**The extreme-quintessence corner.** With `w0` near $-0.5$ and `wa` positive,
`w(a)` climbs toward zero at early times. CLASS refused about 0.02 % of the
training solves there, all of them in that one corner, so the training set has
a small hole exactly where a forecast is most likely to wander. The generator
records failures rather than filling them, because a set with silent gaps
trains perfectly well and is wrong in a place nothing points at.
