# The spectrum

`emu_pk` returns the linear matter power spectrum for a cosmology and a
redshift. The parameter vector is in `emu_pk.box.PARAMS` order:

```python
import numpy as np
from emu_pk import PkEmulator, box

print(box.PARAMS)
# ('omega_b', 'omega_cdm', 'h', 'n_s', 'ln10A_s', 'sum_mnu', 'w0', 'wa')

emu = PkEmulator()
k = np.logspace(-4, 2, 400)                      # h/Mpc
theta = np.array([0.02237, 0.1200, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0])

pk = emu.pk(k, z=0.0, params=theta)
```

![The spectrum, and its residual against CLASS](../_static/figures/01_spectrum.png)

*Left:* $P_m(k,z)$ at the Planck 2018 cosmology across the trained redshift
range. *Right:* the same five curves, as a fractional residual against a CLASS
solve of the same cosmology. The shaded band is the range the accuracy claims
are scored over, $k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$.

Two things to read off it. The residual is well under a tenth of a percent
across six decades of $k$ and the whole redshift range, and its structure is
concentrated at the acoustic scale, $k \approx 0.1$–$0.5\
h\,\mathrm{Mpc}^{-1}$, where the spectrum has the most features per decade and
where any emulator works hardest.

```{note}
This is the **fiducial** cosmology, which sits near the middle of the training
box where the emulator is at its best. It is not a held-out average, and the
residual here is smaller than the accuracy you should assume. For that, see
{doc}`02_accuracy`, which scores a held-out design and reports a median of
0.111 %.
```

## Two spectra

```python
pk_m = emu.pk(k, 0.0, theta)          # cdm + baryons + massive neutrinos
pk_cb = emu.pk_cb(k, 0.0, theta)      # cdm + baryons only
```

`cb` is **c**old dark matter **and b**aryons. Neutrinos do not fall into haloes
below their free-streaming scale, so the field that collapses is the cold one
and $\sigma(M)$ is better built from $P_{cb}$; lensing and anything responding
to all the mass wants $P_m$. See {doc}`../quickstart` for the relation between
them.

They come from **one network with two output heads**, which is what stops them
drifting apart in a way that would show up downstream as a spurious
cold-versus-total effect. With $\Sigma m_\nu = 0$ they are identical, because
with no massive species the cold field and the total field are the same field.

## Many redshifts

Redshift is a network input, not a separate model, so asking for several costs
one vectorised evaluation:

```python
z = np.linspace(0, 5, 21)
pk = emu.pk(k, z, theta)              # (21, 400)
```

## Above the grid

The training grid reaches $k = 200\ h\,\mathrm{Mpc}^{-1}$, which is what a
halo-model $\sigma(M)$ integral needs. Above it the emulator continues as a
power law rather than clamping — `jnp.interp` would hold the last value, and a
clamped linear spectrum is *flat* where it should be falling as
$k^{-3}\ln^2 k$. That is a safety net here rather than a load-bearing
extrapolation, but it is a net and not a cliff.
