# Quickstart

## A spectrum

```python
import numpy as np
from emu_pk import PkEmulator

emu = PkEmulator()

k = np.logspace(-4, 2, 400)                      # h/Mpc
theta = np.array([
    0.02237,   # omega_b
    0.1200,    # omega_cdm
    0.6736,    # h
    0.9649,    # n_s
    3.044,     # ln10A_s
    0.06,      # sum_mnu  [eV]
    -1.0,      # w0
    0.0,       # wa
])

pk = emu.pk(k, z=0.0, params=theta)              # (Mpc/h)^3
```

The parameter order is `emu_pk.box.PARAMS`, and it is read from there by
everything downstream rather than repeated.

## Several redshifts at once

```python
z = np.array([0.0, 0.5, 1.0, 2.0])
pk = emu.pk(k, z, theta)          # shape (4, 400)
```

## The cold field: `pk` versus `pk_cb`

The two methods return the power spectrum of two different fields.

| | field | contains |
|---|---|---|
| `emu.pk` | total matter, $\delta_m$ | cold dark matter **+ baryons + massive neutrinos** |
| `emu.pk_cb` | the cold field, $\delta_{cb}$ | cold dark matter **+ baryons** |

So `cb` is **c**old dark matter **and b**aryons — not cold dark matter alone.
"Cold" here is in contrast to the neutrinos, which are light enough to be
relativistic in the early universe and to free-stream out of potential wells
afterwards. Everything that is *not* a neutrino is cold, and baryons are.

```python
pk_m = emu.pk(k, z=0.0, params=theta)        # cdm + baryons + neutrinos
pk_cb = emu.pk_cb(k, z=0.0, params=theta)    # cdm + baryons
```

### Why the distinction exists

Below the neutrino free-streaming scale, neutrinos do not fall into haloes.
The field that actually collapses is the cold one, so quantities built for the
halo model — $\sigma(M)$, the mass function, the halo bias — are more accurately
predicted from $P_{cb}$ than from $P_m$. Lensing and other observables that
respond to *all* the mass want $P_m$. Which one you need depends on what you
are computing, which is why both are here.

### How they relate

With $f_\nu = \Omega_\nu/\Omega_m$ the neutrino mass fraction,

$$\delta_m = (1 - f_\nu)\,\delta_{cb} + f_\nu\,\delta_\nu .$$

On large scales the neutrinos cluster along with everything else,
$\delta_\nu \to \delta_{cb}$, and the two spectra converge. On small scales
they free-stream away, $\delta_\nu \to 0$, and

$$P_m \to (1 - f_\nu)^2\,P_{cb},$$

so **$P_{cb}$ is the larger of the two**. For $\Sigma m_\nu = 0.3$ eV at the
Planck cosmology, $f_\nu = 0.023$ and the emulator gives $P_{cb}/P_m = 1.0000$
at $k = 10^{-4}$ and $1.0456$ at $k = 1\ h\,\mathrm{Mpc}^{-1}$, against the
free-streaming limit $1/(1-f_\nu)^2 = 1.0472$.

With $\Sigma m_\nu = 0$ there are no massive neutrinos, the cold field *is* the
total field, and the two are the same spectrum.

### One network, two heads

Both come from a single network with two output heads rather than two separately
trained models. That is what stops them drifting apart in a way that would show
up downstream as a spurious cold-versus-total effect — a difference between the
two spectra that is a fitting artefact rather than physics.

## Derivatives

This is the point of the package. `emu.pk` is a JAX function, so:

```python
import jax
import jax.numpy as jnp

def ln_pk(t):
    return jnp.log(emu.pk(k, 0.0, t))

jac = jax.jacfwd(ln_pk)(jnp.asarray(theta))       # (400, 8)
```

`jac[:, i]` is $\partial\ln P/\partial\theta_i$ across $k$. For `ln10A_s` it is
exactly 1 and for `n_s` exactly $\ln(kh/k_*)$, because those two are not
learned — see {doc}`design_notes`.

You can differentiate with respect to redshift the same way, which is what
$f\sigma_8$ is built from:

```python
dlnP_dz = jax.jacfwd(lambda s: jnp.log(emu.pk(k, s, theta)))(0.5)
```

## Staying inside the box

Outside its training bounds the network does not fail — it extrapolates,
returning a number that is finite, smooth and unwarranted. `PkEmulator` checks
on every call where the values are concrete:

```python
bad = theta.copy()
bad[6] = -0.2                                     # w0, outside [-1.5, -0.5]
emu.pk(k, 0.0, bad)
# ValueError: outside the emulator training box, where the network
# extrapolates with no accuracy guarantee: w0 = -0.2 not in [-1.5, -0.5].
```

The check is skipped under `jax.jit` tracing, where the values are not
available — attempting it there would raise `ConcretizationTypeError` and break
the gradient the package exists to provide. A jitted forward model is checked
once when it is built. Pass `PkEmulator(check_box=False)` to opt out entirely.
