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
everything downstream rather than repeated — a silently permuted column is the
kind of error that trains perfectly well and predicts nonsense.

## Several redshifts at once

```python
z = np.array([0.0, 0.5, 1.0, 2.0])
pk = emu.pk(k, z, theta)          # shape (4, 400)
```

## The cold field

Haloes form from the cold field rather than the total one, so both are
available and both come from **one network with two heads** — they cannot drift
apart the way two separately trained models would.

```python
pk_cb = emu.pk_cb(k, z=0.0, params=theta)
```

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
