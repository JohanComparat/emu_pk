# Validation

This page gathers the tests of the shipped network that a user of its spectra or its derivatives needs: its accuracy, the null test of its two heads at zero neutrino mass, and its speed.
Each section gives the numbers and links to the study that measured them; the scripts and raw results are in the [`studies/`](https://github.com/JohanComparat/emu_pk/tree/main/studies) directory of the repository.
All numbers refer to the network shipped since 2.1.0 (SHA-256 in [Provenance](provenance.md)).

## Accuracy of the spectrum

`emu_pk.validate` scores the network on 32 test cosmologies, against its own training truth and against a converged CAMB reference; [Provenance](provenance.md) gives the commands and the checksums of the two records.

| truth | median | largest |
|---|---|---|
| training truth (`validation.json`) | 0.064 % | 0.177 % |
| converged CAMB (`validation_reference.json`) | 0.063 % | 0.19 % |

The statistic is the maximum over $10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}$ of the shape error of $P_{\rm m}$ at $z = 0$, and the median and largest are over the 32 cosmologies.
The [accuracy tutorial](tutorial/02_accuracy.md) carries the derivatives, per parameter and per redshift, and the floor of each comparison.

## Null test at zero neutrino mass

At $\Sigma m_\nu = 0$ the cold and total matter fields coincide, so $P_{\rm cb}/P_{\rm m} = 1$ exactly; the network predicts the two with separate heads and is not told so.
We measure $\ln(P_{\rm cb}/P_{\rm m})$ and its Jacobian on 1 000 cosmologies spanning the box with `sum_mnu` set to 0 ([study](https://github.com/JohanComparat/emu_pk/tree/main/studies/null_mnu)).

| statistic of $\max_k \lvert\ln(P_{\rm cb}/P_{\rm m})\rvert$ | value |
|---|---|
| median, $z = 0$ | 3.6e-4 |
| 95th percentile, $z = 0$ | 9.9e-4 |
| largest, $z = 0$ | 1.6e-3 |
| median, $z = 5$ | 9.2e-4 |
| largest, all redshifts | 3.4e-3 |

The mean signed value at $z = 0$ is $+2.0 \times 10^{-4}$: $P_{\rm cb}$ comes out slightly high.
The violation is below the network's own error on either head.

The derivatives of the ratio are exactly zero in `n_s` and `ln10A_s`, which both heads carry analytically.
Along the other cosmological directions they are $10^{-4}$ to $2 \times 10^{-3}$ of the derivative of $\ln P_{\rm m}$.
The neutrino mass split `nu_r1`, `nu_r2` has no physical effect at zero mass, yet the network responds to it at about $10^{-3}$; this response is spurious.
The derivative in `sum_mnu` is physical: 0.108 per eV from the network, against $2 f_\nu/\Sigma m_\nu = 0.112\ {\rm eV}^{-1}$ on small scales.

A head predicting $\ln(P_{\rm cb}/P_{\rm m}) = \Sigma m_\nu\, g(\theta, k, z)$ would make the null exact by construction.
The shipped network does not have it; it is planned for the next retraining.

## Timings

We time the compiled functions with `jax.block_until_ready`, on a laptop with an Intel Core Ultra 9 285HX and an NVIDIA RTX PRO 4000 Blackwell Laptop GPU, for 200 wavenumbers and one cosmology ([study](https://github.com/JohanComparat/emu_pk/tree/main/studies/timing)).
The Jacobian is $\partial P_{\rm m}(k, z)/\partial\theta$ over the eleven parameters of the box at the 31 training redshifts.

| | CPU | GPU |
|---|---|---|
| cold start, `import emu_pk` to the first spectrum | 5.1 s | 8.5 s |
| compilation | 0.3–0.8 s | 1–14 s |
| one spectrum, compiled | 0.36 ms | 0.30 ms |
| Jacobian, `jax.jacfwd` | 11 ms, 1.9 MB | 0.14 ms, 3.8 MB |
| Jacobian, `jax.jacrev` | 4.2 s, 1.4 GB | 48 ms, 1.5 GB |
| gradient of a scalar, `jax.grad` | 2.9 ms | 0.12 ms |

Memory is XLA's scratch space beyond inputs and outputs.
Use `jax.jacfwd` for the Jacobian: eleven inputs against up to 6 200 outputs make forward mode eleven passes.
Use reverse mode only for the gradient of a scalar such as a likelihood.
On a GPU shared with other processes, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`: JAX otherwise reserves 75 % of the memory at start-up and the larger programs fail with `RESOURCE_EXHAUSTED`.
CPU times vary by up to a factor of three between runs on this laptop, which throttles under load; take them as orders of magnitude.
