# Null test of the two heads at zero neutrino mass

At $\Sigma m_\nu = 0$ the cold and total matter fields coincide, so $P_{\rm cb}/P_{\rm m} = 1$ exactly in every solver and in every training row with no massive neutrinos.
The network predicts $\ln P_{\rm m}$ and $\ln P_{\rm cb}$ with two output heads on one trunk and is not told that they coincide there.
`null_test.py` measures how far apart they come out, and how far apart their derivatives come out.

## Set-up

- 1 000 cosmologies from `box.sample(1000, seed=1)` with `sum_mnu` pinned to 0; every other parameter spans the box.
- $\ln(P_{\rm cb}/P_{\rm m})$ on the emulator grid, at the 31 training redshifts; statistics over $10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}$.
- The Jacobian of $\ln(P_{\rm cb}/P_{\rm m})$ and of $\ln P_{\rm m}$ with respect to the eleven parameters, by `jax.jacfwd`, at $z = 0$ and $z = 1$.

```bash
python studies/null_mnu/null_test.py --n 1000 --json studies/null_mnu/null_test.json
```

## Results

The ratio.
At $z = 0$ the median over cosmologies of $\max_k |\ln(P_{\rm cb}/P_{\rm m})|$ is $3.6 \times 10^{-4}$, the 95th percentile $9.9 \times 10^{-4}$ and the largest $1.6 \times 10^{-3}$; the mean signed value is $+2.0 \times 10^{-4}$, so $P_{\rm cb}$ comes out slightly high.
The violation grows with redshift: the median of the maximum is $4.3 \times 10^{-4}$ at $z = 1$ and $9.2 \times 10^{-4}$ at $z = 5$, and the largest over all redshifts is $3.4 \times 10^{-3}$.
For comparison, the network's own shape error on $P_{\rm m}$ is 0.064 % in the median and 0.177 % at most at $z = 0$, so the null violation is of the same size as, and below, the error of either head.

The derivatives, at $z = 0$, as the median of $\max_k$ of each derivative:

| parameter | $\partial \ln(P_{\rm cb}/P_{\rm m})$ | $\partial \ln P_{\rm m}$ | ratio |
|---|---|---|---|
| `omega_b` | 0.021 | 16.8 | 1.1e-3 |
| `omega_cdm` | 0.0033 | 13.1 | 2.6e-4 |
| `h` | 0.0016 | 4.47 | 3.6e-4 |
| `n_s`, `ln10A_s` | 0 | 4.93, 1 | 0 (exact) |
| `w0` | 3.8e-4 | 0.34 | 1.0e-3 |
| `wa` | 2.3e-4 | 0.091 | 2.2e-3 |
| `Omega_k` | 0.0032 | 1.65 | 2.0e-3 |
| `nu_r1` | 1.5e-3 | 2.9e-3 | — |
| `nu_r2` | 6.9e-4 | 1.6e-3 | — |
| `sum_mnu` | 0.108 | 0.23 | physical |

The derivatives in `n_s` and `ln10A_s` are exactly zero, because both heads carry the same analytic primordial term.
Along the other cosmological directions, the derivative of the ratio is $10^{-4}$ to $2 \times 10^{-3}$ of the derivative of $\ln P_{\rm m}$, so a Fisher matrix built on $P_{\rm cb}$ differs from one built on $P_{\rm m}$ at that level at zero mass.
The mass split `nu_r1`, `nu_r2` has no effect at zero mass, so both columns are null tests on their own: the network's response to them, $\sim 10^{-3}$, is spurious, and equally so in both heads.
The derivative in `sum_mnu` is the physical one: $2 f_\nu / \Sigma m_\nu = 2/(93.14\,\omega_{\rm m}) = 0.112\ {\rm eV}^{-1}$ on small scales, against 0.108 from the network.

## A ratio head constrained to one at zero mass

A head predicting $\ln(P_{\rm cb}/P_{\rm m}) = \Sigma m_\nu\, g(\theta, k, z)$, with $g$ the network output, would make the null exact by construction, in value and in every derivative except the one in $\Sigma m_\nu$, as the primordial split does for $A_s$ and $n_s$.
It also needs no new information: the training targets are the two spectra already stored.
We did not retrain the shipped network for it: the violation it removes is below the network's own error, and the change of output form would need a new checkpoint field and a full training run (about an hour on an A100, 240 epochs).
It is the change to make at the next retraining.
