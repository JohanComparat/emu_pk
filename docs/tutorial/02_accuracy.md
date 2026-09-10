# Accuracy

Every number on this page is written by `python -m emu_pk.validate` into
`emu_pk/data/validation.json`, so it describes the weights that ship beside it.
The scores come from held-out CLASS solves, on a Latin hypercube drawn from a
different seed than the training design. Three quantities are measured: the
amplitude of the spectrum, its shape, and its derivatives.

![Shape error and derivative error](../_static/figures/02_accuracy.png)

*Left:* the shape error across redshift. *Right:* the derivative error at
$z = 0$ for the nine fitted axes, with the comparison's own finite-difference
floor marked. `ln10A_s` and `n_s` are restored in closed form rather than
fitted and are left off; see {doc}`03_derivatives`.

## 1. Amplitude

The amplitude is the value of $P$ at the normalisation scale,
$k = 0.05\ h\,\mathrm{Mpc}^{-1}$, scored over 32 held-out cosmologies. At
$z = 0$ the median is **0.010 %**, the 90th percentile 0.030 % and the maximum
0.062 %. Across the trained redshift range:

| z | 0 | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| amplitude | 0.010 % | 0.012 % | 0.012 % | 0.009 % | 0.010 % | 0.014 % |

The shape metric below divides this factor out, so `validate` records it beside
every shape summary.

## 2. Shape, normalised at $k = 0.05$

Shape error is the largest fractional departure from CLASS over
$k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$, after both spectra are
renormalised at $k = 0.05\ h\,\mathrm{Mpc}^{-1}$, which makes it independent of
the amplitude above.

At $z = 0$ the median is **0.064 %**, the 90th percentile 0.152 % and the
maximum 0.225 %. The median stays between 0.062 % and 0.070 % across the
redshift range, and the cold spectrum $P_{cb}$ scores 0.063 % at $z = 0$.

| at $z = 0$ | median | 90th | max |
|---|---|---|---|
| amplitude at $k = 0.05$ | 0.010 % | 0.030 % | 0.062 % |
| shape, renormalised | 0.064 % | 0.152 % | 0.225 % |
| **total, absolute** | **0.066 %** | 0.144 % | 0.238 % |
| *the metric's own floor* | *0.020 %* | *0.031 %* | *0.032 %* |

**0.066 %** is the single number to quote for $P(k)$; the amplitude is an order
of magnitude better, so the shape sets the total.

The floor row is the metric's own: a CLASS spectrum pushed through the
emulator's interpolation with no network involved. The network predicts on a
fixed 400-node $\ln k$ grid while the metric asks CLASS at 300 other
wavenumbers, so what happens between the nodes is scored as network error. A
linear interpolant gives 0.11 % there, above the number it bounds, so
`model._interp_lnk` is a fixed four-point cubic. See {doc}`../design_notes`.

The scored range stops at $k = 10\ h\,\mathrm{Mpc}^{-1}$. The emulator is
trained to 200 to feed a halo-model $\sigma(M)$ integral; below 10 is where a
*linear* spectrum is the quantity an analysis uses directly.

![Residuals against CLASS](../_static/figures/02_residuals.png)

Twelve held-out cosmologies at $z = 0$, renormalised at
$k = 0.05\ h\,\mathrm{Mpc}^{-1}$. The shaded band is the median shape error
from the table above, for scale.

The error varies with $k$. Away from the acoustic scale the residuals reach
0.076 % below the pivot and 0.075 % above $k = 1\ h\,\mathrm{Mpc}^{-1}$; across
the BAO region, $k \approx 0.07$–$0.7\ h\,\mathrm{Mpc}^{-1}$, they reach
0.147 %, where the spectrum carries the most structure per decade.
`emu_pk.validate` scores a chosen band, for an analysis dominated by the
acoustic peak.

### The lowest decade

$k \in [10^{-4}, 10^{-3}]\ h\,\mathrm{Mpc}^{-1}$ is generated and trained on
but sits outside the scored range, and `validate` reports it separately:
median **0.995 %**, 90th percentile 1.57 %, maximum 6.94 %. Curvature acts
here and nowhere else — the curvature scale
$k_{\rm curv} = \sqrt{|\Omega_k|}H_0/c$ is $1.3\times10^{-4}\ h\,$Mpc$^{-1}$ at
the edge of the box, inside the grid rather than below it. Treat the decade
below $10^{-3}$ as indicative; {doc}`../design_notes` covers why `K_MIN` stays
at $10^{-4}$.

## 3. Derivatives

Derivative error is the median over $k$ of

$$\frac{\left|\partial\ln P/\partial\theta\ \text{(emulator)} -
        \partial\ln P/\partial\theta\ \text{(CLASS)}\right|}
       {\left|\partial\ln P/\partial\theta\ \text{(CLASS)}\right|},$$

the emulator side from automatic differentiation, the CLASS side from central
differences. The ratio is taken against CLASS's own derivative, so an axis the
emulator responds to weakly scores near 100 % rather than near zero.

At $z = 0$, with the comparison's own floor beside each:

| parameter | error | floor | | parameter | error | floor |
|---|---|---|---|---|---|---|
| `ln10A_s` | **exact** | — | | `sum_mnu` | 0.303 % | 0.010 % |
| `n_s` | **exact** | — | | `wa` | 0.353 % | 0.032 % |
| `omega_cdm` | 0.046 % | 0.038 % | | `Omega_k` | 0.141 % | 0.007 % |
| `h` | 0.094 % | 0.010 % | | `nu_r1` | 5.58 % | 0.261 % |
| `w0` | 0.097 % | 0.036 % | | `nu_r2` | 5.05 % | 0.202 % |
| `omega_b` | 0.159 % | 0.013 % | | | | |

`ln10A_s` and `n_s` are exact by construction: the primordial power law is
divided out of the training target and restored in closed form, so neither is a
network input and the two derivatives are $1$ and $\ln(kh/k_\ast)$. They score
$2\times10^{-14}$ and $8\times10^{-6}$, which is what the float32 forward pass
and CLASS's own finite difference leave behind. A Fisher matrix built on this
network is exact in two of its eleven directions.

The two mass ratios are the weakest axes at 5 %, and the ones with the least
signal to fit: splitting the sum unevenly moves $P(k)$ by 0.32 % at
$\Sigma m_\nu = 0.10$ eV and by under 0.012 % above 0.25 eV, so over most of the
box the effect is smaller than the emulator's own shape error.

The redshift dependence is not flat. `w0` and `wa` reach 1.50 % and 1.76 % at
$z = 5$, where CPL has least leverage on the expansion history, and `Omega_k`
reaches 0.577 %; `omega_cdm`, `h` and `sum_mnu` are flat or improve.
`validate --z` scores any redshift on its own.

### With respect to redshift

| | z = 0 | z = 0.5 | z = 1 | z = 2 |
|---|---|---|---|---|
| $\partial\ln P/\partial z$ | 0.060 % | 0.023 % | 0.012 % | 0.012 % |
| floor | 0.038 % | 0.009 % | 0.007 % | 0.006 % |

Away from $z = 0$ this sits within a factor of two or three of what the
comparison can resolve. At $z = 0$ the ratio is 1.6, where the node is an
endpoint in slope and the $z < 0$ side is outside the trained range.

### The floor

The reference is a central difference of CLASS, which carries a truncation term
and a solver-noise term of its own. `validate` recomputes it at half the step
and reports the difference as a **floor**, the smallest error the comparison
can resolve. At $z = 0$ that floor runs from 0.007 % for `Omega_k` to 0.261 %
for `nu_r1`, and is 0.038 % for the redshift derivative.

A score is read against its floor. `omega_cdm` scores 0.046 % against 0.038 %,
so the comparison barely resolves the network there; `sum_mnu` scores 0.303 %
against 0.010 %, thirty times its floor.

## Where in the box

An eleven-dimensional Latin hypercube samples the interior densely and the
corners sparsely, while a sampler with wide priors spends much of its time near
the walls, so `validate` scores regions separately. At $z = 0$:

| region | n | median | max |
|---|---|---|---|
| interior | 7 | 0.044 % | 0.109 % |
| edge (within 10 % of a bound) | 25 | 0.076 % | 0.225 % |
| extreme quintessence | 2 | 0.188 % | 0.221 % |
| $\lvert\Omega_k\rvert > 0.10$ | 8 | 0.077 % | 0.221 % |
| $\Sigma m_\nu < 0.30$ eV | 16 | 0.077 % | 0.221 % |
| $\Sigma m_\nu \ge 0.30$ eV | 16 | 0.061 % | 0.225 % |
| $\Omega_{\rm de} < 0$ | 0 | — | — |

The edge scores 1.7× the interior, the expected shape for a hypercube fit.

The curvature stratum matches the box as a whole, and the flat slice
$\Omega_k = 0$ scores 0.067 % against the full box's 0.064 %: the curvature axis
costs the other ten nothing. `validate --flat-only` produces that comparison.

The degenerate neutrino point $r = (1/3, 1/3)$ is a vertex of the sampled
simplex rather than an interior point, so a hypercube almost never lands on it,
and it is where an analysis keeping the degenerate approximation sits. It is
scored as its own stratum for both reasons.
