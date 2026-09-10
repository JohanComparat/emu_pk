# Accuracy

Every number on this page is written by `python -m emu_pk.validate` into
`emu_pk/data/validation.json`, so the figures quoted always describe the
weights that ship beside them.

The scores come from held-out CLASS solves: a Latin hypercube drawn from a
different seed than the training design, so every scored point is new to the
network. Three quantities are measured, taken here in the order they compose —
the amplitude of the spectrum, its shape, and its derivatives.

![Shape error and derivative error](../_static/figures/02_accuracy.png)

*Left:* the shape error across redshift. *Right:* the derivative error at
$z=0$ for the nine fitted axes, with the comparison's own finite-difference
floor marked. `ln10A_s` and `n_s` are restored in closed form rather than
fitted and are left off; see {doc}`03_derivatives`.

## 1. Amplitude

The amplitude is the value of $P$ at the normalisation scale,
$k = 0.05\ h\,\mathrm{Mpc}^{-1}$. `emu_pk` reproduces it to a **median
0.010 %** at $z = 0$, with a 90th percentile of 0.030 % and a maximum of
0.062 % over 32 held-out cosmologies.

It holds across the trained range, between 0.009 % and 0.014 % at every scored
redshift:

| z | 0 | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| amplitude | 0.010 % | 0.012 % | 0.012 % | 0.009 % | 0.010 % | 0.014 % |

This is the factor the shape metric below divides out, so `validate` records it
beside every shape summary. The two together describe $P(k)$ itself.

## 2. Shape, normalised at $k = 0.05$

Shape error is the largest fractional departure from CLASS over
$k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$, after both spectra are
renormalised at $k = 0.05\ h\,\mathrm{Mpc}^{-1}$. Renormalising isolates the
shape, so this number and the amplitude above are independent statements.

`emu_pk` reaches a **median 0.064 %** at $z = 0$, with a 90th percentile of
0.152 % and a maximum of 0.225 %. CosmoPower's released linear-matter model
reaches 0.159 % on the same measure, on a box that is narrower in four of the
five axes the two share and equal on the fifth.
The median stays between 0.062 % and 0.070 % across the whole redshift range.
The cold spectrum $P_{cb}$ scores 0.063 % at $z = 0$, the same to within the
sampling.

Combining the two rows gives the accuracy of $P(k)$ as it stands:

| at $z = 0$ | median | 90th | max |
|---|---|---|---|
| amplitude at $k = 0.05$ | 0.010 % | 0.030 % | 0.062 % |
| shape, renormalised | 0.064 % | 0.152 % | 0.225 % |
| **total, absolute** | **0.066 %** | 0.144 % | 0.238 % |
| *the metric's own floor* | *0.020 %* | *0.031 %* | *0.032 %* |

**0.066 %** is the single number to quote for $P(k)$. It sits alongside the
0.064 % shape figure because the amplitude is accurate to roughly an order of
magnitude better, leaving the shape as the term that sets the total.

The floor row is measured the way `derivative_error` measures its own. The
network predicts on a fixed 400-node $\ln k$ grid and the metric asks CLASS at
300 other wavenumbers, so the interpolation between nodes is scored as network
error; the floor is a CLASS spectrum pushed through that same interpolation
with no network involved. At 0.020 % it is about a third of the reported
median. It was 1.01× the median under the linear interpolation this package
shipped before — the metric was measuring itself — which is why
`model._interp_lnk` is a fixed four-point cubic. See {doc}`../design_notes`.

The scored range stops at $k = 10\ h\,\mathrm{Mpc}^{-1}$. The emulator is
trained to 200, and that reach exists to feed a halo-model $\sigma(M)$
integral; the range scored here is the one where a *linear* spectrum is the
quantity an analysis uses directly.

![Residuals against CLASS](../_static/figures/02_residuals.png)

Twelve held-out cosmologies at $z=0$, renormalised at
$k = 0.05\ h\,\mathrm{Mpc}^{-1}$. The shaded band is CosmoPower's median shape
error, for scale.

**The error varies with $k$**, which is what makes the figure more informative
than the median. Away from the acoustic scale the residuals sit within about
$\pm 0.05\ \%$; through the BAO region, $k \approx 0.1$–$0.3\
h\,\mathrm{Mpc}^{-1}$, they reach $\pm 0.3\ \%$, where the spectrum carries the
most structure per decade. An analysis dominated by the acoustic peak is best
served by scoring that range specifically, which `emu_pk.validate` supports.

### The lowest decade

$k \in [10^{-4}, 10^{-3}]\ h\,\mathrm{Mpc}^{-1}$ is generated and trained on
but sits outside the scored range, and `validate` reports it as its own number:
**median 0.995 %**, 90th percentile 1.57 %, maximum 6.94 %. That is fifteen
times the trusted band and it is where curvature acts — the curvature scale
$k_{\rm curv} = \sqrt{|\Omega_k|}H_0/c$ is $1.3\times10^{-4}\ h\,$Mpc$^{-1}$ at
the edge of the box, inside the grid rather than below it. Treat the decade
below $10^{-3}$ as indicative. See {doc}`../design_notes` for why `K_MIN` stays
at $10^{-4}$ regardless.

## 3. Derivatives

Derivative error is the median over $k$ of

$$\frac{\left|\partial\ln P/\partial\theta\ \text{(emulator)} -
        \partial\ln P/\partial\theta\ \text{(CLASS)}\right|}
       {\left|\partial\ln P/\partial\theta\ \text{(CLASS)}\right|},$$

with the emulator side from automatic differentiation and the CLASS side from
central differences. The ratio is taken against CLASS's own derivative so that
a parameter that an emulator responds to weakly scores near 1, which a Fisher
matrix shows as a flat direction.

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
network input and the two derivatives are $1$ and $\ln(k/k_\ast)$ exactly. The
$2\times10^{-14}$ and $8\times10^{-6}$ they score is what the float32 forward
pass and CLASS's own finite difference leave behind. A Fisher matrix built on
this network is exact in two of its eleven directions. See
{doc}`../design_notes`.

The two mass ratios are the weakest axes, at 5 %. They are also the axes with
the least to fit: splitting the sum three ways instead of evenly moves $P(k)$
by 0.32 % at $\Sigma m_\nu = 0.10$ eV and by under 0.012 % above 0.25 eV, so
most of the box carries a signal smaller than the emulator's own shape error.
What the 5 % says is that the response is present and roughly right; an axis
the network ignored entirely would score near 100 %.

These are $z = 0$ figures and the redshift dependence is not flat. `w0` and
`wa` degrade towards the top of the range — 1.50 % and 1.76 % at $z = 5$ —
where CPL has least leverage on the expansion history; `Omega_k` reaches
0.577 %. `omega_cdm`, `h` and `sum_mnu` are flat or improve. `validate --z`
scores any redshift on its own.

### With respect to redshift

| | z = 0 | z = 0.5 | z = 1 | z = 2 |
|---|---|---|---|---|
| $\partial\ln P/\partial z$ | 0.060 % | 0.023 % | 0.012 % | 0.012 % |
| floor | 0.038 % | 0.009 % | 0.007 % | 0.006 % |

Away from $z = 0$ this sits within a factor of about two or three of what the
comparison can resolve. At $z = 0$ the ratio is 1.6, where the node is an
endpoint in slope and the $z < 0$ side is outside the trained range.

### The floor

The reference is a central difference of CLASS, which carries a truncation term
and a solver-noise term of its own. `validate` recomputes it at half the step
and reports the difference as a **floor**: the smallest error the comparison
can resolve. At $z = 0$ that floor runs from 0.007 % for `Omega_k` to 0.261 %
for `nu_r1`, and is 0.038 % for the redshift derivative.

Reading a score against its floor is what separates the network from the ruler.
`omega_cdm` scores 0.046 % against a floor of 0.038 %, so the measurement barely
resolves the network there; `sum_mnu` scores 0.303 % against 0.010 %, which is
the network's own error thirty times over.

## Where in the box

An eleven-dimensional Latin hypercube samples the interior densely and the
corners sparsely, while a sampler with wide priors spends much of its time near
the walls. `validate` therefore scores regions separately, each with its own
count, at $z = 0$:

| region | n | median | max |
|---|---|---|---|
| interior | 7 | 0.044 % | 0.109 % |
| edge (within 10 % of a bound) | 25 | 0.076 % | 0.225 % |
| extreme quintessence | 2 | 0.188 % | 0.221 % |
| $\lvert\Omega_k\rvert > 0.10$ | 8 | 0.077 % | 0.221 % |
| $\Sigma m_\nu < 0.30$ eV | 16 | 0.077 % | 0.221 % |
| $\Sigma m_\nu \ge 0.30$ eV | 16 | 0.061 % | 0.225 % |
| $\Omega_{\rm de} < 0$ | 0 | — | — |

The edge is where the error is, at 1.7× the interior. That is the expected
shape for a hypercube fit and it is the reason the regions are reported apart:
a sampler running against a wall is scoring the third row, not the first.

The curvature stratum is the same as the box as a whole, and the flat slice
$\Omega_k = 0$ scores 0.067 % against the full box's 0.064 %. **The ninth
parameter cost the other eight nothing** — that comparison is what
`validate --flat-only` exists for.

The degenerate neutrino point $r = (1/3, 1/3)$ is a vertex of the sampled
simplex rather than an interior point, so a hypercube almost never lands on it;
it is scored as its own stratum for that reason, and every published result
from before version 2 sits there.
