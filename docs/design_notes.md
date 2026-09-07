# Design notes

This is for someone reading or changing the source; using the package needs
none of it.

## The activation uses `jax.nn.sigmoid`, not `1/(1 + exp(-x))`

They agree to the last bit in *value* and not in *gradient*. Written out, the
reverse-mode derivative of the naive form carries
$e^{-\beta x}/(1+e^{-\beta x})^2$, and once $\beta x \lesssim -88$ the
exponential overflows to `inf` in float32: the value is still a correct `0` and
the gradient is `inf`/`inf` = `NaN`. `jax.nn.sigmoid` is the numerically stable
form and differentiates to $s(1-s)$, which is `0` there.

In a package whose entire purpose is that something differentiates it, an
activation that is differentiable only where it was tested is the worst
available defect. `tests/test_model.py` evaluates the gradient far into both
tails for this reason.

## The primordial power law is divided out of the training target

In linear theory with a power-law primordial spectrum,

$$\ln P(k,z) = \ln 10A_s + (n_s - 1)\ln\!\big(kh/k_*\big) + \ln T^2(k,z),$$

exactly: the transfer function does not know what $A_s$ or $n_s$ are, and
CLASS's `pk_lin` is that product. The network is given the second term only and
the first is added back by `model.primordial_ln_pk`.

Two consequences, and the second is the larger one. The derivatives
$\partial\ln P/\partial \ln 10A_s = 1$ and
$\partial\ln P/\partial n_s = \ln(kh/k_*)$ become *exact* rather than fitted —
a Fisher matrix built on this network is exactly right in two of its eight
directions. And amplitude and tilt are the two largest variance directions in
the target over this box, so removing them is capacity the network gets back
for the transfer function, the BAO and the neutrino suppression.

That `pk_lin` really does factorise this way is asserted against CLASS itself
in `tests/test_generate.py`, not assumed. If it were false, every spectrum
would be wrong by a smooth power law that nothing else in the suite could see.

## `k_pivot` is stated, not left to CLASS's default

It is CLASS's default value, but the training target is `ln P` with the
primordial power law divided out, which puts the pivot in the *inference* path,
and a pivot there cannot be a default: change CLASS's and every shipped weight
file silently means a different spectrum, with nothing raising. It is passed
to CLASS explicitly and written into the `.npz` beside the weights trained
against it.

Note the units: `k_pivot` is in **1/Mpc**, CLASS's convention, while everything
else in this package is in **h/Mpc**. The two meet in exactly one expression,
in `primordial_ln_pk`, which is where the factor of `h` lives.

## `validate.K_NORM` and `cosmo.K_PIVOT` are different things

They happen to share the number 0.05. `K_NORM` is where the shape comparison
renormalises and is in h/Mpc; `K_PIVOT` is the primordial pivot and is in
1/Mpc. The reduced target makes the second one load-bearing, so they carry
separate names even where they collide numerically.

## The correction table ships as a full cube, not two factors

A factorised table — a neutrino factor times a dark-energy factor — is far
cheaper: 2² and 2³ derivative arrays over small cubes against 2⁴ over a
1.8-million-element one. `assemble.build_ratio` builds the full grid and
measures the cross term a factorisation would discard, storing it as
`resid_max`. It reaches 1.61 % where the emulator's own shape error is 0.16 %,
which is why the table ships whole: the number is measured rather than
assumed.

The two effects couple physically — more late-time growth is more time for free
streaming to suppress — so the residual grows with the *product* of neutrino
mass and dark-energy deviation.

## The interpolator carries every mixed partial

Carrying one slope array per axis is C¹ *at a node* in every axis, because the
Hermite coefficients multiplying the carried slopes vanish there. Between nodes
it leaves a 1.6 × 10⁻² jump in $\partial r/\partial w_0$; with all 2⁴ mixed
partials that becomes 4.7 × 10⁻⁷.

Both constructions are in `interp.py` and both are tested: the cheap one is
the evidence for the expensive one, and the test suite parametrises over
on-node *and* off-node, because the on-node case passes either way.

## $\Omega_m$ contains the neutrinos

$\Omega_\nu = \Sigma m_\nu/(93.14 h^2)$, and $\Omega_m$ includes it. That is
`ggah_mod`'s convention, and a table built in a different one is wrong in a way
no test on either side can see, because each is self-consistent.
`tests/test_conventions.py` is the seam: it asserts the two agree whenever
`ggah_mod` is importable.

The 93.14 eV denominator is itself a *convention*, 0.53 % from the exact
Fermi-Dirac integral, and is committed to everywhere for exactly that reason.

## The curvature scale sits inside the k grid

`K_MIN` is $10^{-4}\ h\,\mathrm{Mpc}^{-1}$ and the curvature scale is
$k_{\rm curv} = \sqrt{|\Omega_k|}\,H_0/c$, which at the edge of the box
($|\Omega_k| = 0.15$) is $1.29\times10^{-4}$. The grid reaches below it.

That matters because curvature is otherwise the cheapest axis in the box.
Measured against CLASS, above $k \approx 10^{-2}$ the response is a
*k-independent growth rescaling*: $\partial\ln P/\partial\Omega_k = -1.81$
across the box, flat in $k$ to better than 0.5 % over four decades. The
transfer function is fixed long before curvature matters, so $\Omega_k$ moves
only the growth — and the network already spans that direction. Solving 40
held-out cosmologies twice, once flat and once curved, the extra spread the
ninth parameter adds to the training target is:

| band [h/Mpc] | extra variance at z=0 | z=1 | z=3 |
|---|---|---|---|
| $10^{-4}$–$10^{-3}$ | **+91 %** | +69 % | +67 % |
| $10^{-3}$–$10^{-2}$ | +20 % | +12 % | +5 % |
| $10^{-2}$–$1$ | +3.2 % | +1.8 % | −0.7 % |
| $1$–$200$ | −3.7 % | −4.2 % | −2.9 % |

The negatives are sampling noise on 40 points, and they are the finding: over
almost all of the scored range, adding curvature does not widen the target at
all. The whole cost is in the lowest decade, where the response turns over,
crosses zero near $k \approx 6\times10^{-4}$, reaches $2.5$ in $\ln P$ at
$\Omega_k = +0.15$, and is **not monotonic in $\Omega_k$** — at $k = 10^{-4}$,
$\Omega_k = -0.05$ gives $-0.843$ and $-0.15$ gives $-0.635$. A smooth MLP has
to represent a saturating, sign-changing ridge there.

**`K_MIN` stays at $10^{-4}$ anyway**, for a reason that is not about this
package. `ggah_mod`'s `DIFFERENTIABLE` and `FAST` backends both quadrature
$\sigma(M)$ from $k_{\min} = 10^{-4}$, and `_interp_lnk` continues *below* the
grid as a power law with the slope of the bottom two nodes. For a flat
cosmology that continuation is roughly $k^{n_s}$ and is a safety net; for a
curved one it is wrong by construction, because the true slope there is steep,
sign-changing and $\Omega_k$-dependent. Raising `K_MIN` would not remove the
feature, it would move it into an extrapolation the consumer integrates over
and no test can see — the same shape of defect as the `jnp.interp` clamp this
package already carries a test for.

So the band is **scored instead**. `validate.K_LOWK` reports
$k \in [10^{-4}, 10^{-3}]$ separately from `K_TRUSTED`, and the headline
numbers keep their existing definition. Sixteen per cent of the network's
outputs live in that band; before this they were generated and never scored,
which is a claim nobody had checked rather than a limitation anybody had
stated.

There is deliberately **no per-band weight in the trainer**. From the shipped
`feat_std_m`, the band is 64 of 400 modes carrying 16.9 % of
$\sum\mathrm{feat\_std}^2$; the curvature response takes that to about 19 %.
Per-wavenumber standardisation absorbs it, so a weight would be a knob with no
measured justification, and it would stop `val_loss` being the mean squared
error in $\ln P$ — which is the property that makes $\sqrt{\mathrm{val\_loss}}$
readable as an RMS fractional error at all.

## A checkpoint must feed every sampled parameter, not merely name known ones

`__init__` refuses a checkpoint naming an input this package does not know.
That is the easy direction and it was the only one checked. The dangerous
direction is the other: a checkpoint written against a *narrower* box names
nothing unknown, so it loads, `_in_idx` comes out bit-identical, and it
predicts a spectrum that silently ignores whatever axis has been added since.

Growing the box from eight parameters to nine is exactly when that fires, and
it fires on the file the package itself ships. So the absent case is refused
too, with `ANALYTIC` — `ln10A_s` and `n_s`, which the reduced target restores
in closed form — as the one legitimate exemption. `ANALYTIC` therefore lives in
`model`, where the inference path can see it, and `train` re-exports it: the
trainer must drop exactly the inputs the predictor is willing to find missing.

## JAX clamps an out-of-range gather, so a short `theta` returned a spectrum

`_forward` reads its inputs with `p[self._in_idx]`. JAX does not raise on an
out-of-range index — it clamps — and `_validate` could not catch it either,
because it zips `box.PARAMS` against `params` and `zip` stops at the shorter.
A `theta` one element short therefore gave the missing parameter some other
parameter's value and returned a finite, smooth, wrong spectrum. Measured
against the shipped weights: a seven-long vector where eight were wanted moved
$P(0.05)$ by 10.3 %.

Nothing passed seven, so it was unreachable — until the box grew, at which
point every caller written against the old length becomes a short vector and
`Omega_k` takes `wa`'s value. Since `wa = 0` is the common case, it would have
looked right most of the time. The length is now checked explicitly; `.shape`
is static under `jit`, so the check costs no tracing and does not touch the
gradient.

## The network's redshift input is `log10(1+z)`

$\mathrm{d}\ln P/\mathrm{d}\log_{10}(1+z) = -2\ln(10)\,f(z)$ with the growth
rate $f$ bounded in roughly $[0.5, 1]$, so `ln P` is nearly linear in this
variable with a bounded, monotonic slope. Measured against CLASS over
$z \in [0,5]$: departure from a straight line is 0.196 in $\log_{10}(1+z)$,
0.359 in $z$ and 0.782 in $a = 1/(1+z)$.

It matters because $z = 0$ is a node in *value* and an endpoint in *slope* —
nothing on the $z<0$ side constrains it — and a nearly straight function gives
a network very little reason to bend there. In this variable
$\partial\ln P/\partial z$ at $z=0$ is accurate to 0.155 %, against a
finite-difference floor of 0.049 %, and to better than 0.02 % everywhere else
in the range. $z=0$ remains the worst node, because it is the endpoint.

The transform is internal. `pk(k, z, ...)` takes a redshift and `jax.grad` of
it is still $\mathrm{d}/\mathrm{d}z$, by the chain rule.

## The output is standardised `ln P` per wavenumber, not PCA coefficients

This is what CosmoPower's own released linear-matter model does, and they
report having tested PCA against it and preferring the direct form. The reason
it matters here is not compression — the PCA residual is 7.6e-6, four orders
below the error — but that a basis makes every coefficient error *non-local in
k*. One bad coefficient is a wiggle across all 400 wavenumbers, and the metric
is the **max** fractional error over `k`.

Both forms are supported; the checkpoint declares which it is in
`output_form`, and loading a file under the wrong assumption would return a
spectrum rather than an error, so it is read and never inferred.

## The shipped file is the best epoch; the resume file is the last

They answer different questions. The validation curve is noisy at the
tens-of-per-cent level around a descending trend, so the last epoch is not
reliably the best one and shipping it is a coin flip nobody sees. A restart, by
contrast, has to continue from where training actually *is* — the optimiser
trajectory is a property of that, not of where it was best.

The resume file also carries the optimiser state, including the learning-rate
schedule's position. Without it a preempted run reinitialises Adam *and*
rewinds the schedule to its peak.

## The extras split is load-bearing

`import emu_pk` in an environment with no `classy` and no `optax` must work,
and `tests/` asserts it. That is what lets a downstream package depend on this
one without inheriting a Boltzmann solver or a training stack. Generation and
validation are `[gen]`; training is `[train]`.

## `load_weights` is cached on the file's identity, not its path

Caching on the path is right until something rewrites that path, and then it is
silently wrong: retraining to the same filename and reloading returns the
previous network. The cache key includes mtime and size.

## The extrapolation above the grid is a power law, not a clamp

`jnp.interp` clamps at the edges, and a clamped linear spectrum is *flat* above
the last mode instead of falling as $k^{-3}\ln^2 k$. The grid reaches
200 h/Mpc, which is what the consumer integrates to, so the tail is a safety net
rather than a load-bearing extrapolation — but it is a net, not a cliff.
