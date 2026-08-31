# Accuracy

Every number here is from `emu_pk/data/validation.json`, which is written by
`python -m emu_pk.validate` and never typed by hand — a validation figure
retyped is a validation figure that can silently outlive the weights it
describes.

## Against CLASS

![Shape error and derivative error](../_static/figures/02_accuracy.png)

*Left:* the shape error against held-out CLASS solves, across redshift. The
design is drawn from a different seed from the training set's, so no scored
point was trained on. *Right:* the derivative error at $z=0$, with the
comparison's own finite-difference floor marked — a score at its floor is a
statement about the ruler, not the network.

![Residuals against CLASS](../_static/figures/02_residuals.png)

Twelve held-out cosmologies at $z=0$, renormalised at
$k = 0.05\ h\,\mathrm{Mpc}^{-1}$ so that a spectrum right in shape and wrong in
amplitude is not scored as wrong in both. The shaded band is CosmoPower's
median shape error for scale.

**The error is not uniform in $k$**, and the figure is more useful than the
median for that reason. Away from the acoustic scale the residuals sit within
about $\pm 0.05\%$; through the BAO region, $k \approx 0.1$–$0.3\
h\,\mathrm{Mpc}^{-1}$, they reach $\pm 0.3\%$. That is where the spectrum has
the most structure per decade and where any emulator works hardest. If your
analysis is dominated by the acoustic peak, score that range specifically
rather than relying on a median over four decades.

## What the metric is

**Shape error** is the largest fractional departure from CLASS over
$k \in [10^{-3}, 10]\ h\,\mathrm{Mpc}^{-1}$, after renormalising at
$k = 0.05$. Not the full grid: the emulator is trained to 200 h/Mpc, but a
*linear* spectrum there is far inside the regime a halo model replaces, and
scoring it would report a number nobody uses.

**Amplitude error** is what that renormalisation throws away: the fractional
error in $P$ at $k = 0.05\ h\,\mathrm{Mpc}^{-1}$ itself. Dividing it out is
what makes the shape number a statement about shape — but on its own it would
let a spectrum wrong by a constant factor score perfectly, so `validate`
reports the discarded factor beside every shape summary rather than discarding
it. Read together they are an accuracy claim about $P(k)$, not about its shape
alone.

**Derivative error** is the median over $k$ of

$$\frac{\left|\partial\ln P/\partial\theta\ \text{(emulator)} -
        \partial\ln P/\partial\theta\ \text{(CLASS)}\right|}
       {\left|\partial\ln P/\partial\theta\ \text{(CLASS)}\right|},$$

with the emulator side from automatic differentiation and the CLASS side from
central differences. Dividing by CLASS's own derivative is deliberate: a
parameter the emulator is simply *blind* to then scores 1 rather than something
small. A derivative that is absent shows up in a Fisher matrix as a flat
direction, which is visible; one that is merely wrong does not.

## The amplitude is not where the error is

The natural worry about a renormalised metric is that it hides a constant
offset. Here it does not — the amplitude is roughly an order of magnitude
better than the shape, so removing it barely moves the number. At $z = 0$,
over the same held-out design:

| | median | 90th | max |
|---|---|---|---|
| amplitude at $k = 0.05$ | 0.012 % | 0.035 % | 0.059 % |
| shape, renormalised | 0.111 % | 0.224 % | 0.621 % |
| **total, nothing removed** | **0.112 %** | 0.221 % | 0.603 % |

The last row is the one to quote if you want a single number for $P(k)$: the
median absolute error is **0.112 %**, against 0.111 % for the shape. The
amplitude holds across the range too, between 0.005 % and 0.012 % at every
scored redshift.

Two things follow. The 0.111 % headline is not an artefact of the
renormalisation. And an analysis that marginalises over amplitude — which most
do, through $A_s$ or $\sigma_8$ — is using the part of the prediction that was
already the more accurate of the two.

## The floor

The reference is a finite difference, which is not exact. `validate` recomputes
it at half the step and reports the difference as a **floor**: the metric cannot
resolve an error below it. At $z=0$ that floor runs from 0.008 % (`h`) to
0.05 % (`wa`), and is 0.049 % for the redshift derivative — so the parameters
with the largest quoted errors are also the ones measured against the bluntest
ruler.

This matters for reading the numbers. `emu_pk`'s redshift derivative is
0.015 % at $z = 0.5$ against a floor of 0.011 % — the comparison can barely
tell the network apart from CLASS, and quoting a smaller number would be
quoting noise.

## Where in the box

An eight-dimensional Latin hypercube essentially never samples a corner, so a
median over the design says nothing about the walls — and the walls are where a
sampler with a wide prior spends its time. `validate` reports the edge (within
10 % of any bound) and the extreme-quintessence corner separately from the
interior.
