# Reproducing the training set

None of this is needed to *use* `emu_pk` — the trained weights and the
correction table ship inside the package. It is here so the result is
reproducible, and because a reader who wants to retrain on a different box
needs the same machinery.

## The short version

```bash
pip install 'emu_pk[gen,train]'

# One shard of CLASS solves.  The design is regenerated from the seed, so a
# shard is reproducible from its indices alone.
python -m emu_pk.generate --mode emu --shard 0 --n-per-shard 100 \
       --n-total 150000 --out shards

# ... many shards later ...
python -m emu_pk.assemble --mode emu --shards shards --out training_set.npz
python -m emu_pk.train --dataset training_set.npz --out weights.npz
python -m emu_pk.validate --weights weights.npz --json validation.json
```

**A shard directory belongs to one box.** `emu_shard` skips a chunk whose
output exists, and the filename carries the shard index and the design offset
but *not* the parameters — so a directory reused across a box change keeps the
old shards and mixes two designs. `z` and `lnk` do not change when the box
does, so the grid check would not see it either. Each shard therefore stamps
`box.PARAMS` into its `.npz` and `assemble` refuses a mismatch by name, which
also catches a *permuted* column. Generate into a fresh directory.

**The flat control.** `--pin Omega_k=0` holds a design column fixed after the
draw, giving a design identical to the curved one in its other ten columns.
It separates "the wider box is worse" from "this design
is smaller than the one that scored 0.064 %" — at any size below production
both are true, and only the control tells them apart. A network trained on a
pinned column has near-zero `x_std` along it and is meaningful only at
`Omega_k = 0`; score it with `validate --flat-only`.

## What it costs

Measured, not estimated:

| | |
|---|---|
| CLASS solves in the design | 150 000 |
| seconds per solve, production settings | ~8.7 |
| **core-hours** | **~360** |
| training rows (31 redshifts per solve) | ~4.6 million |
| training, 240 epochs on 32 CPU cores | ~2.5 hours |
| assembled training set on disk | ~9 GB |

Two measurements set the per-solve figure. **Curvature is free**: 2.7–3.0 s
flat and at $\Omega_k = \pm0.15$ on the same machine, statistically identical.
**Three separate neutrino species cost 1.34×**: 2.81 s degenerate against
3.76 s at `N_ncdm=3`, at production settings. The overhead amortises over the
expensive high-$k$ solve rather than tripling anything. On the cluster's slower
cores that 1.34× applies to 6.5 s.

Generation is embarrassingly parallel across shards and is the only part that
needs a cluster — and it genuinely needs one. Measured on a mobile i9 (8
physical cores behind 16 threads): 2.75 s/solve solo on a cool machine, and
**0.44 solves/s in aggregate at any worker count** once it is hot, because the
cores drop to 1.1 GHz at 100 °C and sixteen concurrent CLASS instances thrash a
24 MiB shared L3. A rate taken from a burst is off by an order of magnitude
from what a sustained run delivers, which is what the `calibrate` gate exists
to prevent. Training is a 4×512 network and fits comfortably on a CPU
node; a GPU is not required.

## The properties that make it restartable

Three, and every one of them is load-bearing on a preemptible queue:

- **A shard skips if its output exists**, so a killed job re-runs and costs only
  the work it had not finished.
- **Shards write every 50 cosmologies**, not at the end, so a kill loses
  minutes rather than hours.
- **Training checkpoints every epoch**, including the optimiser state and the
  learning-rate schedule's position, so a preempted run resumes where it was
  rather than reinitialising Adam and rewinding the schedule to its peak.

## Where CLASS refuses

About 0.02 % of solves fail, all `CosmoComputationError` out of
`perturbations_solve`, and they are not scattered: they sit in the corner where
`w0` is near $-0.5$ and `wa` is positive, so `w(a)` climbs toward zero at early
times and dark energy behaves like matter before recombination.

**Curvature adds no new refusals inside the box**, and that is what fixes its
width. Measured on the closed side at the low-density corner
(`omega_cdm = 0.05`, `h = 0.85`, $\Omega_m = 0.108$), CLASS fails from
$\Omega_k = -0.275$ onward; at $\Omega_m = 0.261$ it survives to $-0.40$. At
$|\Omega_k| \le 0.15$ nothing in the box refuses. The open side never refuses
at all — not even where the closure drives $\Omega_{\rm de}$ to $-0.4$ — so it
has a *stated* bound rather than a discovered one.

`assemble.build_training_set` **reports the missing design indices rather than
filling them**. A training set with silent gaps trains perfectly well and is
wrong exactly where CLASS refused, which is the part of the box a forecast is
most likely to wander into.

## Reproducibility and the solver version

Everything here is reproducible from a seed and an index **given the same CLASS
version**. CLASS changes; its precision settings and its `pk_lin` interpolation
change with it, so two runs of the commands above against different `classy`
builds are not guaranteed to agree at the accuracy this package is scored at.

The shipped weights were trained against **CLASS v3.3.4**. That version is not
stamped into the `.npz` files, so it cannot be recovered from them: if you
regenerate, record the `classy` version alongside your own weights. It is the
one input to this pipeline that a seed and an index do not capture.

## Cluster scripts

The `oarsub/` directory in the repository holds the job scripts used on the
GRICAD clusters (OAR resource manager). They are site-specific and will not
transfer unchanged, but they document the structure of a production run — gate
on a measured solve rate before sizing anything, run a `--devel` smoke before
committing array elements, and keep every step restartable.
