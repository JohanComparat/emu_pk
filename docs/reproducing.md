# Reproducing the training set

None of this is needed to *use* `emu_pk` — the trained weights and the
correction table ship inside the package. It is here so the result is
reproducible, and because a reader who wants to retrain on a different box
needs the same machinery.

## The short version

```bash
pip install 'emu_pk[gen,train]' camb==1.6.6

# One shard of solves: CAMB at the training precision, times the CLASS
# heating pair.  The design is regenerated from the seed, so a
# shard is reproducible from its indices alone.
python -m emu_pk.generate --mode emu --shard 0 --n-per-shard 100 \
       --n-total 150000 --out shards

# ... many shards later ...
python -m emu_pk.assemble --mode emu --shards shards --out training_set.npz
python -m emu_pk.train --dataset training_set.npz --out weights.npz
python -m emu_pk.validate --weights weights.npz --json validation.json
python -m emu_pk.validate --weights weights.npz --truth reference --json reference.json
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
It separates "curvature costs accuracy" from "this design is smaller than the
production one" — at any size below production both are true, and only the
control tells them apart. A network trained on a
pinned column has near-zero `x_std` along it and is meaningful only at
`Omega_k = 0`; score it with `validate --flat-only`.

## What it costs

Measured, not estimated:

| | |
|---|---|
| solves in the design | 150 000 |
| seconds per solve, two Dahu cores | ~160 |
| **core-hours** | **~13 000** |
| training rows (31 redshifts per solve) | ~4.65 million |
| assembled training set on disk | ~15 GB, in 32 parts |
| training, 240 epochs, 32 Dahu CPU cores | ~10 hours (154 s/epoch) |
| training, 240 epochs, one A100, full float32 | ~1 hour (14.4 s/epoch) |
| training, 240 epochs, a 16 GB laptop GPU | ~50 minutes (12 s/epoch) |
| validation against the training truth | ~840 solves |

A solve is CAMB at `lAccuracyBoost = 3`, `AccuracyBoost = 2` and no late
radiation truncation, plus a CLASS pair at CLASS's defaults for the heating
ratio, to $k = 300$. The raised precision is most of the cost, and it is what
puts the training truth 0.039 % from CAMB's converged answer in the median
over this box; CLASS at its own defaults sits 0.41 % from its converged answer
over the same box, more than six times the network's error.

Generation is embarrassingly parallel across shards and is the part that needs
a cluster. Measured on a mobile i9 (8 physical cores behind 16 threads), with
CLASS at its defaults: 2.75 s/solve solo on a cool machine, and **0.44 solves/s
in aggregate at any worker count** once it is hot, because the cores drop to
1.1 GHz at 100 °C and sixteen concurrent solver instances thrash a 24 MiB shared
L3. A rate taken from a burst is off by an order of magnitude from what a
sustained run delivers, which is what the `calibrate` gate exists to prevent.

## Training on a GPU

The network is 4×512, so a step is limited by its overhead rather than by
arithmetic, and a GPU is fast for that reason rather than for its FLOPs: the
laptop card above is as quick as the A100. Two settings matter, and both fail
quietly:

- **Full float32 matrix products.** On an A100, JAX multiplies float32 matrices
  in TF32 unless told otherwise. With the same data, seed and schedule, a TF32
  run tracks a CPU run for the first epochs and then stops at a validation loss
  of 5.9e-7 against 3.7e-7, with a worst-case error 1.6 times higher. Set
  `JAX_DEFAULT_MATMUL_PRECISION=highest` for training; `oarsub/run_train.sh`
  does. Evaluating the shipped network on a GPU is unaffected on the laptop
  card above: it agrees with the CPU to 2e-5 in $\ln P$ at either setting.
- **`--host-targets` on a card smaller than the training set.** The 150k set's
  targets are 13.5 GiB, and putting them on the device takes twice that for a
  moment, which a 16 GB card cannot do. The option keeps them in host memory
  and sends each batch over: same batches, same order, same weights bit for bit.

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest python -m emu_pk.train \
    --dataset training_set.npz --out weights.npz --epochs 240 --host-targets
```

## The properties that make it restartable

Three, and every one of them is load-bearing on a preemptible queue:

- **A shard skips if its output exists**, so a killed job re-runs and costs only
  the work it had not finished.
- **Shards write every 50 cosmologies**, not at the end, so a kill loses
  minutes rather than hours.
- **Training checkpoints every epoch**, including the optimiser state and the
  learning-rate schedule's position, so a preempted run resumes where it was
  rather than reinitialising Adam and rewinding the schedule to its peak.

## Where the solvers refuse

The generator refuses 25 of the 150 000 points, 0.017 %. Twenty-two of them
sit where `w0 + wa` approaches zero, so `w(a)` climbs toward zero at early
times and dark energy behaves like matter before recombination; the other three
are on the phantom side, at `w0 + wa` near $-1.5$. The CLASS half of the
heating pair is retried once at a thousandfold tighter tolerance before a point
counts as refused: near the quintessence corner CLASS's stiff integrator can
lose its step size on the last modes below $k = 300$, and the tighter tolerance
solves them.

**Curvature adds no new refusals inside the box**, and that is what fixes its
width. Measured with CLASS on the closed side at the low-density corner
(`omega_cdm = 0.05`, `h = 0.85`, $\Omega_m = 0.108$), CLASS fails from
$\Omega_k = -0.275$ onward; at $\Omega_m = 0.261$ it survives to $-0.40$. At
$|\Omega_k| \le 0.15$ nothing in the box refuses. The open side never refuses
at all — not even where the closure drives $\Omega_{\rm de}$ to $-0.4$ — so it
has a *stated* bound rather than a discovered one.

`assemble.build_training_set` **reports the missing design indices rather than
filling them**. A training set with silent gaps trains perfectly well and is
wrong exactly where the solver refused, which is the part of the box a forecast
is most likely to wander into.

## Reproducibility and the solver version

Everything here is reproducible from a seed and an index **given the same
solver versions**. Shards, datasets and weights record which solver, which
precision settings and which heating wrote them, and `assemble` refuses a
directory that mixes two. They do not record the solver *version*, which is
the one input to this pipeline a seed and an index do not capture.

The shipped weights were trained on **CAMB 1.6.6** and **CLASS v3.3.4**. The
version matters at the accuracy this package is scored at. Re-solving six
training cosmologies with CAMB 2.0.4 differs from the stored rows by up to
4.4e-4 in $\ln P$, at $k \approx 0.1\ h\,\mathrm{Mpc}^{-1}$ and low redshift,
against a network error of about 6e-4. With CAMB 1.6.6 on a different machine,
CPU and compiler, the same six reproduce to 6.6e-7, which is the float32 the
rows are stored in. Pin CAMB to validate the shipped weights, and record both
versions beside your own.

`validate --cache` keeps truth solves between runs. Its keys hash the point, the
redshifts and the wavenumbers rounded well above machine noise, so a cache
filled on one machine serves another: the scoring wavenumbers come out of
`exp`, and numpy rounds that differently with and without AVX-512.

## Cluster scripts

The `oarsub/` directory in the repository holds the job scripts used on the
GRICAD clusters (OAR resource manager). They are site-specific and will not
transfer unchanged, but they document the structure of a production run — gate
on a measured solve rate before sizing anything, run a `--devel` smoke before
committing array elements, and keep every step restartable.
