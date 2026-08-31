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

## What it costs

Measured, not estimated:

| | |
|---|---|
| CLASS solves in the design | 150 000 |
| seconds per solve, production settings | ~6.5 |
| **core-hours** | **~271** |
| training rows (31 redshifts per solve) | ~4.6 million |
| training, 240 epochs on 32 CPU cores | ~2.5 hours |
| assembled training set on disk | ~9 GB |

Generation is embarrassingly parallel across shards and is the only part that
needs a cluster. Training is a 4×512 network and fits comfortably on a CPU
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
