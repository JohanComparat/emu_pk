r"""Fit the PCA basis and the MLP.  Needs ``optax``; nothing else does.

Two output representations, selectable so the choice is measurable rather
than assumed.

``--direct`` (CosmoPower's, for this quantity)
    the network predicts standardised :math:`\ln P` at every wavenumber.
    CosmoPower's released linear-matter model is ``PKLIN_NN``, a
    ``cosmopower_NN``; Spurio Mancini et al. (2022) say they tested both
    approaches and found the direct one "in general more accurate", and reserve
    PCA for :math:`C_\ell^{TE}` and :math:`C_\ell^{\phi\phi}`, where some values
    are negative so the logarithm is unavailable.  :math:`\ln P` is positive
    everywhere, so that reason does not apply here.

default (PCA)
    a 64-component basis with the network predicting coefficients.  Kept as the
    baseline the direct form is measured against.

The architecture and activation *are* CosmoPower's -- four layers of 512 with a
learned :math:`(\gamma + (1-\gamma)\sigma(\beta x))x` -- which is what makes the
comparison one of training sets and objectives rather than of two different
ideas.

What is different here is the box (:mod:`emu_pk.box`: eight parameters against
five, and ``w0``/``wa`` present at all) and the wavenumber reach (200 h/Mpc
against 14.6), which is the whole point.

**Checkpoint every epoch.**  The training job runs under OAR ``besteffort``,
which means it can be killed at any moment to make room for someone else's
work; a run that keeps its state only in memory would restart from scratch each
time and never finish.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import box, cosmo
from .model import Z_VARS as model_Z_VARS
from .model import activation, primordial_ln_pk

__all__ = ["COLS", "fit_pca", "reduce_target", "train", "main"]

#: Columns of the assembled design matrix, in order.  ``assemble`` writes the
#: eight sampled parameters and then ``z``; this is that layout named once so
#: the trainer selects columns by name rather than by a number that is right
#: until the box changes.
COLS = list(box.PARAMS) + ["z"]

#: Inputs the reduced target does not need, because it does not depend on them.
#: See :func:`emu_pk.model.primordial_ln_pk`.
ANALYTIC = ("ln10A_s", "n_s")

#: Staged learning schedule, ``(lr, patience, max_epochs)``.
#:
#: CosmoPower's rates, minus the top one, and with the epoch budgets resized.
#: Their ``cosmopower_NN.train`` defaults are five stages from 1e-2 to 1e-6,
#: each 100 patience out of at most 1000 epochs, and **they do not transfer to
#: this training set**.  Measured, in job 468324:
#:
#: ===============  ==================  =====================
#: .                CosmoPower          emu_pk
#: ===============  ==================  =====================
#: training rows    1.8e5 spectra       3.0e6 rows
#: steps per epoch  ~176                ~2780
#: ===============  ==================  =====================
#:
#: Adam moves each parameter by about ``lr`` per step whatever the gradient
#: scale, so one epoch here is sixteen of theirs' worth of updates at the same
#: nominal rate.  At 1e-2 the fit reaches 4.9 % RMS in its first epoch, blows
#: up to 243 % in its second and never beats epoch one again, at a cost of 100
#: epochs of patience -- two and a half hours -- to establish it.
#:
#: So: start at 1e-3, which is stable on this training set, and end at 1e-6.
#: Budgets are per-stage epochs, and early stopping rather than a fixed count
#: because the point of a stage is to run until its rate has nothing left to
#: give.  Batch stays 1024 throughout, as it does for them.
STAGES = ((1e-3, 20, 150),
          (1e-4, 20, 150),
          (1e-5, 20, 150),
          (1e-6, 20, 150))


def reduce_target(Y, X, lnk, k_pivot=cosmo.K_PIVOT, chunk=200_000):
    r"""Subtract the closed-form primordial term from ``ln P``, **in place**.

    Turns the target into :math:`\ln P - \ln10A_s - (n_s-1)\ln(kh/k_*)`,
    which is exactly independent of both, so the network is left with the
    transfer function alone and those two derivatives become analytic.

    In place and in chunks because the target is a few million rows by 400
    modes: the obvious ``Y - term`` allocates a second copy of a multi-gigabyte
    array, and the machine this runs on is a shared node with other people's
    jobs on it.  ``Y`` is the array ``load_training_set`` just returned and
    nothing else holds a reference, so overwriting it is safe -- but it *is*
    overwritten, hence the name and this paragraph.

    The term comes from :func:`emu_pk.model.primordial_ln_pk`, the same function
    inference adds back.  One definition, or the training set and the predictor
    disagree by a power law that is smooth, finite, and invisible to every test
    that does not involve CLASS.
    """
    i_h, i_ns, i_as = (COLS.index(c) for c in ("h", "n_s", "ln10A_s"))
    lnk = np.asarray(lnk)[None, :]
    for a in range(0, len(Y), chunk):
        b = min(a + chunk, len(Y))
        term = np.asarray(primordial_ln_pk(
            lnk, X[a:b, i_h:i_h + 1], X[a:b, i_ns:i_ns + 1],
            X[a:b, i_as:i_as + 1], k_pivot))
        Y[a:b] -= term.astype(Y.dtype)
    return Y


def fit_pca(Y, n_comp: int, sample: int = 40_000, seed: int = 0):
    """Mean, basis and coefficient scaling for ``ln P``.

    The SVD is taken on a random subsample rather than the whole set: the basis
    of a smooth family converges long before the row count does, and a full SVD
    of a multi-million-row matrix costs more than the training that follows it.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Y), size=min(sample, len(Y)), replace=False)
    mean = Y.mean(axis=0)
    _, _, vt = np.linalg.svd(Y[idx] - mean, full_matrices=False)
    basis = vt[:n_comp]                                   # (n_comp, n_k)
    coeff = (Y - mean) @ basis.T
    return (mean.astype(np.float64), basis.astype(np.float64),
            coeff.mean(axis=0).astype(np.float64),
            coeff.std(axis=0).astype(np.float64) + 1e-30)


def _init(key, sizes, scale=0.05):
    """Dense stack plus the two learned activation parameters per hidden layer."""
    p = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k1 = jax.random.split(key)
        p[f"W{i}"] = jax.random.normal(k1, (a, b)) * (scale * np.sqrt(2.0 / a))
        p[f"b{i}"] = jnp.zeros(b)
        if i < len(sizes) - 2:
            p[f"beta{i}"] = jnp.ones(b)
            p[f"gamma{i}"] = jnp.zeros(b)
    return p


def _opt_to_arrays(state) -> dict:
    """An optax state as flat arrays, keyed by leaf index.

    Leaf *order* is the contract, and it is stable: it comes from the pytree
    structure of the state, which is fixed by the optimiser and the parameter
    tree.  :func:`_opt_from_arrays` rebuilds against a freshly initialised state
    of the same shape and refuses anything that does not line up, so a
    checkpoint written by a different optimiser cannot be loaded into this one.
    """
    leaves = jax.tree_util.tree_leaves(state)
    d = {f"opt{i:03d}": np.asarray(v) for i, v in enumerate(leaves)}
    d["opt_n"] = np.int64(len(leaves))
    return d


def _opt_from_arrays(d, fresh):
    """Rebuild an optax state, or ``None`` if the checkpoint does not fit it.

    ``None`` rather than an exception: a checkpoint whose optimiser state is
    missing or stale is still a perfectly good set of *weights*.  Resuming the
    weights while re-initialising the optimiser is the right fallback, and the
    caller says so out loud rather than doing it silently.
    """
    leaves = jax.tree_util.tree_leaves(fresh)
    if int(d.get("opt_n", -1)) != len(leaves):
        return None
    got = []
    for i, ref in enumerate(leaves):
        key = f"opt{i:03d}"
        if key not in d.files:
            return None
        a = d[key]
        # A NaN in an Adam moment poisons every subsequent step exactly the way
        # a NaN weight does, and it is not visible in `val_loss`.
        if a.shape != np.shape(ref) or not np.all(np.isfinite(a)):
            return None
        got.append(jnp.asarray(a, dtype=np.asarray(ref).dtype))
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(fresh), got)


def _apply(p, x, n_layers):
    for i in range(n_layers - 1):
        x = activation(x @ p[f"W{i}"] + p[f"b{i}"], p[f"beta{i}"], p[f"gamma{i}"])
    return x @ p[f"W{n_layers - 1}"] + p[f"b{n_layers - 1}"]


def train(dataset, out, n_comp=64, hidden=(512, 512, 512, 512), epochs=60,
          batch=1024, lr=1e-3, seed=0, resume=True, val_frac=0.05,
          reduced=True, weighted=True, schedule=True, warmup_epochs=2,
          lr_end=1e-5, direct=False, staged=False, stages=STAGES,
          z_var="z"):
    r"""Train both heads at once and write ``out``.

    One network with two heads rather than two networks: ``P_m`` and ``P_cb``
    differ only through the neutrino sector, share every other dependence, and
    training them together is what stops them drifting apart in a way that would
    show up as a spurious cold-vs-total effect downstream.

    Three things are flags rather than decisions, because each replaced
    something that shipped and each has to be measurable against what it
    replaced on the same data:

    ``reduced``
        fit :math:`\ln P` with the primordial power law divided out, so
        ``ln10A_s`` and ``n_s`` leave the network's inputs and re-enter
        analytically.  See :func:`reduce_target`.
    ``weighted``
        weight the per-component loss by the coefficient's own standard
        deviation, which makes the objective the mean squared error in
        :math:`\ln P` rather than in whitened coefficients.  See below.
    ``schedule``
        warm up and then cosine-decay the learning rate, instead of holding
        1e-3 for the whole run.
    ``direct``
        predict standardised :math:`\ln P` at every wavenumber instead of
        coefficients on a PCA basis.  This is CosmoPower's own choice for this
        quantity; see the module docstring.
    ``staged``
        run ``STAGES`` -- CosmoPower's own schedule, five learning rates
        from 1e-2 to 1e-6, each until early stopping.  Overrides ``schedule``,
        ``epochs`` and ``lr``, because it supplies all three.
    ``z_var``
        what to feed the redshift column as; see :data:`emu_pk.model.Z_VARS`.
        ``log10_1pz`` is the variable :math:`\ln P` is nearly linear in, which
        is where the network's freedom to bend at :math:`z=0` comes from.
    """
    import optax

    from .assemble import load_training_set

    out = pathlib.Path(out)
    X, Ym, Ycb, lnk = load_training_set(dataset)
    print(f"training set: {X.shape[0]} rows x {Ym.shape[1]} modes, "
          f"{X.shape[1]} inputs", flush=True)
    if X.shape[1] != len(COLS):
        raise ValueError(
            f"{dataset} has {X.shape[1]} design columns; this box has "
            f"{len(COLS)} ({', '.join(COLS)}).  The dataset was assembled "
            "against a different box and its columns do not mean what the "
            "trainer would assume they mean.")

    # -- what the network is asked to predict, and what it is fed -------------
    if reduced:
        reduce_target(Ym, X, lnk)
        reduce_target(Ycb, X, lnk)
        keep = [c for c in COLS if c not in ANALYTIC]
        print(f"  target: ln P with the primordial power law divided out; "
              f"{', '.join(ANALYTIC)} are analytic and leave the inputs "
              f"({len(COLS)} -> {len(keep)})", flush=True)
    else:
        keep = list(COLS)
        print("  target: ln P as generated (--no-reduced)", flush=True)
    Xs = X[:, [COLS.index(c) for c in keep]]
    # `keep` ends with "z" (COLS does, and ANALYTIC does not contain it), so
    # the last column is the redshift.  Fancy indexing above already copied.
    if z_var != "z":
        Xs[:, -1] = np.asarray(model_Z_VARS[z_var](Xs[:, -1]))
        print(f"  redshift fed as {z_var}", flush=True)

    x_mean, x_std = Xs.mean(axis=0), Xs.std(axis=0) + 1e-30
    Xn = ((Xs - x_mean) / x_std).astype(np.float32)

    # -- the output representation -------------------------------------------
    # Either way this produces `T`, the standardised target, and `lw`, the
    # per-output scale that converts an error in `T` back into an error in
    # `ln P`.  The loss weighting below is then identical in both modes, which
    # is what makes `--direct` and `--no-weighted` independent choices rather
    # than one confounded knob.
    n_k = Ym.shape[1]
    decode = {}
    if direct:
        # CosmoPower's construction for this quantity: standardise `ln P` per
        # wavenumber and predict it.  No basis, so no truncation and no
        # smearing of one coefficient's error across every k -- which is what a
        # metric defined as the *max* fractional error over k cares about.
        for tag, Y in (("m", Ym), ("cb", Ycb)):
            fmean = Y.mean(axis=0)
            fstd = Y.std(axis=0) + 1e-30
            decode[f"feat_mean_{tag}"] = fmean.astype(np.float64)
            decode[f"feat_std_{tag}"] = fstd.astype(np.float64)
            Y -= fmean                    # in place: these are gigabytes
            Y /= fstd
        print(f"  output: standardised ln P at {n_k} wavenumbers, direct "
              f"({2 * n_k} network outputs)", flush=True)
        # `copy=False`, because at 3e6 rows this array is 9.6 GB and the inputs
        # are already float32: a plain `.astype` would hold a second copy of it
        # alongside the concatenation and the two it was built from.
        T = np.concatenate([Ym, Ycb], axis=1).astype(np.float32, copy=False)
        lw = np.concatenate([decode["feat_std_m"],
                             decode["feat_std_cb"]]).astype(np.float32)
    else:
        pca = {}
        for tag, Y in (("m", Ym), ("cb", Ycb)):
            mean, basis, cmean, cstd = fit_pca(Y, n_comp, seed=seed)
            pca[tag] = (mean, basis, cmean, cstd)
            # On a held-out sample, not on all three million rows: the full
            # reconstruction is two (3e6 x 400) temporaries, several GB apiece,
            # for a number that a hundred thousand rows already pins to three
            # digits -- and rows the basis was not fitted on are the more
            # honest measure.
            rng = np.random.default_rng(seed + 1)
            held = rng.choice(len(Y), size=min(100_000, len(Y)), replace=False)
            Yh = Y[held]
            resid = np.std(Yh - ((Yh - mean) @ basis.T) @ basis - mean)
            print(f"  PCA[{tag}]: {n_comp} components, "
                  f"residual {resid:.3e} in ln P (100k held-out rows)",
                  flush=True)
            decode[f"pca_{tag}"] = basis
            decode[f"pca_mean_{tag}"] = mean
            decode[f"coeff_mean_{tag}"] = cmean
            decode[f"coeff_std_{tag}"] = cstd
        T = np.concatenate([
            (((Ym - pca["m"][0]) @ pca["m"][1].T) - pca["m"][2]) / pca["m"][3],
            (((Ycb - pca["cb"][0]) @ pca["cb"][1].T) - pca["cb"][2]) / pca["cb"][3],
        ], axis=1).astype(np.float32)
        lw = np.concatenate([pca["m"][3], pca["cb"][3]]).astype(np.float32)
    # Several gigabytes each, and nothing below reads them.  Direct mode has
    # already consumed them in place; PCA mode has projected them.
    del Ym, Ycb

    # -- the loss weight ------------------------------------------------------
    # The target above is whitened per component, so an unweighted MSE over it
    # weights all 128 outputs equally.  They are not equal: the SVD basis is
    # orthonormal, so an error of `dc` in the raw coefficients is an error of
    # exactly `|dc|` in `ln P`, and `dc = dT * cstd`.  With `cstd` spanning
    # orders of magnitude across 64 components, an unweighted MSE spends most of
    # the fit on the components that contribute least to the spectrum.
    #
    # Weighting by `cstd` and dividing by the number of modes makes the loss
    # *literally* the mean squared error in `ln P` -- so `sqrt(val)` is the RMS
    # fractional error in P, which is the number the package is scored on rather
    # than a proxy for it.
    if weighted:
        w_loss = jnp.asarray(lw / np.sqrt(2.0 * n_k))
        print(f"  loss: mean squared error in ln P (component weights span "
              f"{lw.max() / lw.min():.3g}x)", flush=True)
    else:
        w_loss = jnp.asarray(
            np.full(T.shape[1], 1.0 / np.sqrt(T.shape[1]), dtype=np.float32))
        print("  loss: unweighted MSE on whitened coefficients "
              "(--no-weighted)", flush=True)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xn))
    n_val = int(len(Xn) * val_frac)
    val, tr = perm[:n_val], perm[n_val:]

    sizes = [Xn.shape[1], *hidden, T.shape[1]]
    n_layers = len(sizes) - 1
    params = _init(jax.random.PRNGKey(seed), sizes)

    # -- the learning rate ----------------------------------------------------
    # A constant 1e-3 for the whole run is still descending at epoch 240 on this
    # training set.  Warmup keeps the first few hundred steps from throwing the
    # weights somewhere the activation overflows -- which is not hypothetical
    # here, see `emu_pk.model.activation` -- and the cosine tail is where a
    # regression fit of this kind stops bouncing around its minimum and settles
    # into it.
    # Fixed here rather than beside the epoch loop, because the schedule is
    # defined in *steps* and has to agree with what the loop actually runs.
    n_batch = max(1, len(tr) // batch)

    # One plan drives both modes: a list of (lr, patience, max_epochs).  The
    # non-staged modes are a single stage with no patience, which keeps the
    # loop below from having two shapes.
    if staged:
        plan = [tuple(s) for s in stages]
        total_planned = sum(s[2] for s in plan)
        print(f"  lr: staged, {' -> '.join(f'{s[0]:.0g}' for s in plan)}; "
              f"each until {plan[0][1]} epochs without improvement, "
              f"at most {total_planned}", flush=True)
    elif schedule:
        # `decay_steps` is the whole run, and optax gives the cosine whatever is
        # left after the warmup -- so a warmup at least as long as the run
        # leaves it zero steps and optax raises.  That is not a corner: a
        # two-epoch smoke against the default two-epoch warmup hits it exactly,
        # which is how it was found.  Cap the warmup at a fifth of the run.
        total_steps = max(2, epochs * n_batch)
        warmup_steps = max(1, min(warmup_epochs * n_batch, total_steps // 5))
        lr_sched = optax.warmup_cosine_decay_schedule(
            init_value=lr * 1e-2, peak_value=lr, warmup_steps=warmup_steps,
            decay_steps=total_steps, end_value=lr_end)
        plan = [(lr_sched, None, epochs)]
        print(f"  lr: {lr:.2g} peak at step {warmup_steps}, cosine to "
              f"{lr_end:.2g} over {epochs} epochs ({total_steps} steps)",
              flush=True)
    else:
        plan = [(lr, None, epochs)]
        print(f"  lr: {lr:.2g} constant (--no-schedule)", flush=True)

    def _optimiser(stage_i):
        return optax.adam(plan[stage_i][0])

    opt = _optimiser(0)
    state = opt.init(params)
    stage, start = 0, 0          # stage index, and epochs done inside it
    stage_best, stage_since = float("inf"), 0
    done = 0                     # epochs finished in *earlier* stages

    # Two files, because they answer two questions.  `out` is the model that
    # ships, and it is the *best* epoch seen -- the validation curve is noisy at
    # the tens-of-per-cent level around a descending trend, so the last epoch is
    # not reliably the best one and shipping it is a coin flip nobody sees.
    # `resume_path` is the *last* epoch, which is what a restart has to continue
    # from: the optimiser trajectory is a property of where training actually
    # is, not of where it was best.
    resume_path = out.with_name(out.stem + ".resume.npz")
    best = float("inf")
    # `keep` is COLS filtered, and COLS ends with "z", so this still ends with
    # "z" -- which `model.PkEmulator` checks rather than trusts.
    meta = {"params_order": np.array(keep, dtype="U16"),
            "target_form": "reduced" if reduced else "raw",
            # How to read the network's output.  `PkEmulator` branches on this
            # at load time; a direct file decoded as PCA (or the reverse) does
            # not fail, it returns a spectrum.
            "output_form": "direct" if direct else "pca",
            "z_var": z_var,
            # `val_loss` is not comparable across these two.  Unweighted it is a
            # mean square over whitened PCA coefficients; weighted it is the
            # mean squared error in ln P, and they differ by orders of
            # magnitude.  A guard that compares one against the other reads as
            # a huge improvement and means nothing, so the number carries its
            # own definition.
            "loss_form": "lnp_mse" if weighted else "whitened_mse",
            "k_pivot": np.float64(cosmo.K_PIVOT)}

    if resume and resume_path.exists():
        out_for_resume = resume_path
    else:
        out_for_resume = out
    if resume and out_for_resume.exists():
        # Reconstruct by the exact key names the architecture implies, rather
        # than by pattern-matching what is in the file: a checkpoint written by
        # a *different* architecture must fail to load, not load partially.
        with np.load(out_for_resume) as d:
            names = list(params)
            # Shapes, not just names and layer count.  The reduced target
            # narrows the input from nine columns to seven, which leaves
            # `n_layers` and every key name identical and changes `W0` from
            # (9, h) to (7, h) -- so a check on names alone accepts a
            # full-target checkpoint into a reduced-target network and dies at
            # the first matmul, hours into a queue slot.  Checkpoints from a
            # different target sit at exactly the path a new run writes to, so
            # this is an expected case and not a corner.
            fits = (int(d.get("n_layers", -1)) == n_layers
                    and "epoch" in d.files
                    and all(n in d.files for n in names)
                    and all(np.shape(d[n]) == np.shape(params[n])
                            for n in names))
            if fits:
                # A checkpoint can be poison: a run that hits a NaN gradient
                # keeps writing checkpoints, and resuming from one of those
                # trains sixty more epochs of NaN from a clean start.  Check
                # before trusting.
                bad = [n for n in names if not np.all(np.isfinite(d[n]))]
                if bad or not np.isfinite(float(d["val_loss"])):
                    print(f"  checkpoint at epoch {int(d['epoch'])} is not "
                          f"finite ({'weights: ' + ', '.join(bad[:3]) if bad else 'val_loss'})"
                          f"; starting from scratch rather than resuming it",
                          flush=True)
                else:
                    params = {n: jnp.asarray(d[n]) for n in names}
                    best = float(d.get("best_val", d["val_loss"]))
                    # Which stage, and how far into it.  Without this a
                    # preempted staged run restarts at stage 0 and 1e-2 --
                    # undoing every decade of learning rate it had bought, on a
                    # queue where preemption is routine.
                    stage = int(d.get("stage", 0))
                    if stage >= len(plan):
                        stage = len(plan) - 1
                    start = int(d.get("stage_epoch", d["epoch"]))
                    stage_best = float(d.get("stage_best", best))
                    stage_since = int(d.get("stage_since", 0))
                    # The global epoch counter minus the in-stage one is what
                    # earlier stages contributed; recomputing it from `plan`
                    # would assume every earlier stage ran to its maximum, and
                    # early stopping is the whole point of them not doing that.
                    done = max(0, int(d["epoch"]) - start)
                    opt = _optimiser(stage)   # `step` is built from this below
                    # **Restore the optimiser, not just the weights.**  An
                    # unconditional `opt.init(params)` throws away Adam's
                    # moments on every restart -- and this job runs
                    # `besteffort`, so it is restarted routinely rather than
                    # exceptionally.  With a *scheduled* learning rate that
                    # goes from harmless-but-wasteful to actively harmful: the
                    # schedule position lives in the optimiser state too, so a
                    # fresh init also rewinds the learning rate to its peak, and
                    # a run preempted often enough would never decay at all.
                    restored = _opt_from_arrays(d, opt.init(params))
                    if restored is None:
                        state = opt.init(params)
                        print("  no usable optimiser state in the checkpoint; "
                              "Adam and the lr schedule restart from step 0 "
                              "(the weights still resume)", flush=True)
                    else:
                        state = restored
                    where = (f"stage {stage + 1}/{len(plan)} "
                             f"(lr {plan[stage][0]:.0g}), epoch {start} of it"
                             if staged else f"epoch {start}")
                    print(f"  resuming from {where} "
                          f"(val {float(d['val_loss']):.3e}, "
                          f"best so far {best:.3e})", flush=True)
            elif "epoch" in d.files:
                shapes = {n: (np.shape(d[n]), np.shape(params[n]))
                          for n in names
                          if n in d.files and np.shape(d[n]) != np.shape(params[n])}
                why = (f"input width {np.shape(d['W0'])} vs {np.shape(params['W0'])}"
                       if "W0" in shapes else
                       f"{len(shapes)} arrays differ" if shapes else
                       f"n_layers {int(d.get('n_layers', -1))} vs {n_layers}")
                print(f"  checkpoint at {out_for_resume.name} is a different "
                      f"architecture ({why}); starting from scratch "
                      f"(pass --no-resume to silence)", flush=True)

    def loss(p, xb, tb):
        # `w_loss` already carries the 1/sqrt(n_modes), so this sums over
        # components and means over the batch: with `weighted`, exactly the mean
        # squared error in ln P.
        return jnp.mean(
            jnp.sum(((_apply(p, xb, n_layers) - tb) * w_loss) ** 2, axis=-1))

    def _make_step(optimiser):
        """A freshly jitted step bound to *this* optimiser.

        Not one `step` closing over a rebindable `opt`.  `jax.jit` traces a
        function once and caches on the argument signature; a Python value
        captured from the enclosing scope is baked into that trace and
        rebinding it afterwards does **not** retrace.  So a single jitted
        `step` reading `opt` from the enclosing scope keeps using whichever
        optimiser existed when it was first traced -- every stage would have run
        at the first stage's learning rate, and the only symptom is that the
        schedule does nothing.

        Caught by running a two-stage plan whose second stage has lr 1e-12: the
        weights kept moving, and at 1e-12 they cannot.  A new function object
        per stage gets its own cache entry.
        """
        @jax.jit
        def step(p, s, xb, tb):
            l, g = jax.value_and_grad(loss)(p, xb, tb)
            upd, s = optimiser.update(g, s)
            return optax.apply_updates(p, upd), s, l
        return step

    step = _make_step(opt)

    Xtr, Ttr = jnp.asarray(Xn[tr]), jnp.asarray(T[tr])
    Xva, Tva = jnp.asarray(Xn[val]), jnp.asarray(T[val])

    for si in range(stage, len(plan)):
        slr, patience, smax = plan[si]
        if si != stage:
            # A new stage is a new optimiser.  CosmoPower rebuilds Adam at each
            # learning rate and so does this: the moments were accumulated at a
            # rate ten times higher and carrying them into the next decade is
            # carrying momentum the new rate did not ask for.
            opt = _optimiser(si)
            state = opt.init(params)
            step = _make_step(opt)          # or the new lr is never applied
            start, stage_best, stage_since = 0, float("inf"), 0
        stage = si
        if start >= smax:
            # Resumed past the end of this stage -- it finished before the
            # preemption.  Without this the inner range is empty, `ep` is
            # whatever the previous stage left it at, and the epoch count goes
            # backwards.
            print(f"  -- stage {si + 1}/{len(plan)}: already complete "
                  f"({start} epochs)", flush=True)
            done += smax
            start = 0
            continue
        if staged:
            print(f"  -- stage {si + 1}/{len(plan)}: lr {slr:.0g}, "
                  f"patience {patience}, at most {smax} epochs", flush=True)

        ep_done = start
        for ep in range(start, smax):
            t0 = time.time()
            order = np.asarray(rng.permutation(len(tr)))
            tot = 0.0
            for b in range(n_batch):
                sl = order[b * batch:(b + 1) * batch]
                params, state, l = step(params, state, Xtr[sl], Ttr[sl])
                tot += float(l)
            vl = float(loss(params, Xva, Tva))
            tl = tot / n_batch
            # With the weighted loss the number has units: it is the mean
            # squared error in ln P, so its root is the RMS fractional error in
            # P and the log says directly what the run is scored on.
            rms = f"  rms {np.sqrt(abs(vl)):.4%}" if weighted else ""
            tag = (f"{si + 1}.{ep + 1:<4d}" if staged
                   else f"{ep + 1:3d}/{smax}")
            print(f"  epoch {tag}  train {tl:.3e}  "
                  f"val {vl:.3e}{rms}  {time.time() - t0:.1f} s", flush=True)
        # Stop on the first non-finite epoch rather than checkpointing NaN for a
        # day.  The first run of this trained sixty epochs of `nan` before
        # anyone looked, and a checkpoint of NaN weights also poisons `resume`.
            if not (np.isfinite(tl) and np.isfinite(vl)):
                raise FloatingPointError(
                    f"epoch {ep + 1}: loss is not finite (train {tl}, val "
                    f"{vl}). The data is not the usual cause -- check it first "
                    f"with np.isfinite on X and T -- and the usual cause is a "
                    f"gradient that overflows where the value does not; see "
                    f"`emu_pk.model.activation` for the one that did. Nothing "
                    f"has been checkpointed for this epoch.")

            if vl < stage_best:
                stage_best, stage_since = vl, 0
            else:
                stage_since += 1

            # The optimiser state goes only into the resume file.  The shipped
            # file answers "what does this network predict"; an Adam trajectory
            # is no part of that, and it would double what `ggah_mod` loads.
            _save(resume_path, params, decode, x_mean, x_std, lnk, n_layers,
                  done + ep + 1, vl, best=min(best, vl), meta=meta,
                  opt_state=state, stage=si, stage_epoch=ep + 1,
                  stage_best=stage_best, stage_since=stage_since)
            if vl < best:
                best = vl
                _save(out, params, decode, x_mean, x_std, lnk, n_layers,
                      done + ep + 1, vl, best=best, meta=meta, stage=si,
                      stage_epoch=ep + 1)
                print(f"           ^ best so far, written to {out.name}",
                      flush=True)

            ep_done = ep + 1
            if patience is not None and stage_since >= patience:
                print(f"  -- stage {si + 1} stopped at epoch {ep + 1}: "
                      f"{patience} epochs without improving on "
                      f"{stage_best:.3e}", flush=True)
                break
        done += ep_done
        start = 0
    return out


def _save(out, params, decode, x_mean, x_std, lnk, n_layers, epoch, val,
          best=None, meta=None, opt_state=None, stage=0, stage_epoch=0,
          stage_best=None, stage_since=0):
    d = {k: np.asarray(v) for k, v in params.items()}
    d.update({k: np.asarray(v) for k, v in decode.items()})
    d.update(x_mean=np.asarray(x_mean), x_std=np.asarray(x_std), lnk=np.asarray(lnk),
             n_layers=np.int64(n_layers), epoch=np.int64(epoch),
             val_loss=np.float64(val),
             best_val=np.float64(val if best is None else best),
             # Where the schedule is, so a preempted staged run resumes into
             # the stage it was in rather than back at the top of the plan.
             stage=np.int64(stage), stage_epoch=np.int64(stage_epoch),
             stage_best=np.float64(val if stage_best is None else stage_best),
             stage_since=np.int64(stage_since))
    # `params_order` is what the network was actually fed and `target_form` is
    # what it was asked to predict.  Both are read back by
    # `model.PkEmulator.__init__` rather than assumed, because under the reduced
    # target the input list is shorter than `box.PARAMS` and a file loaded under
    # the wrong assumption returns a spectrum that is wrong by a power law and
    # finite everywhere.
    d.update(meta or {"params_order": np.array(list(box.PARAMS) + ["z"],
                                               dtype="U16"),
                      "target_form": "raw", "loss_form": "whitened_mse",
                      "output_form": "pca", "z_var": "z",
                      "k_pivot": np.float64(cosmo.K_PIVOT)})
    if opt_state is not None:
        d.update(_opt_to_arrays(opt_state))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="emu_pk/data/emu_pk_mlp.npz")
    ap.add_argument("--n-comp", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512] * 4,
                    help="hidden layer widths, e.g. --hidden 1024 1024 1024 1024")
    ap.add_argument("--no-resume", action="store_true")
    # The three target/loss/schedule choices, each switchable, so each is
    # measurable against its alternative on the same data.  Defaults are on;
    # the flags exist for the ablation.
    ap.add_argument("--no-reduced", action="store_true",
                    help="fit whole ln P instead of dividing out the "
                         "primordial power law")
    ap.add_argument("--no-weighted", action="store_true",
                    help="unweighted MSE on whitened PCA coefficients, "
                         "instead of mean squared error in ln P")
    ap.add_argument("--no-schedule", action="store_true",
                    help="hold the learning rate constant instead of "
                         "warming up and cosine-decaying it")
    ap.add_argument("--staged", action="store_true",
                    help="run CosmoPower's own schedule: five learning rates "
                         "from 1e-2 to 1e-6, each until early stopping.  "
                         "Overrides --epochs, --lr and --no-schedule")
    ap.add_argument("--z-var", default="z", choices=("z", "log10_1pz"),
                    help="what to feed the redshift as.  ln P is nearly linear "
                         "in log10(1+z) -- its slope there is -2 ln(10) f(z) "
                         "and f is bounded -- so the network has less curvature "
                         "to represent and less room to bend at z=0")
    ap.add_argument("--direct", action="store_true",
                    help="predict standardised ln P at every wavenumber "
                         "instead of PCA coefficients -- CosmoPower's own "
                         "choice for this quantity")
    a = ap.parse_args(argv)
    train(a.dataset, a.out, n_comp=a.n_comp, epochs=a.epochs, batch=a.batch,
          lr=a.lr, resume=not a.no_resume, hidden=tuple(a.hidden),
          reduced=not a.no_reduced, weighted=not a.no_weighted,
          schedule=not a.no_schedule, warmup_epochs=a.warmup_epochs,
          lr_end=a.lr_end, direct=a.direct, staged=a.staged, z_var=a.z_var)


if __name__ == "__main__":  # pragma: no cover
    main()
