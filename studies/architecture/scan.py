r"""Train one network of the architecture scan and write its scores.

The scan asks how the test error of the emulator depends on the depth and width
of the network and on the number of training cosmologies, on the production
training set (``work/v21/training_set_v21_c.npz``, 150 000 CAMB cosmologies at
31 redshifts).  Every run uses the same held-out cosmologies, so their scores
are comparable:

* the cosmologies are permuted once (``SPLIT_SEED``); the first ``N_TEST`` are
  the test set, the next ``N_VAL`` the validation set, the rest the training
  pool;
* a run trains on the first ``--n-cos`` cosmologies of the pool, so the
  training sets are nested;
* the target, the input and output standardisation and the loss (mean squared
  error in :math:`\ln P`) are those of :func:`emu_pk.train.train`, with the
  standardisation taken on the run's own training set;
* Adam, a warm-up and cosine schedule over a fixed number of steps
  (``--steps``), batches of 1024 rows; the weights kept are those of the best
  validation loss.

The scores, written to ``<out>/<tag>.json``, are the test-set statistics of
``validate``: per cosmology and redshift, the maximum over
:math:`10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}` of :math:`|\Delta\ln P_{\rm m}|`,
then its median, 90th percentile and maximum over cosmologies, at :math:`z = 0`
and over all redshifts.  Also written: the training and validation losses at
the kept weights, and the roughness of :math:`\partial\ln P/\partial\omega_{\rm b}`
along a line in :math:`\omega_{\rm b}` (the root mean square of its second
difference over that of the derivative), which is large for a piecewise-affine
network, and the derivative itself along that line.

Run (GPU)::

    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest \
        python studies/architecture/scan.py --hidden 512 512 512 512 \
        --n-cos 145500 --out ~/data/emu_pk_architecture_scan
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from emu_pk import box
from emu_pk.assemble import load_training_set
from emu_pk.model import Z_VARS
from emu_pk.train import ANALYTIC, COLS, _apply, _init, reduce_target

DATASET = pathlib.Path(__file__).resolve().parents[2] / "work" / "v21" / "training_set_v21_c.npz"
SPLIT_SEED = 20261115
N_TEST, N_VAL = 3000, 1500
K_TRUSTED = (1e-3, 10.0)
BATCH = 1024


#: Activations of the activation test.  ``cp`` is CosmoPower's learned
#: :math:`[\gamma + (1-\gamma)\,\sigma(\beta x)]\,x` (two parameters per unit), the
#: one ``emu_pk`` ships; the others are fixed functions.  ``silu`` is ``cp`` with
#: :math:`\beta = 1, \gamma = 0` frozen.
ACTS = {"relu": jax.nn.relu, "gelu": jax.nn.gelu, "silu": jax.nn.silu,
        "tanh": jnp.tanh, "sigmoid": jax.nn.sigmoid, "softplus": jax.nn.softplus}


def tag_of(hidden, n_cos, steps, seed, act="cp"):
    tag = f"h{'x'.join(map(str, hidden))}_n{n_cos}_s{steps}_r{seed}"
    return tag if act == "cp" else f"{tag}_{act}"


def make_apply(act):
    """The network for ``act``: :func:`emu_pk.train._apply` for ``cp``."""
    if act == "cp":
        return _apply
    f = ACTS[act]

    def apply(p, x, n_layers):
        for i in range(n_layers - 1):
            x = f(x @ p[f"W{i}"] + p[f"b{i}"])
        return x @ p[f"W{n_layers - 1}"] + p[f"b{n_layers - 1}"]
    return apply


def split(n_cos_total):
    """Test, validation and training-pool cosmology indices, fixed for the scan."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_cos_total)
    return perm[:N_TEST], perm[N_TEST:N_TEST + N_VAL], perm[N_TEST + N_VAL:]


def rows_of(cos, n_z):
    return (np.sort(cos)[:, None] * n_z + np.arange(n_z)[None, :]).ravel()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, nargs="+", required=True)
    ap.add_argument("--n-cos", type=int, required=True,
                    help="training cosmologies, taken from the start of the pool")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--act", default="cp", choices=["cp", *ACTS])
    ap.add_argument("--save-weights", action="store_true",
                    help="also write the kept weights to <out>/<tag>_weights.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    net = make_apply(a.act)

    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tag = tag_of(a.hidden, a.n_cos, a.steps, a.seed, a.act)
    if (out / f"{tag}.json").exists():
        print(f"{tag}: done already", flush=True)
        return
    t_start = time.monotonic()

    # -- data, split by cosmology ---------------------------------------------
    X, Ym, Ycb, lnk = load_training_set(a.dataset)
    zc = X[:, COLS.index("z")]
    n_z = len(np.unique(zc))
    n_cos_total = len(X) // n_z
    assert np.array_equal(zc[:n_z], zc[n_z:2 * n_z]), "redshift axis does not tile"
    test_c, val_c, pool = split(n_cos_total)
    if a.n_cos > len(pool):
        raise SystemExit(f"--n-cos {a.n_cos} > pool of {len(pool)}")
    tr_r, va_r, te_r = rows_of(pool[:a.n_cos], n_z), rows_of(val_c, n_z), rows_of(test_c, n_z)
    print(f"{tag}: {a.n_cos} training cosmologies ({len(tr_r)} rows), "
          f"{len(val_c)} validation, {len(test_c)} test", flush=True)

    reduce_target(Ym, X, lnk)
    reduce_target(Ycb, X, lnk)
    keep = [c for c in COLS if c not in ANALYTIC]
    Xs = X[:, [COLS.index(c) for c in keep]].astype(np.float64)
    Xs[:, -1] = np.asarray(Z_VARS["log10_1pz"](Xs[:, -1]))
    x_mean, x_std = Xs[tr_r].mean(0), Xs[tr_r].std(0) + 1e-30
    Xn = ((Xs - x_mean) / x_std).astype(np.float32)
    del Xs

    def standardise(Y):
        m, s = Y[tr_r].mean(0, dtype=np.float64), Y[tr_r].std(0, dtype=np.float64) + 1e-30
        return m, s

    mm, sm = standardise(Ym)
    mc, scb = standardise(Ycb)
    n_k = Ym.shape[1]
    f32 = [v.astype(np.float32) for v in (mm, sm, mc, scb)]

    def target(r, chunk=200_000):
        # In float32 and in chunks: the full training split is 15 GB.
        T = np.empty((len(r), 2 * n_k), dtype=np.float32)
        for i in range(0, len(r), chunk):
            rr = r[i:i + chunk]
            T[i:i + len(rr), :n_k] = (Ym[rr] - f32[0]) / f32[1]
            T[i:i + len(rr), n_k:] = (Ycb[rr] - f32[2]) / f32[3]
        return T

    T_tr = target(tr_r)
    X_tr = Xn[tr_r]
    X_va, T_va = jnp.asarray(Xn[va_r]), jnp.asarray(target(va_r))
    lnp_te_m = Ym[te_r].astype(np.float64)      # reduced ln P_m, the truth
    X_te = jnp.asarray(Xn[te_r])
    z_te = zc[te_r]
    del Ym, Ycb, Xn, X
    on_device = T_tr.nbytes < 3e9
    if on_device:
        T_tr, X_tr = jnp.asarray(T_tr), jnp.asarray(X_tr)

    w_loss = jnp.asarray(np.concatenate([sm, scb]).astype(np.float32) / np.sqrt(2.0 * n_k))
    sizes = [len(keep), *a.hidden, 2 * n_k]
    n_layers = len(sizes) - 1
    params = _init(jax.random.PRNGKey(a.seed), sizes)   # same weights for every act
    if a.act != "cp":
        params = {k: v for k, v in params.items() if not k.startswith(("beta", "gamma"))}
    n_par = int(sum(np.size(v) for v in jax.tree_util.tree_leaves(params)))

    sched = optax.warmup_cosine_decay_schedule(
        init_value=a.lr * 1e-2, peak_value=a.lr, warmup_steps=max(1, a.steps // 50),
        decay_steps=a.steps, end_value=a.lr_end)
    opt = optax.adam(sched)
    state = opt.init(params)

    def loss_fn(p, x, t):
        return jnp.mean(jnp.sum(((net(p, x, n_layers) - t) * w_loss) ** 2, axis=1))

    @jax.jit
    def step(p, s, x, t):
        loss, g = jax.value_and_grad(loss_fn)(p, x, t)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss

    @jax.jit
    def batched_loss(p, x, t):
        return loss_fn(p, x, t)

    def full_loss(p, x, t, chunk=32_768):
        tot = 0.0
        for i in range(0, len(x), chunk):
            xb, tb = x[i:i + chunk], t[i:i + chunk]
            tot += float(batched_loss(p, xb, tb)) * len(xb)
        return tot / len(x)

    # -- training -------------------------------------------------------------
    rng = np.random.default_rng(a.seed + 1)
    n_tr = len(tr_r)
    order, pos = rng.permutation(n_tr), 0
    best, best_p, best_step, hist = np.inf, params, 0, []
    t0 = time.monotonic()
    for it in range(1, a.steps + 1):
        if pos + BATCH > n_tr:
            order, pos = rng.permutation(n_tr), 0
        idx = np.sort(order[pos:pos + BATCH])
        pos += BATCH
        if on_device:
            xb, tb = X_tr[idx], T_tr[idx]
        else:
            xb, tb = jnp.asarray(X_tr[idx]), jnp.asarray(T_tr[idx])
        params, state, loss = step(params, state, xb, tb)
        if it % a.eval_every == 0 or it == a.steps:
            v = full_loss(params, X_va, T_va)
            hist.append((it, float(loss), v))
            if v < best:
                best, best_p, best_step = v, params, it
            if it % (10 * a.eval_every) == 0 or it == a.steps:
                print(f"  step {it:7d}  batch {float(loss):.3e}  val {v:.3e}  "
                      f"best {best:.3e} @ {best_step}  {time.monotonic() - t0:6.0f} s",
                      flush=True)
    t_train = time.monotonic() - t0

    # -- scores ---------------------------------------------------------------
    sub = np.random.default_rng(3).choice(n_tr, size=min(n_tr, 100_000), replace=False)
    sub.sort()
    if on_device:
        train_loss = full_loss(best_p, X_tr[sub], T_tr[sub])
    else:
        train_loss = full_loss(best_p, jnp.asarray(X_tr[sub]), jnp.asarray(T_tr[sub]))

    pred = np.concatenate([np.asarray(jax.jit(lambda x: net(best_p, x, n_layers))(X_te[i:i + 32_768]))
                           for i in range(0, X_te.shape[0], 32_768)])
    lnp_pred_m = pred[:, :n_k].astype(np.float64) * sm + mm
    d = np.abs(lnp_pred_m - lnp_te_m)
    kk = np.exp(lnk)
    trusted = (kk >= K_TRUSTED[0]) & (kk <= K_TRUSTED[1])
    per = d[:, trusted].max(axis=1)              # one value per (cosmology, z)
    z0 = np.isclose(z_te, 0.0)

    def stats(v):
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "max": float(np.max(v))}

    # Roughness of d ln P / d omega_b at k = 0.1, z = 0, the others at the box centre.
    centre = np.array([(lo + hi) / 2 for lo, hi in (box.BOX[c] for c in keep[:-1])])
    i_ob = keep.index("omega_b")
    lo, hi = box.BOX["omega_b"]
    ob = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 400)
    ik = int(np.argmin(np.abs(kk - 0.1)))
    xs, xsd = jnp.asarray(x_mean[:-1]), jnp.asarray(x_std[:-1])

    def lnp_at(o):
        th = jnp.asarray(centre).at[i_ob].set(o)
        x = jnp.concatenate([(th - xs) / xsd, jnp.atleast_1d((0.0 - x_mean[-1]) / x_std[-1])])
        return net(best_p, x[None, :].astype(jnp.float32), n_layers)[0, ik] * sm[ik]

    der = np.asarray(jax.vmap(jax.grad(lnp_at))(jnp.asarray(ob, dtype=jnp.float32)), dtype=np.float64)
    rough = float(np.sqrt(np.mean(np.diff(der, 2) ** 2)) / np.sqrt(np.mean(der ** 2)))

    res = {
        "tag": tag, "act": a.act, "hidden": a.hidden, "depth": len(a.hidden), "n_par": n_par,
        "n_cos": a.n_cos, "n_rows": int(n_tr), "steps": a.steps, "seed": a.seed,
        "epochs": a.steps * BATCH / n_tr, "best_step": best_step,
        "val_loss": best, "train_loss": train_loss,
        "test_z0": stats(per[z0]), "test_allz": stats(per),
        "rms_lnp_test": float(np.sqrt(np.mean(d[:, trusted] ** 2))),
        "roughness_dlnp_domega_b": rough,
        "t_train_s": t_train, "t_total_s": time.monotonic() - t_start,
        "device": str(jax.devices()[0]), "history": hist,
        "dlnp_domega_b": {"omega_b": ob.tolist(), "k": float(kk[ik]), "der": der.tolist()},
    }
    if a.save_weights:
        np.savez_compressed(out / f"{tag}_weights.npz",
                            **{k: np.asarray(v) for k, v in best_p.items()})
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"{tag}: median z=0 {res['test_z0']['median']:.2e}, p90 {res['test_z0']['p90']:.2e}, "
          f"train/val {train_loss:.2e}/{best:.2e}, {n_par} weights, {t_train:.0f} s", flush=True)


if __name__ == "__main__":
    main()
