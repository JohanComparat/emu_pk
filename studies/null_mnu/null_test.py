"""Null test of the two output heads at sum_mnu = 0.

With no massive neutrinos the cold and total matter fields are the same field,
so every solver gives P_cb / P_m = 1 exactly, and every training row at
sum_mnu = 0 carries pcb = pm.  The network has two independent heads and is
never told so; this script measures how far apart they come out.

Measured: ln(P_cb/P_m) on the emulator grid, 1e-3 < k < 10 h/Mpc (the scoring
range) and the full grid, at every training redshift, over N cosmologies drawn
from the box with sum_mnu pinned to 0; and the gradient of ln(P_cb/P_m) with
respect to every parameter, which a Fisher forecast built on P_cb inherits.
For scale, the same gradients of ln P_m and the physical d ln(P_cb/P_m)/d sum_mnu.
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from emu_pk import PkEmulator, box, grid

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--json", default=None)
a = ap.parse_args()

emu = PkEmulator(check_box=False)
lnk = np.asarray(emu.lnk)
k = np.exp(lnk)
score = (k > 1e-3) & (k < 10.0)
z_nodes = np.asarray(grid.Z_NODES_EMU)
i_mnu = box.PARAMS.index("sum_mnu")

theta = box.sample(a.n, seed=a.seed, pin={"sum_mnu": 0.0})


def lnratio(p, z):
    return emu._forward(p, z, "cb") - emu._forward(p, z, "m")


f_all = jax.jit(jax.vmap(jax.vmap(lnratio, (None, 0)), (0, None)))
r = np.asarray(f_all(jnp.asarray(theta), jnp.asarray(z_nodes)))   # (n, n_z, n_k)

jac_r = jax.jit(jax.vmap(jax.jacfwd(lnratio), (0, None)))
jac_m = jax.jit(jax.vmap(jax.jacfwd(lambda p, z: emu._forward(p, z, "m")), (0, None)))
g_r, g_m = [], []
for z in (0.0, 1.0):
    g_r.append(np.asarray(jac_r(jnp.asarray(theta), z)))   # (n, n_k, n_par)
    g_m.append(np.asarray(jac_m(jnp.asarray(theta), z)))

out = {"n": a.n, "seed": a.seed}
ab = np.abs(r[..., score])
out["ratio_z0"] = {
    "median_of_max_abs": float(np.median(ab[:, 0].max(-1))),
    "p95_of_max_abs": float(np.percentile(ab[:, 0].max(-1), 95)),
    "max_abs": float(ab[:, 0].max()),
    "median_abs_all_k": float(np.median(ab[:, 0])),
    "mean_signed": float(r[:, 0][:, score].mean()),
}
out["ratio_all_z"] = {
    "median_of_max_abs": float(np.median(ab.max((-1, -2)))),
    "max_abs": float(ab.max()),
    "max_abs_full_grid": float(np.abs(r).max()),
    "per_z_median_of_max_abs": {f"{z:g}": float(np.median(ab[:, i].max(-1)))
                                for i, z in enumerate(z_nodes)},
}
out["gradient"] = {}
for iz, z in enumerate((0.0, 1.0)):
    gr, gm = g_r[iz][:, score, :], g_m[iz][:, score, :]
    out["gradient"][f"z={z:g}"] = {
        p: {"median_max_abs_dlnratio": float(np.median(np.abs(gr[..., j]).max(-1))),
            "median_max_abs_dlnPm": float(np.median(np.abs(gm[..., j]).max(-1))),
            "relative": float(np.median(np.abs(gr[..., j]).max(-1) /
                                        np.maximum(np.abs(gm[..., j]).max(-1), 1e-30)))}
        for j, p in enumerate(box.PARAMS)}

# the physical response, for scale: 2 f_nu / sum_mnu on small scales
om = (theta[:, 0] + theta[:, 1]) / theta[:, 2] ** 2
out["physical_dlnratio_dmnu_smallk"] = float(np.median(2.0 / (93.14 * theta[:, 2] ** 2 * om)))
print(json.dumps(out, indent=1))
if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
