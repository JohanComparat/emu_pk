"""Does the GPU's matrix-product precision change the shipped network's output?

On an A100, JAX multiplies float32 matrices in TF32 (ten mantissa bits) unless
JAX_DEFAULT_MATMUL_PRECISION=highest.  This evaluates ln P_m for 1 000
cosmologies of the box at the 31 training redshifts on the CPU and on the GPU at
the default and at the highest precision, and reports the largest difference.

    python precision_check.py --json precision_<node>.json     # on a GPU node
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

CHILD = r"""
import sys, numpy as np, jax, jax.numpy as jnp, emu_pk
emu = emu_pk.PkEmulator()
th = jnp.asarray(emu_pk.box.sample(1000, seed=7))
k = jnp.logspace(-4, np.log10(300.0), 200)
zs = jnp.asarray(emu_pk.grid.Z_NODES_EMU)
f = jax.jit(jax.vmap(lambda t: jnp.log(emu.pk(k, zs, t))))
np.save(sys.argv[1], np.asarray(f(th)))
print(jax.devices()[0].device_kind)
"""

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
a = ap.parse_args()
runs = {"cpu": {"JAX_PLATFORMS": "cpu"},
        "gpu_default": {},
        "gpu_highest": {"JAX_DEFAULT_MATMUL_PRECISION": "highest"}}
out, dev = {}, {}
tmp = tempfile.mkdtemp()
for name, env in runs.items():
    e = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    e.pop("JAX_DEFAULT_MATMUL_PRECISION", None)
    e.update(env)
    if name != "cpu":
        e.pop("JAX_PLATFORMS", None)
    r = subprocess.run([sys.executable, "-c", CHILD, os.path.join(tmp, f"lnp_{name}.npy")], env=e,
                       capture_output=True, text=True, check=True)
    dev[name] = r.stdout.strip().splitlines()[-1]
    out[name] = np.load(os.path.join(tmp, f"lnp_{name}.npy"))
res = {"device": dev, "n_cosmologies": 1000, "n_z": 31, "n_k": 200}
for name in ("gpu_default", "gpu_highest"):
    d = np.abs(out[name] - out["cpu"])
    res[name] = {"max_abs_dlnP": float(d.max()), "median_abs_dlnP": float(np.median(d)),
                 "p99_abs_dlnP": float(np.percentile(d, 99))}
print(json.dumps(res, indent=1))
json.dump(res, open(a.json, "w"), indent=1)
