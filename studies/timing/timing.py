"""Timings of the shipped emulator: cold, JIT compile and warm, and the Jacobian.

Every timed call ends in ``block_until_ready``: JAX dispatches asynchronously,
and a clock stopped before the result exists measures the dispatch.

* cold: a new process, from ``import emu_pk`` to the first spectrum on the
  device -- loading the weights, tracing, compiling and running once.
* compile: ``jit(f).lower(...).compile()`` alone, for each function.
* warm: the median of repeated calls of the compiled function.
* Jacobian: d P_m(k, z) / d theta over all box parameters, with ``jacfwd`` and
  ``jacrev``; memory is XLA's own accounting of the compiled program
  (``memory_analysis``: arguments, outputs and temporaries).
"""
import argparse
import json
import platform
import subprocess
import sys
import time

COLD = r"""
import time; t0 = time.perf_counter()
import jax, jax.numpy as jnp, numpy as np
import emu_pk
emu = emu_pk.PkEmulator()
k = jnp.logspace(-3, 1, 200)
p = jnp.array([0.0224, 0.12, 0.674, 0.965, 3.044, 0.06, -1.0, 0.0, 0.0, 1/3, 1/3])
jax.block_until_ready(emu.pk(k, 0.0, p))
print(time.perf_counter() - t0)
"""

ap = argparse.ArgumentParser()
ap.add_argument("--repeat", type=int, default=200)
ap.add_argument("--n-k", type=int, default=200)
ap.add_argument("--json", default=None)
ap.add_argument("--only", default=None, help="time one function, in this process")
a = ap.parse_args()

cold = [0.0] * 5 if a.only else sorted(float(subprocess.run([sys.executable, "-c", COLD], capture_output=True,
                                   text=True, check=True).stdout.split()[-1])
              for _ in range(5))

import jax
import jax.numpy as jnp
import numpy as np

import emu_pk

emu = emu_pk.PkEmulator()
k = jnp.logspace(-3, 1, a.n_k)
p = jnp.array([0.0224, 0.12, 0.674, 0.965, 3.044, 0.06, -1.0, 0.0, 0.0, 1 / 3, 1 / 3])
zs = jnp.asarray(emu_pk.grid.Z_NODES_EMU)            # 31 redshifts

funcs = {
    "pk_1z": (lambda p: emu.pk(k, 0.0, p), p),
    "pk_31z": (lambda p: emu.pk(k, zs, p), p),
    "pk_and_pkcb_31z": (lambda p: (emu.pk(k, zs, p), emu.pk_cb(k, zs, p)), p),
    "jacfwd_1z": (jax.jacfwd(lambda p: emu.pk(k, 0.0, p)), p),
    "jacrev_1z": (jax.jacrev(lambda p: emu.pk(k, 0.0, p)), p),
    "jacfwd_31z": (jax.jacfwd(lambda p: emu.pk(k, zs, p)), p),
    "jacrev_31z": (jax.jacrev(lambda p: emu.pk(k, zs, p)), p),
    "grad_scalar_31z": (jax.grad(lambda p: jnp.sum(jnp.log(emu.pk(k, zs, p)))), p),
}
# a batch: 1000 cosmologies at one redshift, as a sampler's vectorised call
theta = jnp.asarray(emu_pk.box.sample(1000, seed=3))
funcs["pk_batch1000_1z"] = (jax.vmap(lambda p: emu.pk(k, 0.0, p)), theta)
funcs["jacfwd_batch1000_1z"] = (jax.vmap(jax.jacfwd(lambda p: emu.pk(k, 0.0, p))), theta)

if not a.only:
    # each function in a fresh process, so no compiled program or thread pool
    # left by one is still resident while the next is timed
    res = {}
    for name in funcs:
        r = subprocess.run([sys.executable, __file__, "--only", name, "--repeat",
                            str(a.repeat), "--n-k", str(a.n_k)],
                           capture_output=True, text=True, timeout=600)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        res[name] = json.loads(line[-1]) if line else {"error": r.stderr[-500:]}
        print(f"{name:24s}", json.dumps(res[name]), flush=True)
    funcs = {}
res = res if not a.only else {}
for name, (f, x) in funcs.items():
    if name != a.only:
        continue
    t0 = time.perf_counter()
    lowered = jax.jit(f).lower(x)
    t1 = time.perf_counter()
    comp = lowered.compile()
    t2 = time.perf_counter()
    jax.block_until_ready(comp(x))
    ts = []
    for _ in range(a.repeat):
        s = time.perf_counter()
        jax.block_until_ready(comp(x))
        ts.append(time.perf_counter() - s)
    ma = comp.memory_analysis()
    mem = None
    if ma is not None:
        mem = {"argument_MB": ma.argument_size_in_bytes / 1e6,
               "output_MB": ma.output_size_in_bytes / 1e6,
               "temp_MB": ma.temp_size_in_bytes / 1e6,
               "generated_code_MB": ma.generated_code_size_in_bytes / 1e6}
    res[name] = {"trace_s": t1 - t0, "compile_s": t2 - t1,
                 "warm_median_ms": 1e3 * float(np.median(ts)),
                 "warm_p10_ms": 1e3 * float(np.percentile(ts, 10)),
                 "warm_p90_ms": 1e3 * float(np.percentile(ts, 90)), "memory": mem}
    print(json.dumps(res[name]))
    sys.exit(0)
    print(f"{name:24s} trace {t1-t0:6.2f}s compile {t2-t1:6.2f}s "
          f"warm {res[name]['warm_median_ms']:9.3f} ms  "
          f"temp {mem['temp_MB'] if mem else float('nan'):8.2f} MB", flush=True)

dev = jax.devices()[0]
out = {"device": f"{dev.platform}: {getattr(dev, 'device_kind', '')}",
       "cpu": platform.processor() or platform.machine(),
       "jax": jax.__version__, "emu_pk": emu_pk.__version__,
       "n_k": a.n_k, "n_params": len(emu_pk.box.PARAMS), "repeat": a.repeat,
       "cold_s": {"median": cold[2], "min": cold[0], "max": cold[-1]},
       "functions": res}
print(json.dumps(out["cold_s"]), out["device"])
if a.json:
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)
