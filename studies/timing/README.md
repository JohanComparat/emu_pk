# Timings

This study measures how long the shipped emulator takes to give a spectrum and its Jacobian, and how much memory the Jacobian needs.
`timing.py` writes `timing_cpu.json` and `timing_gpu.json`; the tables below are taken from them.

## Set-up

- Machine: a laptop with an Intel Core Ultra 9 285HX (24 cores) and an NVIDIA RTX PRO 4000 Blackwell Laptop GPU; JAX 0.11.2 on the GPU.
- Input: 200 wavenumbers in $10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}$, one cosmology, either $z = 0$ or the 31 training redshifts; the Jacobian is $\partial P_{\rm m}(k,z)/\partial\theta$ over all eleven box parameters (the paper's ten plus `nu_r2`).
- Every timed call ends in `jax.block_until_ready`.
- Cold: a new Python process, from `import emu_pk` to the first spectrum ready, five times.
- Compile: `jit(f).lower(x).compile()`; warm: the median of 50 calls of the compiled function.
- Memory: XLA's accounting of the compiled program (`memory_analysis().temp_size_in_bytes`), the scratch space beyond inputs and outputs.
- Each function is timed in its own process.

```bash
JAX_PLATFORMS=cpu python studies/timing/timing.py --json studies/timing/timing_cpu.json
python studies/timing/timing.py --json studies/timing/timing_gpu.json      # in a GPU jax environment
```

## Results

Cold start, import to first spectrum: 5.1 s on the CPU (1.9 s to 5.7 s) and 8.5 s on the GPU (6.3 s to 11.1 s), most of it importing JAX and compiling.

| function | CPU compile | CPU warm | CPU temp | GPU compile | GPU warm | GPU temp |
|---|---|---|---|---|---|---|
| $P_{\rm m}$, one $z$ | 0.27 s | 0.36 ms | 0.006 MB | 1.3 s | 0.30 ms | 0.02 MB |
| $P_{\rm m}$, 31 $z$ | 0.33 s | 3.1 ms | 0.17 MB | 7.7 s | 0.17 ms | 0.57 MB |
| $P_{\rm m}$ and $P_{\rm cb}$, 31 $z$ | 0.33 s | 2.6 ms | 0.17 MB | 6.8 s | 0.18 ms | 0.57 MB |
| `jacfwd`, one $z$ | 0.44 s | 1.1 ms | 0.07 MB | 6.9 s | 0.22 ms | 0.06 MB |
| `jacrev`, one $z$ | 0.25 s | 5.0 ms | 1.6 MB | 10.8 s | 0.15 ms | 4.0 MB |
| `jacfwd`, 31 $z$ | 0.25 s | 11 ms | 1.9 MB | 11.1 s | 0.14 ms | 3.8 MB |
| `jacrev`, 31 $z$ | 0.79 s | 4 200 ms | 1 400 MB | — | out of memory | — |
| `grad` of a scalar, 31 $z$ | 0.35 s | 2.9 ms | 0.80 MB | — | failed, see below | — |
| $P_{\rm m}$, 1 000 cosmologies, one $z$ | 0.36 s | 41 ms | 6.1 MB | 11.6 s | 1.0 ms | 5.3 MB |
| `jacfwd`, 1 000 cosmologies, one $z$ | 0.45 s | 156 ms | 68 MB | 13.6 s | 1.1 ms | 74 MB |

The full Jacobian is the job of `jacfwd`: eleven inputs against 200 to 6 200 outputs, so forward mode makes eleven passes where reverse mode makes one per output.
At one redshift both are fast; over 31 redshifts `jacrev` takes 4.2 s and 1.4 GB of scratch on the CPU against 11 ms and 1.9 MB for `jacfwd`, and does not fit on the 16 GB GPU.
The cost of `jacrev` there comes from reversing the cubic interpolation in $k$, a gather whose transpose is a scatter per output row.
Reverse mode is the right choice for the gradient of a scalar, a likelihood, which costs 2.9 ms on the CPU over 31 redshifts.

On the GPU the warm call is 0.1 to 0.3 ms whatever the size, the launch overhead, and compilation takes 1 to 14 s.
The GPU `grad` run failed at instantiation with `RESOURCE_EXHAUSTED` in the CUDA command buffer, in its own process right after the `jacrev` run; we did not investigate it further.
The CPU warm times vary by a factor of 3 between runs on this laptop (0.08 ms to 0.36 ms for one spectrum), with the clock throttling under load; take them as an order of magnitude.
