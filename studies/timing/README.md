# Timings

This study measures how long the shipped emulator, `emu_pk` 2.1.2, takes to give a spectrum and its Jacobian, and how much memory the Jacobian needs, on the CPU and GPU nodes of three computing centres.
`timing.py` writes one JSON file per node into `clusters/`, with an environment file beside it (host, processor, JAX version); the tables below are taken from them.

## Set-up

- Input: 200 wavenumbers in $10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}$, one cosmology, either $z = 0$ or the 31 training redshifts; the Jacobian is $\partial P_{\rm m}(k,z)/\partial\theta$ over all eleven box parameters (the paper's ten plus `nu_r2`).
- Every timed call ends in `jax.block_until_ready`.
- Cold: a new Python process, from `import emu_pk` to the first spectrum ready, five times.
- Compile: `jit(f).lower(x).compile()`; warm: the median of 20 calls of the compiled function.
- Memory: XLA's accounting of the compiled program (`memory_analysis().temp_size_in_bytes`), the scratch space beyond inputs and outputs.
- Each function is timed in its own process, in a batch job on a node shared with other users' jobs; `XLA_PYTHON_CLIENT_PREALLOCATE=false` on the GPU.

```bash
JAX_PLATFORMS=cpu python studies/timing/timing.py --repeat 20 --json clusters/timing_<node>.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python studies/timing/timing.py --repeat 20 --json clusters/timing_<node>.json   # GPU
```

## Hardware

| processor | centre | architecture | base / boost clock | vector unit | L3 per socket | allocation |
|---|---|---|---|---|---|---|
| Intel Xeon Gold 5318Y | GRICAD, Dahu | Ice Lake (2021) | 2.1 / 3.4 GHz | AVX-512 | 36 MB | 16 of 48 cores |
| AMD EPYC 9654 | GRICAD, Kraken | Zen 4 (2022) | 2.4 / 3.7 GHz | AVX-512 | 384 MB | 16 of 192 cores, no SMT |
| AMD EPYC 7453 | CC-IN2P3, `htc` | Zen 3 (2021) | 2.75 / 3.45 GHz | AVX2 | 64 MB | 32 of 112 hardware threads (16 cores) |
| NVIDIA V100 SXM2 32 GB | GRICAD, Bigfoot | Volta (2017) | 15.7 TFLOP/s FP32 | — | 900 GB/s HBM2 | one GPU, 10 CPU cores |
| NVIDIA V100 PCIe 32 GB | CC-IN2P3, `gpu_v100` | Volta (2017) | 14 TFLOP/s FP32 | — | 900 GB/s HBM2 | one GPU, 5 CPU cores |

The core counts, base clocks and allocations are those the batch systems report for the nodes that ran the jobs; the boost clocks, vector units and caches are the vendors' specifications.
JAX is 0.10.2 at GRICAD and 0.11.2 at CC-IN2P3.

## CPU

Warm call, median, in ms; the 10th and 90th percentiles lie within 15 % of the median on every node.

| function | Xeon Gold 5318Y | EPYC 9654 | EPYC 7453 |
|---|---|---|---|
| $P_{\rm m}$, one $z$ | 0.49 | 0.082 | 0.17 |
| $P_{\rm m}$, 31 $z$ | 1.0 | 0.39 | 0.95 |
| $P_{\rm m}$ and $P_{\rm cb}$, 31 $z$ | 1.3 | 0.39 | 0.98 |
| `jacfwd`, one $z$ | 0.56 | 0.24 | 0.51 |
| `jacfwd`, 31 $z$ | 3.0 | 1.6 | 3.4 |
| `jacrev`, 31 $z$ | 1 150 | 980 | 1 330 |
| `grad` of a scalar, 31 $z$ | 2.7 | 0.89 | 2.5 |
| $P_{\rm m}$, 1 000 cosmologies, one $z$ | 7.0 | 3.0 | 7.6 |
| `jacfwd`, 1 000 cosmologies, one $z$ | 28 | 29 | 48 |

Cold start, import to the first spectrum, is 7.9 s on the Xeon, 2.8 s on the EPYC 9654 and 4.2 s on the EPYC 7453 (median of five), most of it importing JAX from a network file system and compiling; compilation alone takes 0.1 to 0.4 s per function.
The scratch memory of each program is the same on every node, to 10 %: 1.9 MB for the `jacfwd` Jacobian over 31 redshifts and 1.4 GB for the `jacrev` one.

The differences follow two regimes.
One spectrum is latency-bound: a few small matrix products and an interpolation, too little work to spread over cores.
Its cost follows the speed of one core, and does not change with the allocation: on the EPYC 7453 it is 0.15 ms with 16 hardware threads and 0.17 ms with 32.
The EPYC 9654 has the newest cores and the highest boost clock, and is fastest at 0.08 ms; the Xeon ran at 60 % of its maximum clock when the job started, which is part of its factor of six.
A batch of 1 000 cosmologies is throughput-bound and scales with the cores and their vector units.
On the EPYC 7453, 16 hardware threads, about 8 physical cores, take 30 ms and 32 threads take 7.6 ms; at 16 physical cores the Xeon and the EPYC 7453 agree, at 7.0 and 7.6 ms, and the EPYC 9654, with AVX-512 and the newer core, is 2.5 times faster.
We do not separate the core generation, the vector unit and the memory system further; the node allocations differ in cores (Kraken and Dahu by core, CC-IN2P3 by hardware thread), so the comparison is at 16 physical cores throughout.

On every node the order of the functions is the same.
The full Jacobian is the job of `jacfwd`: eleven inputs against 200 to 6 200 outputs, so forward mode makes eleven passes where reverse mode makes one per output.
Over 31 redshifts `jacrev` is 300 to 600 times slower than `jacfwd` and needs 1.4 GB of scratch against 1.9 MB.
Its cost comes from reversing the cubic interpolation in $k$, a gather whose transpose is a scatter per output row.
Reverse mode is the right choice for the gradient of a scalar, a likelihood: 0.9 to 2.7 ms over 31 redshifts.

## GPU

Warm call, median, in ms:

| function | V100 SXM2 | V100 PCIe |
|---|---|---|
| $P_{\rm m}$, one $z$ | 0.096 | 0.085 |
| $P_{\rm m}$, 31 $z$ | 0.19 | 0.13 |
| `jacfwd`, one $z$ | 0.15 | 0.16 |
| `jacfwd`, 31 $z$ | 0.26 | 0.25 |
| `jacrev`, 31 $z$ | 136 | 123 |
| `grad` of a scalar, 31 $z$ | 0.20 | 0.22 |
| $P_{\rm m}$, 1 000 cosmologies, one $z$ | 0.43 | 0.38 |
| `jacfwd`, 1 000 cosmologies, one $z$ | 3.0 | 2.7 |

Cold start is 5.1 s and 3.6 s, and compilation 0.3 to 0.9 s per function.
The single calls cost 0.1 to 0.3 ms whatever their size, the launch overhead of the GPU; the two V100 agree to 30 %, so the SXM2 and PCIe forms make no difference at this size.
The batch of 1 000 cosmologies is where the GPU pays: 0.4 ms against 3 to 8 ms on 16 CPU cores, and 3 ms for its Jacobian against 28 to 48 ms.
`jacrev` over 31 redshifts takes 0.12 to 0.14 s and 1.1 to 1.5 GB, so it fits, but it stays 500 times slower than `jacfwd`.
On a GPU shared with other processes, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`: JAX otherwise reserves 75 % of the memory at start-up, and the larger programs fail with `RESOURCE_EXHAUSTED` when another process holds the rest.
