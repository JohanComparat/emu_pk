# Linear power-spectrum emulators: accuracy against time

This study compares `emu_pk` with other emulators of the linear matter power spectrum, following Fig. 10 of Sui et al. (2025), and draws Fig. 15 of Paper I, `emulator_benchmark.pdf`.
`benchmark.py` scores each code by the mean of $|P/P_{\rm true} - 1|$ over 200 wavenumbers log-spaced in $[0.009, 9]\,h\,{\rm Mpc}^{-1}$ at $z = 0$, and times one evaluation at $z = 1$, at the four Quijote cosmologies Fid, $w^+$, $w^-$ and $M_\nu^+$.
The two truths are CLASS 3.3.4 at its default precision, set up as in the SYREN-NEW example, and converged CLASS (`pk_ref.pre`, in `class_pk_ref.json`).
The codes, their references and the numbers are in the [README](../../README.md#performance-and-code-comparison).

## Redrawing the figure

`results/` holds the results of the production run, so the figure is redrawn without any solver: the errors and the solver timings from one core of an AMD EPYC 9654 (GRICAD, Kraken), and the GPU timings from an NVIDIA V100 SXM2 (GRICAD, Bigfoot) in full float32.
`runs/` keeps every run of the CPU and GPU stages: also one core of an Intel Xeon Gold 6244 (GRICAD, Dahu) and an NVIDIA A100 PCIe (Bigfoot), with the GPU precision check of each card (`studies/gpu_precision/`).

```bash
python studies/benchmark/benchmark.py plot
```

This writes `emulator_benchmark.pdf`, `emulator_benchmark.png` and `emulator_benchmark.json` (the numbers of the figure) beside the script.

## Rerunning the comparison

```bash
git clone https://github.com/DeaglanBartlett/symbolic_pofk && git -C symbolic_pofk checkout cb4d2bb
export SYMBOLIC_POFK=$PWD/symbolic_pofk
pip install 'emu_pk[gen]' camb==1.6.6 classy==3.3.4 cosmopower-jax matplotlib
python studies/benchmark/benchmark.py truth --cache my_run          # CLASS and CAMB, about 2 min
python studies/benchmark/benchmark.py ref --cache my_run            # converged CLASS, about 11 min
taskset -c 2 python studies/benchmark/benchmark.py cpu --cache my_run   # one core, one thread
python studies/benchmark/benchmark.py gpu --cache my_run            # in a GPU JAX environment
python studies/benchmark/benchmark.py plot --cache my_run
```

The errors do not depend on the machine: the Kraken and Dahu runs reproduce them to every digit. The times do; on one core of the Xeon Gold 6244, `emu_pk` takes 0.43 ms against 0.16 ms on the EPYC 9654, and on the A100, at full float32, a batch of 20 000 takes 0.08 µs per spectrum against 0.25 µs on the V100.
SYREN-NEW runs on the GPU through a JAX transcription of its published NumPy code, made by `benchmark.py` from the source, which reproduces it to $10^{-15}$.
