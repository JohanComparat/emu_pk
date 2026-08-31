# Installation

```bash
pip install emu_pk
```

That is the whole thing for *using* an emulator: `numpy`, `jax` and `jaxlib`,
all of which have wheels on every supported platform. No conda environment is
needed, and neither is a compiler.

## The extras

| | Installs | For |
|---|---|---|
| `emu_pk` | numpy, jax | evaluating a trained network and the correction table |
| `emu_pk[gen]` | + `classy` | generating training data, and **validating** |
| `emu_pk[train]` | + `optax` | training a network |
| `emu_pk[dev]` | + pytest | running the test suite |
| `emu_pk[docs]` | + sphinx | building this documentation |

**The split is load-bearing.** `import emu_pk` in an environment with no
`classy` and no `optax` must work, and the test suite asserts it. That is what
lets another package depend on this one without inheriting a Boltzmann solver
or a training stack.

## A dedicated conda environment

The repository carries an `environment.yml` for a minimal environment — python,
numpy and JAX, and nothing else:

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
mamba activate emu_pk
pip install -e .
```

It pins the **CPU** build of `jaxlib`, which is 64 MB against 199 MB for the
CUDA one; left unpinned, the build you get depends on whether the machine that
solved the environment had a driver. The file says how to swap it for a GPU,
and carries a commented block per extra.

## A note on `[gen]`

`classy` compiles CLASS from source, so it needs a C compiler. It is also the
one part of this that can fail for environmental reasons: CLASS's `setup.py`
invokes `make` with an unbounded `-j`, which on a memory-capped machine gets
the compiler killed —

```
g++: fatal error: Killed signal terminated program cc1plus
```

If you hit that, build on a machine without a per-user memory cap, or install
`classy` separately with a bounded parallelism. Note that **training does not
need CLASS**; only generation and validation do.

## Supported versions

Python 3.11 and newer. `emu_pk` ships inline type information (`py.typed`).

## Verifying the install

```bash
python -c "from emu_pk import PkEmulator; print(PkEmulator().pk([0.1], 0.0, \
    [0.02237, 0.12, 0.6736, 0.9649, 3.044, 0.06, -1.0, 0.0]))"
```

The trained weights and the correction table ship inside the package — about
17 MB — so there is no download step and the package works offline.
