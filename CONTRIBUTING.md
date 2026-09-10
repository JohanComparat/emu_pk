# Contributing

## Running the tests

```bash
pip install -e '.[dev]'
python -m pytest tests/ -q
```

314 tests, about a minute and a half. Some skip without `ggah_mod`, which is
an optional peer; the tests that need `classy` are marked `slow` and skip
without it.

```bash
python -m pytest tests/ -q --cov=emu_pk --cov-report=term-missing
python -m pytest tests/ -q -m slow          # needs pip install -e '.[gen]'
```

## The extras split

`import emu_pk` in an environment with no `classy` and no `optax` must work, so
that a package depending on this one does not inherit a Boltzmann solver or a
training stack. The test suite asserts it, and CI fails on a top-level import
of either.

| extra | adds | needed for |
|---|---|---|
| — | numpy, jax | evaluating a trained network |
| `[gen]` | classy | generating data, validating |
| `[train]` | optax | training |
| `[dev]` | pytest | the test suite |
| `[docs]` | sphinx | the documentation |

## Building the documentation

```bash
pip install -e '.[docs]'
python -m sphinx -b html docs docs/_build/html -W
```

`-W` because that is what ReadTheDocs does: a warning is a broken
cross-reference, and the published site should not have one.

**Figures are committed artefacts.** ReadTheDocs installs the core package
only and cannot run CLASS, so `docs/make_figures.py` is run locally and its
output committed. Regenerate after anything that changes the weights:

```bash
python docs/make_figures.py            # needs [gen] for the CLASS comparison
python docs/make_figures.py --fast     # skips it
```

## Things worth knowing before changing something

- **The parameter order lives in `box.PARAMS`** and is read from there by
  everything downstream. A silently permuted column trains perfectly well and
  predicts nonsense.
- **Checkpoints declare their own format** — `target_form`, `output_form`,
  `z_var`, `loss_form` — and `PkEmulator` reads them rather than inferring.
  A file loaded under the wrong assumption returns a spectrum, not an error.
- **`validation.json` must describe the weights beside it.** A test asserts it.
  Reship weights without re-running `python -m emu_pk.validate` and the numbers
  keep looking current while describing a different network.
- **The box check is skipped under `jax.jit` tracing**, deliberately.
  Attempting it there raises `ConcretizationTypeError` and breaks the gradient
  the package exists to provide.
- **`bash -n a.sh b.sh` parses only `a.sh`.** If you check shell scripts, one
  file per invocation. There is a test for this.
- **The trainer's defaults are the shipped configuration.** `train()` with no
  flags must build the model in `emu_pk/data`. Tests read the checkpoint and
  assert the signature agrees; they exist because the two diverged once and a
  whole campaign trained the wrong network.
- **A metric reports its own floor.** `shape_error` interpolates the network
  onto the scoring wavenumbers, `derivative_error` takes finite differences of
  CLASS, and both cost accuracy. Quote the floor beside the number it bounds.
- **The validation split holds out whole cosmologies.** Rows are
  `(cosmology, redshift)` pairs, so a row split leaks 31 redshifts of each
  cosmology into both halves and `val_loss` stops measuring generalisation.
- **OAR does not propagate the environment to the node.** Anything a job needs
  in order to know which work it is doing travels as an argument. Tests read
  `submit_campaign.sh` and check this.

More of this kind of thing, with the reasons, is in `docs/design_notes.md`.

## Releasing

1. Update `CHANGELOG.md`.
2. Bump the version in **three** places — `pyproject.toml`,
   `emu_pk/__init__.py`, `CITATION.cff`. CI checks they agree.
3. If the weights changed, re-run `python -m emu_pk.validate --json
   emu_pk/data/validation.json` **locally**, with no `--weights`. Run on the
   cluster it records the path it scored, and the shipped record must say
   `"shipped"`; a test asserts it.
4. Update every number quoted from that file. They are listed in
   `RELEASE_TODO.md`, and marked in the source with `NUMBERS-PENDING`.
5. Regenerate the figures.
6. Tag, and let the publish workflow build and upload.
