# Release checklist for 2.1.0

**Delete this file in the release commit.** It exists because the 2.1 weights
and `validation_reference.json` are in, while `validation.json` -- the network
against its own training truth, with the derivatives -- is still being
regenerated, so the numbers quoted from it are 2.0's or blank.

## The blocker

`emu_pk/data/validation.json`, regenerated locally against the shipped weights,
with no `--weights`, on **CAMB 1.6.6** and CLASS 3.3.4 (the `jaxgpu` env), so
the truth is the one the network was trained on:

```bash
JAX_PLATFORMS=cpu python -m emu_pk.validate --cache work/v21/truth_cache_train \
    --json emu_pk/data/validation.json
```

About 840 solves. They are being filled in parallel into that cache first
(`work/v21/prefill_training.py`, 24 workers), so the run above only reads.
`tests/test_model.py::...::test_the_validation_scored_these_bytes` fails until
then, by design.

## Where the numbers are quoted

Marked in the source with `<!-- NUMBERS-PENDING -->` where they are blank.

| file | what |
|---|---|
| `README.md` | the training-truth table (amplitude, shape, total, floor), the derivative table, the $\partial\ln P/\partial z$ table; the coverage sentence under Tests |
| `CHANGELOG.md` | the training-truth median in the 2.1.0 summary |
| `docs/index.md` | the training-truth median and the derivative range in the opening |
| `docs/tutorial/02_accuracy.md` | everything under "The training truth": amplitude across $z$, shape, floor, residuals by $k$, the lowest decade, derivatives with floors, redshift dependence, $\partial\ln P/\partial z$, strata, flat slice |
| `docs/tutorial/03_derivatives.md` | `w0` and `wa` from $z = 0$ to 5 in "A caution" |
| `docs/design_notes.md` | "where the emulator's own shape error is 0.064 %" (the correction table) |

The converged-reference numbers are final: they come from
`validation_reference.json`, which is in.

## Also outstanding

- **Figures.** `python docs/make_figures.py` in the `jaxgpu` env with
  `JAX_PLATFORMS=cpu`, after `validation.json`: thirteen CAMB solves.
- **Coverage.** Re-measure before restating "every statement, 99 % of branches".
- **`CITATION.cff` `date-released`**, at the tag.
