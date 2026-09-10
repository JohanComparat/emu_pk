# Release checklist for 2.0.0

**Delete this file in the release commit.** It exists because the weights
changed before `validation.json` could be regenerated, so the numbers quoted
across the documentation still describe the 1.0.0 network.

## The blocker

`emu_pk/data/validation.json` must be regenerated **locally**, against the
shipped weights, with no `--weights`:

```bash
python -m emu_pk.validate --json emu_pk/data/validation.json
```

Run on the cluster it records the path it scored; the shipped record has to say
`"weights": "shipped"`, and `tests/test_model.py` asserts it. About 850 CLASS
solves.

## Where the numbers are quoted

Every site below carries a 1.0.0 number. Those inside a table are marked in the
source with `<!-- NUMBERS-PENDING -->`.

| file | lines | what |
|---|---|---|
| `README.md` | 60–64 | amplitude / shape / total table, $z = 0$ |
| `README.md` | 79–82 | per-parameter derivative table |
| `README.md` | 88–89 | $\partial\ln P/\partial z$ and its floor |
| `README.md` | 139 | the shape error quoted in the neutrino paragraph |
| `docs/index.md` | 7–8 | shape, amplitude and total in the opening |
| `docs/index.md` | 18 | derivative row |
| `docs/index.md` | 29 | redshift-derivative row |
| `docs/tutorial/01_spectrum.md` | 38 | shape error in prose |
| `docs/tutorial/02_accuracy.md` | 21–29 | amplitude, and its run across $z$ |
| `docs/tutorial/02_accuracy.md` | 41–43, 51–56 | the three-quantity table and the text around it |
| `docs/tutorial/02_accuracy.md` | 94–97 | per-parameter derivatives with floors |
| `docs/tutorial/02_accuracy.md` | 109–110, 121–125 | redshift derivative, floors, closing |
| `docs/design_notes.md` | 68 | "the emulator's own shape error is 0.16 %" |
| `docs/reproducing.md` | 35 | "the one that scored 0.111 %" |

Three new axes have no derivative row anywhere yet: `Omega_k`, `nu_r1`,
`nu_r2`. The tables in `README.md` and `docs/tutorial/02_accuracy.md` need
three rows added, not just numbers replaced.

`docs/index.md` and `docs/tutorial/02_accuracy.md` also quote CosmoPower's
0.159 %, which does not change.

## Also outstanding

- **Figures.** `docs/_static/figures/*.png` are committed artefacts and still
  show the eight-parameter box. `docs/make_figures.py` sizes its panel grid
  from `box.PARAMS` now, so it only needs running:
  ```bash
  python docs/make_figures.py          # needs [gen]
  ```
- **The shape metric's floor** is a new number with no home in the docs yet.
  `validation.json` carries it as `shape_floor`.
- **New strata** in `validation.json` that nothing quotes: `light_nu`,
  `heavy_nu`, `degenerate_nu`, `curved`, `negative_de`, and `shape_lowk`.
