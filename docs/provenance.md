# Provenance of the shipped model

This page lists what the shipped network was built from and how to check it: the weights and their checksum, the training set or the commands that regenerate it, and the validation runs behind the quoted accuracy.

## Version

`emu_pk` 2.1.0, 2.1.1 and 2.1.2 ship the same files under `emu_pk/data/`, byte for byte; 2.1.1 and 2.1.2 changed the documentation only.
Every number on this page holds for all three; the commands pin 2.1.2, the current release.

## Weights and records

| file | SHA-256 |
|---|---|
| `emu_pk_mlp.npz` (the network) | `ad770099f50f1c1b0eb0642c452ea165525aff131d069afe04ae18bf6f899ded` |
| `validation.json` (against the training truth) | `427b4d3972979c9e8ef9739a2b740fa12602aaa440e13da7c2311ae42d37c310` |
| `validation_reference.json` (against converged CAMB) | `ac34a02d224813d6305f0e7d19941afa57ec814ac655d15c8bde8f5e3c7923ea` |
| `class_pk_ratio.npz` (the correction table of `emu_pk.ratio`) | `dac8e9f331eef2081e205ce65fd0465a7e1bc3df2a71351aaa4d464d6f0e1eb7` |

Both validation records carry `weights_sha256`, the checksum of the network they scored, and the test suite fails if it differs from the shipped file.
The network file records its own training: `epoch = 237`, the target and output forms, the input order, the pivot, and `truth_solver = camb` with the CAMB precision settings.

```bash
sha256sum $(python -c "import emu_pk, pathlib; print(pathlib.Path(emu_pk.__file__).parent / 'data')")/*
```

## Training set

The training set is 150 000 points of the Latin hypercube `emu_pk.box.sample(150000, seed=20260827)`, each solved with CAMB 1.6.6 at `lAccuracyBoost = 3`, `AccuracyBoost = 2` and no late radiation truncation, times the reionization heating of a CLASS v3.3.4 pair (see [the design notes](design_notes.md)), at the 31 redshifts of `emu_pk.grid.Z_NODES_EMU`.
CAMB or CLASS refused 25 points, so the set holds 149 975 cosmologies and 4 649 225 rows.
It is 15.5 GB in 33 files, the index `training_set_v21_c.npz` and `training_set_v21_c.part000.npz` to `part031.npz`; their SHA-256 are in [`training_set_v21_c.sha256`](training_set_v21_c.sha256).

The set is regenerated from the seed and the shard index alone, with the commands of [Reproducing the training set](reproducing.md):

```bash
pip install 'emu_pk[gen,train]==2.1.2' camb==1.6.6
python -m emu_pk.generate --mode emu --shard $I --n-per-shard 100 --n-total 150000 --out shards   # I = 0 .. 1499
python -m emu_pk.assemble --mode emu --shards shards --out training_set_v21_c.npz
python -m emu_pk.train --dataset training_set_v21_c.npz --out emu_pk_mlp.npz --epochs 240
```

A regenerated set agrees with the stored rows to 6.6e-7 in $\ln P$ (the float32 the rows are stored in) with CAMB 1.6.6, and to 4.4e-4 with CAMB 2.0.4, so the CAMB version has to be pinned.
Retraining is not bitwise reproducible across hardware; the weights to check against are the shipped ones.

## Validation behind the quoted accuracy

The two records were written by:

```bash
python -m emu_pk.validate --json validation.json                        # training truth
python -m emu_pk.validate --truth reference --json validation_reference.json
```

Against the training truth, the median over 32 test cosmologies at $z = 0$ of the maximum shape error of $P_{\rm m}$ for $10^{-3} < k < 10\,h\,{\rm Mpc}^{-1}$ is 0.064 %, and the largest is 0.177 % (quoted as 0.18 % in Paper I); these are the `total` entries of `validation.json`, tabulated in the [accuracy tutorial](tutorial/02_accuracy.md).
Against the converged CAMB reference, the same statistic is 0.063 % in the median and 0.19 % at most over 32 cosmologies at $z = 0$ (`validation_reference.json`, `shape.m.0.total`).

## Archive

The weights ship in the wheel on PyPI and in the tagged repository (`v2.1.2`).
The training set, with the checksum file above, is on Zenodo: [doi:10.5281/zenodo.24365520](https://doi.org/10.5281/zenodo.24365520).
